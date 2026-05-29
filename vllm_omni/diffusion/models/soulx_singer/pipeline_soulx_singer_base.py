import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin, _unwrap, _wrap
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import (
    SupportAudioInput,
    SupportAudioOutput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.soulx_singer.preprocess.pre_process import build_metadata_processor
from vllm_omni.diffusion.models.soulx_singer.utils import _patch_torchaudio_load, load_config
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

PROJECT_ROOT = Path(__file__).parents[4]
_SOULX_EXAMPLE_AUDIO_DIR = PROJECT_ROOT / "tests" / "assets" / "soulxsinger"

_SOULX_PROFILER_TARGETS: ClassVar[list[str]] = [
    "forward",
]


def _resolve_phoneset_path(model_dir: str) -> str:
    """Resolve phoneme vocabulary; prefer model checkpoint, fall back to bundled fixtures."""
    candidates = (
        Path(model_dir) / "phoneme" / "phone_set.json",
        Path(model_dir) / "phone_set.json",
        _SOULX_EXAMPLE_AUDIO_DIR / "phoneme" / "phone_set.json",
    )
    for path in candidates:
        if path.is_file():
            return str(path)
    raise FileNotFoundError(
        "SoulX-Singer phoneset not found. Expected one of: "
        f"{[str(p) for p in candidates]}. "
        "Copy phoneme/phone_set.json into the model directory or use bundled test assets."
    )


def get_soulxsinger_post_process_func(od_config: OmniDiffusionConfig):
    """Convert pipeline audio tensor output for offline consumers."""
    del od_config

    def post_process_func(
        audio: torch.Tensor,
        output_type: str = "np",
    ):
        if output_type in ("latent", "pt"):
            return audio
        return audio.detach().cpu().float().numpy()

    return post_process_func


class FlowMatchingAudioPipeline(
    nn.Module,
    CFGParallelMixin,
    SupportAudioInput,
    SupportAudioOutput,
    SupportsComponentDiscovery,
    DiffusionPipelineProfilerMixin,
):
    """SoulX flow-matching pipeline with CFG / CFG-parallel support."""

    support_audio_input: ClassVar[bool] = True
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 24000
    _DEFAULT_RESCALE_CFG: ClassVar[float] = 0.75
    _dit_modules: ClassVar[list[str]] = ["cfm_decoder.model.diff_estimator"]
    _encoder_modules: ClassVar[list[str]] = ["mel", "f0_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vocoder"]
    _resident_modules: ClassVar[list[str]] = ["cfm_decoder.model.cond_emb"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()

        # 1. parse model path and yaml config
        model_dir = od_config.model
        assert model_dir is not None, "model_dir is not provided"

        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"""Model directory {model_dir} does not exist. User should refer to
                `examples/offline_inference/text_to_speech/README.md` for usage details."""
            )

        # Empty: skip DiffusersPipelineLoader.get_all_weights() → no _prepare_weights / file glob /
        # safetensors or pt iterator from the framework. Load checkpoints in __init__ instead.
        self.weights_sources = ()

        self.model_path = model_dir

        config_path = os.path.join(model_dir, "config.yaml")
        hf_config = load_config(config_path)

        self.audio_config = hf_config.audio
        self.encoder_config = hf_config.model.encoder
        self.flow_matching_config = hf_config.model.flow_matching

        self.mel_dim = self.audio_config.num_mels

        self.f0_bin = self.encoder_config.f0_bin
        self.f0_dim = self.encoder_config.f0_dim
        self.text_dim = self.encoder_config.text_dim
        self.vocab_size = self.encoder_config.vocab_size
        self.pitch_dim = self.encoder_config.pitch_dim
        self.type_dim = self.encoder_config.type_dim

        self.metadata_processor = build_metadata_processor(od_config)

        _patch_torchaudio_load()

    def _mel_to_audio(self, generated_mel: torch.Tensor, *, squeeze: bool = False) -> torch.Tensor:
        audio = self.vocoder(generated_mel.transpose(1, 2)[0:1, ...].float()).float()
        return audio.squeeze() if squeeze else audio

    def _setup_soulx_profiler(self, *, extra_targets: list[str] | None = None) -> None:
        """Enable stage timing when ``od_config.enable_diffusion_pipeline_profiler`` is set."""
        targets = list(_SOULX_PROFILER_TARGETS)
        if extra_targets:
            targets.extend(extra_targets)
        self.setup_diffusion_pipeline_profiler(
            profiler_targets=targets,
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler,
        )

    @contextmanager
    def _stage_timer(self, name: str):
        if not getattr(self, "enable_diffusion_pipeline_profiler", False):
            yield
            return
        if current_omni_platform.is_available():
            current_omni_platform.synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            if current_omni_platform.is_available():
                current_omni_platform.synchronize()
            duration = time.perf_counter() - start
            with self._profiler_lock:
                self._stage_durations[name] = self._stage_durations.get(name, 0.0) + duration

    def _profiler_stage_durations(self) -> dict[str, float] | None:
        if getattr(self, "enable_diffusion_pipeline_profiler", False):
            return self.stage_durations
        return None

    @staticmethod
    def _build_fp32_audio_modules(audio_config) -> tuple[nn.Module, nn.Module]:
        """Mel/vocoder must stay FP32 even when ``od_config.dtype`` is FP16/BF16."""
        from vllm_omni.diffusion.models.soulx_singer.modules import MelSpectrogramEncoder, Vocoder

        with set_default_torch_dtype(torch.float32):
            mel = MelSpectrogramEncoder(audio_config)
            vocoder = Vocoder()
        return mel, vocoder

    def _finalize_loaded_dtypes(self) -> None:
        """Keep mel/vocoder in FP32; cast DiT trunk to ``od_config.dtype`` after load."""
        self.mel.float()
        self.vocoder.float()
        trunk_dtype = self.od_config.dtype
        if trunk_dtype not in (torch.float16, torch.bfloat16):
            return
        trunk_modules = (
            "cfm_decoder",
            "f0_encoder",
            "preflow",
            "note_text_encoder",
            "note_pitch_encoder",
            "note_type_encoder",
        )
        for name in trunk_modules:
            module = getattr(self, name, None)
            if module is not None:
                module.to(dtype=trunk_dtype)
        whisper = getattr(self, "whisper_encoder", None)
        if whisper is not None:
            whisper.float()

    @property
    def transformer(self) -> nn.Module:
        """Alias expected by ``CFGParallelMixin`` default hooks."""
        return self.cfm_decoder.model.diff_estimator

    @property
    def diffusion_trunk_dtype(self) -> torch.dtype:
        """DiT trunk dtype from ``diff_estimator`` (not ``cond_emb``, which is registered first)."""
        return next(self.cfm_decoder.model.diff_estimator.parameters()).dtype

    def _to_trunk_dtype(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Cast tensors to ``diffusion_trunk_dtype`` at the CFM/DiT boundary."""
        trunk_dtype = self.diffusion_trunk_dtype
        return tuple(t if t.dtype == trunk_dtype else t.to(dtype=trunk_dtype) for t in tensors)

    def _resolve_diffusion_generator(self, sampling_params: Any) -> torch.Generator | None:
        """Build a device-local RNG for CFM noise from ``generator`` or ``seed``."""
        generator = getattr(sampling_params, "generator", None)
        if generator is not None:
            return generator
        seed = getattr(sampling_params, "seed", None)
        if seed is None:
            return None
        return torch.Generator(device=self.device).manual_seed(int(seed))

    def _prepare_cfm_loop_state(
        self,
        prompt: torch.Tensor,
        cond: torch.Tensor,
        n_timesteps: int,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Cast FP32 islands to trunk dtype and build CFM loop tensors/masks."""
        trunk_dtype = self.diffusion_trunk_dtype
        prompt, cond = self._to_trunk_dtype(prompt, cond)
        cond_emb = self.cfm_decoder.model.cond_emb(cond)
        if cond_emb.dtype != trunk_dtype:
            cond_emb = self._to_trunk_dtype(cond_emb)[0]
        device = cond_emb.device

        h = 1.0 / n_timesteps
        h_tensor = torch.tensor(h, device=device, dtype=trunk_dtype)
        prompt_len = prompt.shape[1]
        target_len = cond_emb.shape[1] - prompt_len

        x_mask = torch.ones(cond_emb.shape[0], target_len, device=device, dtype=trunk_dtype)
        prompt_mask = torch.ones(cond_emb.shape[0], prompt_len, device=device, dtype=trunk_dtype)
        xt_mask = torch.cat([prompt_mask, x_mask], dim=1)
        xt = torch.randn(
            cond_emb.shape[0],
            target_len,
            self.mel_dim,
            dtype=trunk_dtype,
            device=device,
            generator=generator,
        )
        return cond_emb, prompt, xt, x_mask, xt_mask, h_tensor, prompt_len

    def predict_noise(
        self,
        *,
        x: torch.Tensor,
        diffusion_step: torch.Tensor,
        cond: torch.Tensor,
        x_mask: torch.Tensor,
        prompt_len: int | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Forward through ``diff_estimator``, optionally slice off prompt frames."""
        x, diffusion_step, cond, x_mask = self._to_trunk_dtype(x, diffusion_step, cond, x_mask)
        flow_pred = self.cfm_decoder.model.diff_estimator(x, diffusion_step, cond, x_mask)
        if prompt_len is not None:
            flow_pred = flow_pred[:, prompt_len:, :]
        return flow_pred

    def combine_cfg_noise(
        self,
        positive_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        negative_noise_pred: torch.Tensor | tuple[torch.Tensor, ...],
        true_cfg_scale: float,
        cfg_normalize: bool = False,
        rescale_cfg: float | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """SoulX CFG: pos + scale * (pos - neg) with std rescaling."""
        del cfg_normalize  # SoulX uses rescale_cfg instead of cfg_normalize_function.
        if rescale_cfg is None:
            rescale_cfg = getattr(self, "rescale_cfg", self._DEFAULT_RESCALE_CFG)

        pos = _unwrap(_wrap(positive_noise_pred))
        neg = _unwrap(_wrap(negative_noise_pred))

        pos_flow_pred_std = pos.std()
        flow_pred_cfg = pos + true_cfg_scale * (pos - neg)
        rescale_flow_pred = flow_pred_cfg * pos_flow_pred_std / flow_pred_cfg.std()
        combined = rescale_cfg * rescale_flow_pred + (1 - rescale_cfg) * flow_pred_cfg
        return combined.to(dtype=pos.dtype)

    def _run_flow_matching_loop(
        self,
        prompt: torch.Tensor,
        cond: torch.Tensor,
        n_timesteps: int,
        cfg: float,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Euler flow-matching loop aligned with ``FlowMatchingTransformer.reverse_diffusion``."""
        cond_emb, prompt, xt, x_mask, xt_mask, h_tensor, prompt_len = self._prepare_cfm_loop_state(
            prompt, cond, n_timesteps, generator=generator
        )
        trunk_dtype = self.diffusion_trunk_dtype
        step_scale = 1.0 / n_timesteps

        for i in range(n_timesteps):
            xt_input = torch.cat([prompt, xt], dim=1)
            diffusion_step = torch.full(
                (xt.shape[0],),
                fill_value=(i + 0.5) * step_scale,
                device=xt.device,
                dtype=trunk_dtype,
            )

            flow_pred = self.predict_noise_maybe_with_cfg(
                do_true_cfg=(cfg > 0),
                true_cfg_scale=cfg,
                positive_kwargs={
                    "x": xt_input,
                    "diffusion_step": diffusion_step,
                    "cond": cond_emb,
                    "x_mask": xt_mask,
                    "prompt_len": prompt_len,
                },
                negative_kwargs={
                    "x": xt,
                    "diffusion_step": diffusion_step,
                    # CFG null cond: zero out target frames only (prompt cond stays in pos branch).
                    "cond": torch.zeros_like(cond_emb)[:, : xt.shape[1], :],
                    "x_mask": x_mask,
                },
                cfg_normalize=False,
            )

            xt = xt + flow_pred * h_tensor

        return xt
