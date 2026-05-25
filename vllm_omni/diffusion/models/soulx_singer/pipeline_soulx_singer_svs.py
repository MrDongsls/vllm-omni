import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.soulx_singer.modules import (
    CFMDecoder,
    ConvNeXtV2Block,
)
from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_base import (
    FlowMatchingAudioPipeline,
)
from vllm_omni.diffusion.models.soulx_singer.utils import _patch_torchaudio_load, f0_to_coarse
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

PROJECT_ROOT = Path(__file__).parents[4]
_SOULX_EXAMPLE_AUDIO_DIR = PROJECT_ROOT / "tests" / "assets" / "soulxsinger"
_SOULX_REQUIRED_EXTRA_KEYS = ("prompt_metadata_path", "target_metadata_path", "audio_path")
_DEFAULT_NUM_INFERENCE_STEPS = 32
_DEFAULT_GUIDANCE_SCALE = 1.0
_DUMMY_WAV_SAMPLE_RATE = 24000
_DUMMY_WAV_DURATION_S = 30.0
_dummy_wav_path: str | None = None


def _is_warmup_request(request: OmniDiffusionRequest) -> bool:
    request_ids = getattr(request, "request_ids", None) or ()
    return len(request_ids) == 1 and request_ids[0] == "dummy_req_id"


def _ensure_dummy_wav_path() -> str:
    global _dummy_wav_path
    if _dummy_wav_path is not None and os.path.isfile(_dummy_wav_path):
        return _dummy_wav_path

    import soundfile as sf

    wav_path = Path(tempfile.gettempdir()) / "vllm_omni_soulx_singer_dummy.wav"
    num_samples = int(_DUMMY_WAV_SAMPLE_RATE * _DUMMY_WAV_DURATION_S)
    silence = np.zeros((num_samples, 1), dtype=np.float32)
    sf.write(str(wav_path), silence, _DUMMY_WAV_SAMPLE_RATE)
    _dummy_wav_path = str(wav_path)
    return _dummy_wav_path


def get_soulxsinger_pre_process_func(od_config: OmniDiffusionConfig):
    """Inject SoulX-Singer metadata paths for DiffusionEngine dummy warmup."""
    del od_config  # reserved for future model-path overrides

    example_dir = _SOULX_EXAMPLE_AUDIO_DIR
    prompt_metadata_path = example_dir / "zh_prompt.json"
    target_metadata_path = example_dir / "music.json"
    audio_path = example_dir / "zh_prompt.mp3"

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        extra_args = dict(getattr(request.sampling_params, "extra_args", None) or {})

        if _is_warmup_request(request):
            if not prompt_metadata_path.is_file() or not target_metadata_path.is_file():
                raise FileNotFoundError(
                    "SoulX-Singer bundled example metadata is missing under "
                    f"{example_dir}. Cannot run DiffusionEngine dummy warmup."
                )
            extra_args.setdefault("control", "score")
            request.sampling_params.num_inference_steps = 1
            extra_args["prompt_metadata_path"] = str(prompt_metadata_path)
            extra_args["target_metadata_path"] = str(target_metadata_path)
            extra_args["audio_path"] = str(audio_path) if audio_path.is_file() else _ensure_dummy_wav_path()
            request.sampling_params.extra_args = extra_args
            return request

        missing = [key for key in _SOULX_REQUIRED_EXTRA_KEYS if not extra_args.get(key)]
        if missing:
            raise ValueError(
                "SoulX-Singer requires the following sampling_params.extra_args fields: "
                f"{list(_SOULX_REQUIRED_EXTRA_KEYS)}. Missing: {missing}"
            )
        return request

    return pre_process_func


class PipelineSoulXSingerSVS(FlowMatchingAudioPipeline):
    """
    Pipeline for the SoulX-Singer model (SoulX-Singer).
    """

    _encoder_modules: ClassVar[list[str]] = [
        "mel",
        "f0_encoder",
        "preflow",
        "note_text_encoder",
        "note_pitch_encoder",
        "note_type_encoder",
    ]

    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "prompt_metadata_path",
            "target_metadata_path",
            "audio_path",
            "control",
            "auto_shift",
            "pitch_shift",
        }
    )

    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset({"f0_shift"})

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)

        self.f0_encoder = nn.Embedding(self.f0_bin, self.f0_dim)
        self.preflow = nn.Sequential(
            *[
                ConvNeXtV2Block(
                    self.text_dim,
                    self.text_dim * 2,
                )
                for _ in range(self.encoder_config.num_layers)
            ]
        )

        self.cfm_decoder = CFMDecoder(self.flow_matching_config)

        if self.audio_config is None and isinstance(self.encoder_config, dict):
            self.audio_config = self.encoder_config.get("audio_config")

        self.mel, self.vocoder = self._build_fp32_audio_modules(self.audio_config)

        # SVS-specific modules
        self.note_text_encoder = nn.Embedding(self.vocab_size, self.text_dim)
        self.note_pitch_encoder = nn.Embedding(256, self.pitch_dim)
        self.note_type_encoder = nn.Embedding(256, self.type_dim)

        self._setup_soulx_profiler()

    def _unwrap_state_dict(self, obj: object) -> dict[str, torch.Tensor]:
        if not isinstance(obj, dict):
            raise TypeError(f"checkpoint root must be dict, got {type(obj)}")
        out = obj
        for key in ("state_dict", "model", "weights"):
            if key in out and isinstance(out[key], dict):
                out = out[key]
                break
        if not isinstance(out, dict):
            raise TypeError("unwrapped checkpoint is not a dict")
        return out

    def _encode_condition(self, *args, **kwargs) -> torch.Tensor:
        def expand_states(h, mel2token):
            """
            Expand the states to the mel-scale.
            args:
                h: states, shape: [B, T, H]
                mel2token: mel2token, shape: [B, F]
            returns:
                h: expanded states, shape: [B, F, H]
            """
            try:
                assert mel2token.max() <= h.size(1) - 1
            except AssertionError:
                logger.warning(
                    "mel2token.max() (%s) is greater than h.size(1) - 1 (%s)",
                    mel2token.max(),
                    h.size(1) - 1,
                )
                mel2token = torch.clamp(mel2token, 0, h.size(1) - 1)
            mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
            h = torch.gather(h, 1, mel2token_)  # [B, T, H]
            return h

        features = (
            self.note_pitch_encoder(kwargs["note_pitch"])
            + self.note_type_encoder(kwargs["note_type"])
            + self.note_text_encoder(kwargs["note_text"])
        )
        features = self.preflow(features)
        features = expand_states(features, kwargs["mel2note"])
        features = features + self.f0_encoder(kwargs["f0_course"])
        return self._to_trunk_dtype(features)[0]

    def _load_metadata_json(self, extra: dict) -> tuple[dict, list[dict]]:
        wav_path = extra["audio_path"]

        with open(extra["prompt_metadata_path"], encoding="utf-8") as f:
            prompt_metadata = json.load(f)
        if not prompt_metadata:
            raise ValueError("prompt_metadata is empty. Please run preprocess on prompt audio first.")

        with open(extra["target_metadata_path"], encoding="utf-8") as f:
            target_metadata = json.load(f)
        if not target_metadata:
            raise ValueError("No target segments. Please run preprocess on target audio first.")

        infer_prompt_meta = self.metadata_processor.process(prompt_metadata[0], wav_path)
        return infer_prompt_meta, target_metadata

    @staticmethod
    def _compute_f0_shift(
        infer_prompt_meta: dict,
        infer_target_meta: dict,
        *,
        auto_shift: bool,
        pitch_shift: int,
    ) -> int:
        pt_note_pitch = infer_prompt_meta["note_pitch"]
        gt_note_pitch = infer_target_meta["note_pitch"]
        pt_f0 = infer_prompt_meta["f0"]
        gt_f0 = infer_target_meta["f0"]

        if auto_shift and pitch_shift == 0:
            if pt_note_pitch is not None and gt_note_pitch is not None:
                gt_median = torch.median(gt_note_pitch[gt_note_pitch >= 1])
                pt_median = torch.median(pt_note_pitch[pt_note_pitch >= 1])
                return torch.round(pt_median - gt_median).int().item()
            if pt_f0 is not None and gt_f0 is not None:
                gt_f0_median = torch.median(gt_f0[gt_f0 > 0])
                pt_f0_median = torch.median(pt_f0[pt_f0 > 0])
                return torch.round(torch.log2(pt_f0_median / gt_f0_median) * 1200 / 100).int().item()
            logger.warning("Pitch shift is enabled but note_pitch or f0 is not provided. Setting f0_shift to 0.")
            return 0
        return pitch_shift

    def _infer_svs_segment(
        self,
        infer_prompt_meta: dict,
        infer_target_meta: dict,
        *,
        f0_shift: int,
        num_inference_steps: int,
        guidance_scale: float,
        prompt_mel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pt_note_text = infer_prompt_meta["phoneme"]
        pt_mel2note = infer_prompt_meta["mel2note"]
        pt_note_type = infer_prompt_meta["note_type"]
        pt_note_pitch = infer_prompt_meta["note_pitch"]
        pt_f0 = infer_prompt_meta["f0"]

        gt_note_text = infer_target_meta["phoneme"]
        gt_mel2note = infer_target_meta["mel2note"]
        gt_note_type = infer_target_meta["note_type"]
        gt_note_pitch = infer_target_meta["note_pitch"]
        gt_f0 = infer_target_meta["f0"]

        if gt_f0 is None or pt_f0 is None:
            gt_f0 = torch.zeros_like(gt_mel2note).float()
            pt_f0 = torch.zeros_like(pt_mel2note).float()
        if gt_note_pitch is None or pt_note_pitch is None:
            gt_note_pitch = torch.zeros_like(gt_note_type).int()
            pt_note_pitch = torch.zeros_like(pt_note_type).int()

        # 3. encode condition and run flow matching (see soulxsinger.models.soulxsinger.infer)
        len_prompt = pt_note_pitch.shape[1]

        note_pitch = torch.cat([pt_note_pitch, gt_note_pitch], dim=1)
        note_text = torch.cat([pt_note_text, gt_note_text], dim=1)
        note_type = torch.cat([pt_note_type, gt_note_type], dim=1)
        mel2note = torch.cat([pt_mel2note, gt_mel2note + len_prompt], dim=1)

        f0_course_pt = f0_to_coarse(pt_f0)
        f0_course_gt = f0_to_coarse(gt_f0, f0_shift=f0_shift * 5)  # NOTE: Why times 5?
        f0_course = torch.cat([f0_course_pt, f0_course_gt], dim=1)

        note_pitch = note_pitch.clone()
        note_pitch[note_pitch > 0] = note_pitch[note_pitch > 0] + f0_shift
        note_pitch = torch.clamp(note_pitch, 0, 255)

        # Mel stays FP32; CFM boundary casts prompt/cond in ``_prepare_cfm_loop_state``.
        if prompt_mel is None:
            pt_wav = infer_prompt_meta["wav"]
            with self._stage_timer("mel"):
                prompt_mel = self.mel(pt_wav.float() if pt_wav.dtype != torch.float32 else pt_wav)

        with self._stage_timer("cond_encode"):
            cond = self._encode_condition(
                note_pitch=note_pitch,
                note_type=note_type,
                note_text=note_text,
                mel2note=mel2note,
                f0_course=f0_course,
            )

        # 4. run flow matching loop
        with self._stage_timer("cfm"):
            generated_mel = self._run_flow_matching_loop(
                prompt_mel,
                cond,
                num_inference_steps,
                guidance_scale,
            )
        with self._stage_timer("vocoder"):
            return self.vocoder(generated_mel.transpose(1, 2)[0:1, ...].float()).float()

    @staticmethod
    def _apply_control_mode(meta: dict, control: str) -> dict:
        meta["note_pitch"] = meta["note_pitch"] if control == "score" else None
        meta["f0"] = meta["f0"] if control == "melody" else None
        return meta

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        # 1. extract parameters from request
        sampling_params = req.sampling_params
        extra_args = dict(getattr(sampling_params, "extra_args", None) or {})

        control = extra_args.get("control")
        if control is None:
            control = "score"
            logger.warning("control is not provided, using 'score' as default")
        elif control not in ("score", "melody"):
            raise ValueError(f"Invalid control: {control}. Must be one of: ['score', 'melody']")
        extra_args["control"] = control
        sampling_params.extra_args = extra_args

        num_inference_steps = sampling_params.num_inference_steps or _DEFAULT_NUM_INFERENCE_STEPS
        guidance_scale = sampling_params.guidance_scale or _DEFAULT_GUIDANCE_SCALE

        auto_shift = extra_args.get("auto_shift", True)
        pitch_shift = int(extra_args.get("pitch_shift", 0))

        with self._stage_timer("preprocess"):
            infer_prompt_meta, target_meta_list = self._load_metadata_json(extra_args)
            infer_prompt_meta = self._apply_control_mode(infer_prompt_meta, control)

        sample_rate = self.audio_config.sample_rate
        generated_len = int(target_meta_list[-1]["time"][1] / 1000 * sample_rate)
        generated_merged = torch.zeros(generated_len, device=self.device, dtype=torch.float32)
        last_f0_shift = 0

        pt_wav = infer_prompt_meta["wav"]
        with self._stage_timer("mel"):
            prompt_mel = self.mel(pt_wav.float() if pt_wav.dtype != torch.float32 else pt_wav)

        for target_raw in target_meta_list:
            infer_target_meta = self.metadata_processor.process(target_raw, None)
            infer_target_meta = self._apply_control_mode(infer_target_meta, control)

            last_f0_shift = self._compute_f0_shift(
                infer_prompt_meta,
                infer_target_meta,
                auto_shift=auto_shift,
                pitch_shift=pitch_shift,
            )
            segment_audio = self._infer_svs_segment(
                infer_prompt_meta,
                infer_target_meta,
                f0_shift=last_f0_shift,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                prompt_mel=prompt_mel,
            )
            segment_flat = segment_audio.squeeze()
            start_sample_idx = int(target_raw["time"][0] / 1000 * sample_rate)
            gen_len = min(segment_flat.shape[0], generated_len - start_sample_idx)
            with self._stage_timer("merge"):
                generated_merged[start_sample_idx : start_sample_idx + gen_len] = segment_flat[:gen_len]

        merged_audio = generated_merged.unsqueeze(0)

        return DiffusionOutput(
            output=merged_audio,
            custom_output={"f0_shift": last_f0_shift},
            stage_durations=self._profiler_stage_durations() or {},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        _patch_torchaudio_load()
        # SoulX checkpoints ship as monolithic model.pt / model-svc.pt (not HF sharded
        # safetensors). The vLLM weights iterator is unused; load from model_path directly.
        weight_path = os.path.join(self.model_path, "model.pt")

        if os.path.exists(weight_path):
            state = torch.load(weight_path, map_location=self.device)
            model_weights = state["state_dict"]
            self.mel.float()

            self.load_state_dict(model_weights, strict=True)
            self._finalize_loaded_dtypes()
            logger.info(f"Loaded model weights from {weight_path}")
        else:
            raise FileNotFoundError(
                f"Model weights not found at {weight_path}. Please check the pretrained model path."
            )
