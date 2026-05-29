import os
from collections.abc import Iterable
from typing import ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.soulx_singer.modules import (
    CFMDecoder,
    ConvNeXtV2Block,
)
from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_base import (
    FlowMatchingAudioPipeline,
)
from vllm_omni.diffusion.models.soulx_singer.preprocess.payload import (
    SOULX_SVS_KIND,
    consume_payload,
)
from vllm_omni.diffusion.models.soulx_singer.preprocess.pre_process import (
    attach_preprocess_for_diffusion_request,
    build_metadata_processor,
)
from vllm_omni.diffusion.models.soulx_singer.utils import (
    _patch_torchaudio_load,
    f0_to_coarse,
    resolve_pitch_shift,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

_DEFAULT_NUM_INFERENCE_STEPS = 32
_DEFAULT_GUIDANCE_SCALE = 1.0


def get_soulxsinger_pre_process_func(od_config: OmniDiffusionConfig):
    """Validate/load SVS preprocess payload for single-stage or stage-1 DiT."""
    metadata_processor = build_metadata_processor(od_config)

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        return attach_preprocess_for_diffusion_request(
            request,
            kind=SOULX_SVS_KIND,
            metadata_processor=metadata_processor,
        )

    return pre_process_func


def _expand_states(h: torch.Tensor, mel2token: torch.Tensor) -> torch.Tensor:
    if mel2token.max() > h.size(1) - 1:
        logger.warning(
            "mel2token.max() (%s) is greater than h.size(1) - 1 (%s); clamping.",
            mel2token.max(),
            h.size(1) - 1,
        )
        mel2token = torch.clamp(mel2token, 0, h.size(1) - 1)
    mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
    return torch.gather(h, 1, mel2token_)


class PipelineSoulXSingerSVS(FlowMatchingAudioPipeline):
    """Pipeline for the SoulX-Singer model (SoulX-Singer)."""

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
            "prompt_audio",
            "target_audio",
            "language",
            "vocal_sep",
            "max_merge_duration",
            "preprocess_weights_dir",
            "preprocess_verbose",
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

        self.mel, self.vocoder = self._build_fp32_audio_modules(self.audio_config)

        self.note_text_encoder = nn.Embedding(self.vocab_size, self.text_dim)
        self.note_pitch_encoder = nn.Embedding(256, self.pitch_dim)
        self.note_type_encoder = nn.Embedding(256, self.type_dim)

        self._setup_soulx_profiler()

    def _encode_condition(
        self,
        *,
        note_pitch: torch.Tensor,
        note_type: torch.Tensor,
        note_text: torch.Tensor,
        mel2note: torch.Tensor,
        f0_coarse: torch.Tensor,
    ) -> torch.Tensor:
        features = (
            self.note_pitch_encoder(note_pitch) + self.note_type_encoder(note_type) + self.note_text_encoder(note_text)
        )
        features = self.preflow(features)
        features = _expand_states(features, mel2note)
        features = features + self.f0_encoder(f0_coarse)
        return self._to_trunk_dtype(features)[0]

    def _infer_svs_segment(
        self,
        prompt_meta: dict,
        target_meta: dict,
        *,
        pitch_shift: int,
        num_inference_steps: int,
        guidance_scale: float,
        prompt_mel: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        prompt_note_text = prompt_meta["phoneme"]
        prompt_mel2note = prompt_meta["mel2note"]
        prompt_note_type = prompt_meta["note_type"]
        prompt_note_pitch = prompt_meta["note_pitch"]
        prompt_f0 = prompt_meta["f0"]

        target_note_text = target_meta["phoneme"]
        target_mel2note = target_meta["mel2note"]
        target_note_type = target_meta["note_type"]
        target_note_pitch = target_meta["note_pitch"]
        target_f0 = target_meta["f0"]

        if target_f0 is None or prompt_f0 is None:
            target_f0 = torch.zeros_like(target_mel2note).float()
            prompt_f0 = torch.zeros_like(prompt_mel2note).float()
        if target_note_pitch is None or prompt_note_pitch is None:
            target_note_pitch = torch.zeros_like(target_note_type).int()
            prompt_note_pitch = torch.zeros_like(prompt_note_type).int()

        len_prompt_note = prompt_note_pitch.shape[1]
        len_prompt_mel = prompt_f0.shape[1] if prompt_f0 is not None else prompt_mel2note.shape[1]

        note_pitch = torch.cat([prompt_note_pitch, target_note_pitch], dim=1)
        note_text = torch.cat([prompt_note_text, target_note_text], dim=1)
        note_type = torch.cat([prompt_note_type, target_note_type], dim=1)
        # Target note indices follow prompt notes in the concatenated score.
        mel2note = torch.cat([prompt_mel2note, target_mel2note + len_prompt_note], dim=1)

        # pitch_shift is semitones; each coarse F0 bin is 20 cents (×5 bins per semitone).
        f0_coarse_prompt = f0_to_coarse(prompt_f0)
        f0_coarse_target = f0_to_coarse(target_f0, f0_shift=pitch_shift * 5)
        f0_coarse = torch.cat([f0_coarse_prompt, f0_coarse_target], dim=1)

        note_pitch = note_pitch.clone()
        note_pitch[note_pitch > 0] = note_pitch[note_pitch > 0] + pitch_shift
        note_pitch = torch.clamp(note_pitch, 0, 255)

        if prompt_mel is None:
            prompt_wav = prompt_meta["wav"]
            with self._stage_timer("mel"):
                prompt_mel = self.mel(prompt_wav.float() if prompt_wav.dtype != torch.float32 else prompt_wav)

        if prompt_mel.shape[1] > len_prompt_mel:
            prompt_mel = prompt_mel[:, :len_prompt_mel, :]
        elif prompt_mel.shape[1] < len_prompt_mel:
            logger.warning(
                "prompt_mel length %s is shorter than metadata frames %s; padding mel.",
                prompt_mel.shape[1],
                len_prompt_mel,
            )
            prompt_mel = F.pad(prompt_mel, (0, 0, 0, len_prompt_mel - prompt_mel.shape[1]))

        with self._stage_timer("cond_encode"):
            cond = self._encode_condition(
                note_pitch=note_pitch,
                note_type=note_type,
                note_text=note_text,
                mel2note=mel2note,
                f0_coarse=f0_coarse,
            )

        with self._stage_timer("cfm"):
            generated_mel = self._run_flow_matching_loop(
                prompt_mel,
                cond,
                num_inference_steps,
                guidance_scale,
                generator=generator,
            )
        with self._stage_timer("vocoder"):
            return self._mel_to_audio(generated_mel)

    @staticmethod
    def _apply_control_mode(meta: dict, control: str) -> dict:
        meta["note_pitch"] = meta["note_pitch"] if control == "score" else None
        meta["f0"] = meta["f0"] if control == "melody" else None
        return meta

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
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

        auto_shift = extra_args.get("auto_shift", False)
        pitch_shift = int(extra_args.get("pitch_shift", 0))
        generator = self._resolve_diffusion_generator(sampling_params)

        with self._stage_timer("consume_payload"):
            payload = consume_payload(req, SOULX_SVS_KIND, self.device)
            prompt_meta = self._apply_control_mode(payload["prompt_meta"], control)
            target_meta_list = payload["target_meta_list"]

        sample_rate = self.audio_config.sample_rate
        generated_len = int(target_meta_list[-1]["time"][1] / 1000 * sample_rate)
        generated_merged = torch.zeros(generated_len, device=self.device, dtype=torch.float32)
        last_pitch_shift = 0

        prompt_wav = prompt_meta["wav"]
        with self._stage_timer("mel"):
            prompt_mel = self.mel(prompt_wav.float() if prompt_wav.dtype != torch.float32 else prompt_wav)

        for target_raw in target_meta_list:
            target_meta = self.metadata_processor.process(target_raw, None)
            target_meta = self._apply_control_mode(target_meta, control)

            last_pitch_shift = resolve_pitch_shift(
                auto_shift=auto_shift,
                manual_shift=pitch_shift,
                prompt_f0=prompt_meta["f0"],
                target_f0=target_meta["f0"],
                prompt_note_pitch=prompt_meta["note_pitch"],
                target_note_pitch=target_meta["note_pitch"],
            )
            segment_audio = self._infer_svs_segment(
                prompt_meta,
                target_meta,
                pitch_shift=last_pitch_shift,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                prompt_mel=prompt_mel,
                generator=generator,
            )
            segment_flat = segment_audio.squeeze()
            start_sample_idx = int(target_raw["time"][0] / 1000 * sample_rate)
            gen_len = min(segment_flat.shape[0], generated_len - start_sample_idx)
            with self._stage_timer("merge"):
                generated_merged[start_sample_idx : start_sample_idx + gen_len] = segment_flat[:gen_len]

        merged_audio = generated_merged.unsqueeze(0)

        return DiffusionOutput(
            output=merged_audio,
            custom_output={"f0_shift": last_pitch_shift},
            stage_durations=self._profiler_stage_durations() or {},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        _patch_torchaudio_load()
        weight_path = os.path.join(self.model_path, "model.pt")
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"Model weights not found at {weight_path}. Please check the pretrained model path."
            )
        state = torch.load(weight_path, map_location=self.device)
        self.mel.float()
        self.load_state_dict(state["state_dict"], strict=True)
        self._finalize_loaded_dtypes()
        logger.info("Loaded model weights from %s", weight_path)
