import os
from collections.abc import Iterable
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.soulx_singer.modules import (
    CFMDecoder,
    WhisperEncoder,
)
from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_base import (
    FlowMatchingAudioPipeline,
)
from vllm_omni.diffusion.models.soulx_singer.preprocess.payload import (
    SOULX_SVC_KIND,
    consume_payload,
)
from vllm_omni.diffusion.models.soulx_singer.preprocess.pre_process import (
    attach_preprocess_for_diffusion_request,
)
from vllm_omni.diffusion.models.soulx_singer.utils import (
    _patch_torchaudio_load,
    f0_to_coarse,
    load_config,
    resolve_pitch_shift,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

_DEFAULT_NUM_INFERENCE_STEPS = 32
_DEFAULT_GUIDANCE_SCALE = 3.0
_LONG_AUDIO_SEGMENT_THRESHOLD_SEC = 30.0


def get_soulxsinger_svc_pre_process_func(od_config: OmniDiffusionConfig):
    """Validate/load SVC preprocess payload for single-stage or stage-1 DiT."""
    hf_config = load_config(os.path.join(od_config.model, "config.yaml"))
    sample_rate = hf_config.audio.sample_rate
    device = get_local_device()

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        return attach_preprocess_for_diffusion_request(
            request,
            kind=SOULX_SVC_KIND,
            sample_rate=sample_rate,
            device=device,
        )

    return pre_process_func


class PipelineSoulXSingerSVC(FlowMatchingAudioPipeline):
    """SVC pipeline for the SoulX-Singer model."""

    _encoder_modules: ClassVar[list[str]] = [
        "whisper_encoder",
        "mel",
        "f0_encoder",
    ]

    EXTRA_BODY_PARAMS: ClassVar[frozenset[str]] = frozenset(
        {
            "prompt_wav_path",
            "target_wav_path",
            "prompt_f0_path",
            "target_f0_path",
            "prompt_audio",
            "target_audio",
            "vocal_sep",
            "preprocess_weights_dir",
            "preprocess_verbose",
            "auto_shift",
            "pitch_shift",
        }
    )

    EXTRA_OUTPUT_PARAMS: ClassVar[frozenset[str]] = frozenset({"pitch_shift"})

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)

        self.f0_encoder = nn.Embedding(self.f0_bin, self.f0_dim)

        self.cfm_decoder = CFMDecoder(self.flow_matching_config)

        self.mel, self.vocoder = self._build_fp32_audio_modules(self.audio_config)
        with set_default_torch_dtype(torch.float32):
            self.whisper_encoder = WhisperEncoder(device=self.device)

        self._setup_soulx_profiler()

    @staticmethod
    def build_vocal_segments(
        f0,
        *,
        f0_rate: int = 50,
        ignore_silent_frames_thresh: int = 5,
        min_duration_sec_per_segment: float = 5.0,
        max_duration_sec_per_segment: float = 30.0,
        num_overlaps: int = 1,
        ignore_silent_frames: bool = True,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Build vocal segments from an F0 contour for chunked SVC inference."""
        if isinstance(f0, torch.Tensor):
            f0_np = f0.detach().float().cpu().numpy()
        else:
            f0_np = np.asarray(f0, dtype=np.float32)
        f0_np = np.squeeze(f0_np)

        total_frames = int(f0_np.shape[0])
        if total_frames == 0:
            return [], []

        min_frames = max(1, int(round(min_duration_sec_per_segment * f0_rate)))
        max_frames = max(1, int(round(max_duration_sec_per_segment * f0_rate)))

        split_points = [0]

        def append_split_point(point: int):
            point = int(max(0, min(point, total_frames)))
            while point - split_points[-1] > max_frames:
                split_points.append(split_points[-1] + max_frames)
            if point > split_points[-1]:
                split_points.append(point)

        idx = 0
        while idx < total_frames:
            if f0_np[idx] == 0:
                run_start = idx
                while idx < total_frames and f0_np[idx] == 0:
                    idx += 1
                run_end = idx
                if (run_end - run_start) >= ignore_silent_frames_thresh:
                    split_point = max(run_end - 5, (run_start + run_end) // 2)
                    append_split_point(split_point)
            else:
                idx += 1
        append_split_point(total_frames)

        segments: list[tuple[float, float]] = []
        overlap_segments: list[tuple[float, float]] = []

        def append_segment(start_idx: int, end_idx: int, overlaps: int = num_overlaps):
            segments.append((split_points[start_idx] / f0_rate, split_points[end_idx] / f0_rate))
            overlap_start_idx = start_idx
            if start_idx > 0 and (split_points[end_idx] - split_points[start_idx - overlaps]) <= max_frames:
                overlap_start_idx = start_idx - overlaps
            overlap_segments.append((split_points[overlap_start_idx] / f0_rate, split_points[end_idx] / f0_rate))

        segment_start, segment_end = 0, 1
        while segment_start < len(split_points) - 1:
            while (
                segment_end < len(split_points)
                and (split_points[segment_end] - split_points[segment_start]) < min_frames
            ):
                segment_end += 1

            if segment_end >= len(split_points):
                append_segment(segment_start, len(split_points) - 1, overlaps=num_overlaps)
                break
            append_segment(segment_start, segment_end, overlaps=num_overlaps)
            segment_start = segment_end
            segment_end = segment_start + 1

        if ignore_silent_frames:
            filtered_idx = []
            for i, seg in enumerate(overlap_segments):
                start_frame = int(seg[0] * f0_rate)
                end_frame = int(seg[1] * f0_rate)
                seg_frames = end_frame - start_frame
                voice_frames = np.sum(f0_np[start_frame:end_frame] > 0)
                if voice_frames / seg_frames > 0.05 and voice_frames >= 10:
                    filtered_idx.append(i)

            overlap_segments = [overlap_segments[i] for i in filtered_idx]
            segments = [segments[i] for i in filtered_idx]

        return overlap_segments, segments

    def _encode_condition(self, *, whisper_features: torch.Tensor, f0_coarse: torch.Tensor) -> torch.Tensor:
        cond = whisper_features + self.f0_encoder(f0_coarse)
        return self._to_trunk_dtype(cond)[0]

    def _encode_prompt_whisper_feature(self, prompt_wav: torch.Tensor) -> torch.Tensor:
        trunk_dtype = self.f0_encoder.weight.dtype
        with self._stage_timer("whisper"):
            return self.whisper_encoder.encode(
                prompt_wav,
                sr=self.audio_config.sample_rate,
                output_dtype=trunk_dtype,
            )

    def _infer_segment(
        self,
        *,
        prompt_mel: torch.Tensor,
        prompt_wav: torch.Tensor,
        target_wav: torch.Tensor,
        prompt_f0: torch.Tensor,
        target_f0: torch.Tensor,
        pitch_shift: int,
        num_inference_steps: int,
        guidance_scale: float,
        prompt_feature: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Single-segment SVC inference aligned with ``SoulXSingerSVC.infer_segment``."""
        len_prompt_mel = prompt_mel.shape[1]
        prompt_f0 = F.pad(prompt_f0, (0, 0, 0, max(0, len_prompt_mel - prompt_f0.shape[1])))[:, :len_prompt_mel]

        f0_coarse_prompt = f0_to_coarse(prompt_f0)
        f0_coarse_target = f0_to_coarse(target_f0, f0_shift=int(pitch_shift * 5))
        f0_coarse = torch.cat([f0_coarse_prompt, f0_coarse_target], dim=1)

        trunk_dtype = self.f0_encoder.weight.dtype
        if prompt_feature is None:
            prompt_feature = self._encode_prompt_whisper_feature(prompt_wav)
        with self._stage_timer("whisper"):
            target_feature = self.whisper_encoder.encode(
                target_wav,
                sr=self.audio_config.sample_rate,
                output_dtype=trunk_dtype,
            )
        prompt_feature = F.pad(
            prompt_feature,
            (0, 0, 0, max(0, f0_coarse_prompt.shape[1] - prompt_feature.shape[1])),
        )[:, : f0_coarse_prompt.shape[1], :]
        target_feature = F.pad(
            target_feature,
            (0, 0, 0, max(0, f0_coarse_target.shape[1] - target_feature.shape[1])),
        )[:, : f0_coarse_target.shape[1], :]

        whisper_features = torch.cat([prompt_feature, target_feature], dim=1)
        with self._stage_timer("cond_encode"):
            cond = self._encode_condition(whisper_features=whisper_features, f0_coarse=f0_coarse)

        with self._stage_timer("cfm"):
            generated_mel = self._run_flow_matching_loop(
                prompt=prompt_mel,
                cond=cond,
                n_timesteps=num_inference_steps,
                cfg=guidance_scale,
                generator=generator,
            )

        with self._stage_timer("vocoder"):
            generated_audio = self._mel_to_audio(generated_mel, squeeze=True)
        target_len = target_wav.shape[-1]
        if generated_audio.shape[-1] > target_len:
            generated_audio = generated_audio[:target_len]
        elif generated_audio.shape[-1] < target_len:
            generated_audio = F.pad(generated_audio, (0, target_len - generated_audio.shape[-1]))
        return generated_audio

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        sampling_params = req.sampling_params
        extra_args = dict(getattr(sampling_params, "extra_args", None) or {})

        num_inference_steps = sampling_params.num_inference_steps or _DEFAULT_NUM_INFERENCE_STEPS
        guidance_scale = sampling_params.guidance_scale or _DEFAULT_GUIDANCE_SCALE

        with self._stage_timer("consume_payload"):
            payload = consume_payload(req, SOULX_SVC_KIND, self.device)
            prompt_wav = payload["prompt_wav"]
            target_wav = payload["target_wav"]
            prompt_f0 = payload["prompt_f0"]
            target_f0 = payload["target_f0"]

        auto_shift = extra_args.get("auto_shift", False)
        pitch_shift = extra_args.get("pitch_shift", 0)
        pitch_shift = resolve_pitch_shift(
            auto_shift=auto_shift,
            manual_shift=int(pitch_shift),
            prompt_f0=prompt_f0,
            target_f0=target_f0,
        )
        generator = self._resolve_diffusion_generator(sampling_params)

        with self._stage_timer("mel"):
            prompt_mel = self.mel(prompt_wav.float() if prompt_wav.dtype != torch.float32 else prompt_wav)

        prompt_feature = self._encode_prompt_whisper_feature(prompt_wav)
        f0_rate = self.audio_config.sample_rate // self.audio_config.hop_size
        target_duration_sec = target_wav.shape[-1] / self.audio_config.sample_rate

        if target_duration_sec < _LONG_AUDIO_SEGMENT_THRESHOLD_SEC:
            generated_audio = self._infer_segment(
                prompt_mel=prompt_mel,
                prompt_wav=prompt_wav,
                target_wav=target_wav,
                prompt_f0=prompt_f0,
                target_f0=target_f0,
                pitch_shift=pitch_shift,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                prompt_feature=prompt_feature,
                generator=generator,
            )
            if generated_audio.dim() == 1:
                generated_audio = generated_audio.unsqueeze(0)
            return DiffusionOutput(
                output=generated_audio,
                custom_output={"pitch_shift": pitch_shift},
                stage_durations=self._profiler_stage_durations() or {},
            )

        overlap_segments, segments = self.build_vocal_segments(
            target_f0,
            f0_rate=f0_rate,
            ignore_silent_frames_thresh=10,
            min_duration_sec_per_segment=min(15.0, max(target_duration_sec, 1e-6)),
            max_duration_sec_per_segment=30.0,
            ignore_silent_frames=True,
        )
        if not segments:
            overlap_segments = [(0.0, target_duration_sec)]
            segments = [(0.0, target_duration_sec)]

        generated_audio = torch.zeros_like(target_wav)
        for idx in range(len(segments)):
            overlap_start_sec, overlap_end_sec = overlap_segments[idx]
            seg_start_sec, seg_end_sec = segments[idx]

            wav_start = int(round(overlap_start_sec * self.audio_config.sample_rate))
            wav_end = int(round(overlap_end_sec * self.audio_config.sample_rate))
            f0_start = int(round(overlap_start_sec * f0_rate))
            f0_end = int(round(overlap_end_sec * f0_rate))

            wav_start = max(0, min(wav_start, target_wav.shape[-1]))
            wav_end = max(wav_start, min(wav_end, target_wav.shape[-1]))
            f0_start = max(0, min(f0_start, target_f0.shape[-1]))
            f0_end = max(f0_start, min(f0_end, target_f0.shape[-1]))

            segment_target_wav = target_wav[:, wav_start:wav_end]
            segment_target_f0 = target_f0[:, f0_start:f0_end]

            segment_generated_audio = self._infer_segment(
                prompt_mel=prompt_mel,
                prompt_wav=prompt_wav,
                target_wav=segment_target_wav,
                prompt_f0=prompt_f0,
                target_f0=segment_target_f0,
                pitch_shift=pitch_shift,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                prompt_feature=prompt_feature,
                generator=generator,
            )

            segment_start = int(round(seg_start_sec * self.audio_config.sample_rate))
            segment_end = int(round(seg_end_sec * self.audio_config.sample_rate))
            generated_audio[:, segment_start:segment_end] = segment_generated_audio[
                segment_start - wav_start : segment_end - wav_start
            ]

        return DiffusionOutput(
            output=generated_audio,
            custom_output={"pitch_shift": pitch_shift},
            stage_durations=self._profiler_stage_durations() or {},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        _patch_torchaudio_load()
        del weights
        weight_path = os.path.join(self.model_path, "model-svc.pt")
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(
                f"Model weights not found at {weight_path}. Please check the pretrained model path."
            )
        state = torch.load(weight_path, map_location=self.device)
        self.mel.float()
        self.load_state_dict(state["state_dict"], strict=True)
        self._finalize_loaded_dtypes()
        logger.info("Loaded model weights from %s", weight_path)
