import os
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.soulx_singer.modules import (
    CFMDecoder,
    WhisperEncoder,
)
from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_base import FlowMatchingAudioPipeline
from vllm_omni.diffusion.models.soulx_singer.utils import _patch_torchaudio_load, f0_to_coarse, load_wav
from vllm_omni.diffusion.request import OmniDiffusionRequest

logger = init_logger(__name__)

PROJECT_ROOT = Path(__file__).parents[4]
_SOULX_EXAMPLE_AUDIO_DIR = PROJECT_ROOT / "tests" / "assets" / "soulxsinger"
_SOULX_REQUIRED_EXTRA_KEYS = ("prompt_wav_path", "target_wav_path", "prompt_f0_path", "target_f0_path")
_DEFAULT_NUM_INFERENCE_STEPS = 32
_DEFAULT_GUIDANCE_SCALE = 3.0
_LONG_AUDIO_SEGMENT_THRESHOLD_SEC = 30.0


def _is_warmup_request(request: OmniDiffusionRequest) -> bool:
    request_ids = getattr(request, "request_ids", None) or ()
    return len(request_ids) == 1 and request_ids[0] == "dummy_req_id"


def get_soulxsinger_svc_pre_process_func(od_config: OmniDiffusionConfig):
    """Inject SoulX-Singer metadata paths for DiffusionEngine dummy warmup."""
    del od_config  # reserved for future model-path overrides

    example_dir = _SOULX_EXAMPLE_AUDIO_DIR
    prompt_wav_path = example_dir / "zh_prompt.mp3"
    target_wav_path = example_dir / "music.mp3"
    prompt_f0_path = example_dir / "zh_prompt_f0.npy"
    target_f0_path = example_dir / "music_f0.npy"

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        extra_args = dict(getattr(request.sampling_params, "extra_args", None) or {})

        if _is_warmup_request(request):
            if (
                not prompt_wav_path.is_file()
                or not target_wav_path.is_file()
                or not prompt_f0_path.is_file()
                or not target_f0_path.is_file()
            ):
                raise FileNotFoundError(
                    "SoulX-Singer bundled example audio files are missing under "
                    f"{example_dir}. Cannot run DiffusionEngine dummy warmup."
                )
            request.sampling_params.num_inference_steps = 1
            extra_args["prompt_wav_path"] = str(prompt_wav_path)
            extra_args["target_wav_path"] = str(target_wav_path)
            extra_args["prompt_f0_path"] = str(prompt_f0_path)
            extra_args["target_f0_path"] = str(target_f0_path)
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

    def _plan_svc_segments(
        self,
        target_f0: torch.Tensor,
        target_duration_sec: float,
        *,
        f0_rate: int,
        ignore_silent_frames_thresh: int = 10,
        min_duration_sec_per_segment: float = 15.0,
        max_duration_sec_per_segment: float = 30.0,
        ignore_silent_frames: bool = True,
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Plan segments so that short and long targets share one inference loop."""
        adaptive_min_duration = min(min_duration_sec_per_segment, max(target_duration_sec, 1e-6))
        overlap_segments, segments = self.build_vocal_segments(
            target_f0,
            f0_rate=f0_rate,
            ignore_silent_frames_thresh=ignore_silent_frames_thresh,
            min_duration_sec_per_segment=adaptive_min_duration,
            max_duration_sec_per_segment=max_duration_sec_per_segment,
            ignore_silent_frames=ignore_silent_frames,
        )
        if not segments:
            overlap_segments = [(0.0, target_duration_sec)]
            segments = [(0.0, target_duration_sec)]
        return overlap_segments, segments

    def _encode_condition(self, *args, **kwargs) -> torch.Tensor:
        feature = kwargs["res"]
        f0_feat = self.f0_encoder(kwargs["f0_course"])
        cond = feature + f0_feat
        return self._to_trunk_dtype(cond)[0]

    def _preprocess_metadata(self, extra: dict) -> tuple:
        prompt_wav_path = extra["prompt_wav_path"]
        prompt_f0_path = extra["prompt_f0_path"]
        target_wav_path = extra["target_wav_path"]
        target_f0_path = extra["target_f0_path"]

        prompt_wav = load_wav(prompt_wav_path, self.audio_config.sample_rate).to(self.device)
        target_wav = load_wav(target_wav_path, self.audio_config.sample_rate).to(self.device)
        prompt_f0 = torch.from_numpy(np.load(prompt_f0_path)).unsqueeze(0).to(self.device)
        target_f0 = torch.from_numpy(np.load(target_f0_path)).unsqueeze(0).to(self.device)

        return prompt_wav, target_wav, prompt_f0, target_f0

    def _resolve_pitch_shift(
        self,
        *,
        auto_shift: bool,
        pitch_shift: int,
        prompt_f0: torch.Tensor,
        target_f0: torch.Tensor,
    ) -> int:
        if auto_shift and pitch_shift == 0:
            if target_f0 is not None and prompt_f0 is not None:
                target_f0_median = torch.median(target_f0[target_f0 > 0])
                prompt_f0_median = torch.median(prompt_f0[prompt_f0 > 0])
                return int(torch.round(torch.log2(prompt_f0_median / target_f0_median) * 1200 / 100).item())
            logger.warning("Pitch shift is enabled but note_pitch or f0 is not provided. Setting pitch_shift to 0.")
            return 0
        return pitch_shift

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
    ) -> torch.Tensor:
        """Single-segment SVC inference aligned with ``SoulXSingerSVC.infer_segment``."""
        len_prompt_mel = prompt_mel.shape[1]
        prompt_f0 = F.pad(prompt_f0, (0, 0, 0, max(0, len_prompt_mel - prompt_f0.shape[1])))[:, :len_prompt_mel]

        f0_course_prompt = f0_to_coarse(prompt_f0)
        f0_course_target = f0_to_coarse(target_f0, f0_shift=int(pitch_shift * 5))
        if not isinstance(f0_course_prompt, torch.Tensor):
            f0_course_prompt = torch.as_tensor(f0_course_prompt, device=prompt_f0.device)
        if not isinstance(f0_course_target, torch.Tensor):
            f0_course_target = torch.as_tensor(f0_course_target, device=target_f0.device)
        f0_course = torch.cat([f0_course_prompt, f0_course_target], dim=1)

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
            (0, 0, 0, max(0, f0_course_prompt.shape[1] - prompt_feature.shape[1])),
        )[:, : f0_course_prompt.shape[1], :]
        target_feature = F.pad(
            target_feature,
            (0, 0, 0, max(0, f0_course_target.shape[1] - target_feature.shape[1])),
        )[:, : f0_course_target.shape[1], :]

        feature = torch.cat([prompt_feature, target_feature], dim=1)
        with self._stage_timer("cond_encode"):
            cond = self._encode_condition(res=feature, f0_course=f0_course)

        with self._stage_timer("cfm"):
            generated_mel = self._run_flow_matching_loop(
                prompt=prompt_mel,
                cond=cond,
                n_timesteps=num_inference_steps,
                cfg=guidance_scale,
            )

        with self._stage_timer("vocoder"):
            generated_audio = self.vocoder(generated_mel.transpose(1, 2)[0:1, ...].float()).squeeze().float()
        target_len = target_wav.shape[-1]
        if generated_audio.shape[-1] > target_len:
            generated_audio = generated_audio[:target_len]
        elif generated_audio.shape[-1] < target_len:
            generated_audio = F.pad(generated_audio, (0, target_len - generated_audio.shape[-1]))
        return generated_audio

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        # 1. extract parameters from request
        sampling_params = req.sampling_params
        extra_args = dict(getattr(sampling_params, "extra_args", None) or {})

        num_inference_steps = sampling_params.num_inference_steps or _DEFAULT_NUM_INFERENCE_STEPS
        guidance_scale = sampling_params.guidance_scale or _DEFAULT_GUIDANCE_SCALE

        with self._stage_timer("preprocess"):
            prompt_wav, target_wav, prompt_f0, target_f0 = self._preprocess_metadata(extra_args)

        # 2. calculate auto pitch shift
        auto_shift = extra_args.get("auto_shift", True)
        pitch_shift = extra_args.get("pitch_shift", 0)
        pitch_shift = self._resolve_pitch_shift(
            auto_shift=auto_shift,
            pitch_shift=pitch_shift,
            prompt_f0=prompt_f0,
            target_f0=target_f0,
        )

        # Mel spectrogram in FP32; CFM trunk dtype align in ``_run_flow_matching_loop``.
        with self._stage_timer("mel"):
            prompt_mel = self.mel(prompt_wav.float() if prompt_wav.dtype != torch.float32 else prompt_wav)

        len_prompt_mel = prompt_mel.shape[1]
        prompt_f0 = F.pad(prompt_f0, (0, 0, 0, max(0, len_prompt_mel - prompt_f0.shape[1])))[:, :len_prompt_mel]
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
            )
            if generated_audio.dim() == 1:
                generated_audio = generated_audio.unsqueeze(0)
            return DiffusionOutput(
                output=generated_audio,
                custom_output={"pitch_shift": pitch_shift},
                stage_durations=self._profiler_stage_durations() or {},
            )

        overlap_segments, segments = self._plan_svc_segments(
            target_f0,
            target_duration_sec,
            f0_rate=f0_rate,
            ignore_silent_frames_thresh=10,
            min_duration_sec_per_segment=15.0,
            max_duration_sec_per_segment=30.0,
            ignore_silent_frames=True,
        )

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
        # Monolithic model-svc.pt checkpoint; see SVS pipeline load_weights for rationale.
        del weights
        weight_path = os.path.join(self.model_path, "model-svc.pt")

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
