"""SoulX preprocess module."""

import json
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportAudioInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.soulx_singer.modules.preprocess.stack import SoulXPreprocessStack
from vllm_omni.diffusion.models.soulx_singer.modules.preprocess.utils import load_mono_audio, resample_mono
from vllm_omni.diffusion.models.soulx_singer.preprocess.metadata_utils import (
    SegmentMetadata,
    convert_metadata,
    merge_short_segments,
)
from vllm_omni.diffusion.models.soulx_singer.preprocess.payload import SOULX_SVC_KIND, SOULX_SVS_KIND
from vllm_omni.diffusion.models.soulx_singer.preprocess.weights import (
    preprocess_weight_paths,
    resolve_preprocess_weights_root,
)
from vllm_omni.diffusion.models.soulx_singer.utils import load_config, load_wav

logger = init_logger(__name__)


class SoulXPreprocessModule(nn.Module, SupportAudioInput, SupportsComponentDiscovery):
    """Lazy-loaded preprocess stack integrated with vLLM-Omni diffusion lifecycle."""

    support_audio_input: ClassVar[bool] = True
    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = [
        "stack.vocal_sep",
        "stack.lyric",
        "stack.rosvot",
    ]
    _vae_modules: ClassVar[list[str]] = []
    _resident_modules: ClassVar[list[str]] = ["stack.rmvpe"]

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        *,
        vocal_sep: bool = True,
        midi_transcribe: bool = True,
        max_merge_duration_ms: int = 60000,
        verbose: bool = False,
        extra_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.od_config = od_config
        extra_args = extra_args or {}
        weights_root = resolve_preprocess_weights_root(od_config, extra_args)
        weights = preprocess_weight_paths(weights_root)
        hf_config = load_config(str(Path(od_config.model) / "config.yaml"))
        audio_config = hf_config.audio
        device = str(get_local_device())

        self.vocal_sep = vocal_sep
        self.midi_transcribe = midi_transcribe
        self.max_merge_duration_ms = max_merge_duration_ms
        self.verbose = verbose
        self.target_sr = int(audio_config.sample_rate)
        self.hop_size = int(audio_config.hop_size)

        self.stack = SoulXPreprocessStack(
            weights,
            device,
            target_sr=self.target_sr,
            hop_size=self.hop_size,
            verbose=verbose,
        )

    def _extract_vocal(
        self,
        audio: str | tuple[np.ndarray, int],
        *,
        vocal_sep: bool | None = None,
    ) -> tuple[np.ndarray, int]:
        use_sep = self.vocal_sep if vocal_sep is None else vocal_sep
        if use_sep:
            return self.stack.extract_vocal(audio)
        return load_mono_audio(audio)

    def extract_f0(self, vocal: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.stack.extract_f0(vocal, sample_rate)

    def run_svs_metadata(
        self,
        audio_source: str | tuple[np.ndarray, int],
        *,
        language: str = "Mandarin",
        vocal_sep: bool | None = None,
        max_merge_duration_ms: int | None = None,
    ) -> list[dict[str, Any]]:
        """Path-based F0 extraction plus merged wav sidecars for SVS metadata."""
        if isinstance(audio_source, tuple):
            vocal, sample_rate = audio_source
            origin_wav_fn = ""
        else:
            vocal, sample_rate = self._extract_vocal(str(audio_source), vocal_sep=vocal_sep)
            origin_wav_fn = str(audio_source)

        with tempfile.TemporaryDirectory(prefix="soulx_preprocess_") as work_dir:
            work = Path(work_dir)
            vocal_path = work / "vocal.wav"
            sf.write(vocal_path, vocal, sample_rate)

            vocal_f0_path = str(vocal_path).replace(".wav", "_f0.npy")
            vocal_f0 = self.stack.extract_f0(str(vocal_path), sample_rate=sample_rate, f0_path=vocal_f0_path)

            if not self.midi_transcribe:
                end_ms = int(len(vocal) / sample_rate * 1000)
                item = SegmentMetadata(
                    item_name="full",
                    wav_fn=str(vocal_path),
                    language=language,
                    start_time_ms=0,
                    end_time_ms=end_ms,
                    note_text=["<SP>"],
                    note_dur=[end_ms / 1000.0],
                    note_pitch=[0],
                    note_type=[1],
                )
                return [convert_metadata(item)]

            segments = self.stack.ensure_segmenter().forward(
                vocal,
                sample_rate,
                vocal_f0,
                base_name="vocal",
                origin_wav_fn=origin_wav_fn,
                verbose=self.verbose,
            )
            cut_dir = work / "cut_wavs"
            cut_dir.mkdir(parents=True, exist_ok=True)
            metadata: list[dict[str, Any]] = []
            lyric = self.stack.ensure_lyric()
            rosvot = self.stack.ensure_rosvot()

            for seg in segments:
                seg_key = seg["item_name"]
                seg_wav_path = cut_dir / f"{seg_key}.wav"
                sf.write(seg_wav_path, seg["wav"], seg["sample_rate"])
                seg_f0_path = str(seg_wav_path).replace(".wav", "_f0.npy")
                self.stack.extract_f0(str(seg_wav_path), sample_rate=seg["sample_rate"], f0_path=seg_f0_path)

                words, durs = lyric.forward(
                    str(seg_wav_path),
                    language,
                    sample_rate=seg["sample_rate"],
                )
                seg_item = {
                    "item_name": seg_key,
                    "wav_fn": str(seg_wav_path),
                    "start_time_ms": seg["start_time_ms"],
                    "end_time_ms": seg["end_time_ms"],
                    "origin_wav_fn": origin_wav_fn or str(vocal_path),
                    "words": words,
                    "word_durs": durs,
                    "language": language,
                }
                metadata.append(rosvot.transcribe(seg_item, segment_info=seg_item, verbose=self.verbose))

            long_cut_dir = work / "long_cut_wavs"
            merged = merge_short_segments(
                vocal,
                sample_rate,
                metadata,
                long_cut_dir,
                max_duration_ms=max_merge_duration_ms or self.max_merge_duration_ms,
            )

            final_metadata: list[dict[str, Any]] = []
            for item in merged:
                merged_f0_path = item.wav_fn.replace(".wav", "_f0.npy")
                self.stack.extract_f0(item.wav_fn, sample_rate=sample_rate, f0_path=merged_f0_path)
                final_metadata.append(convert_metadata(item))
            return final_metadata

    @staticmethod
    def build_svs_payload_from_paths(
        *,
        prompt_metadata_path: str,
        target_metadata_path: str,
        audio_path: str,
        metadata_processor,
    ) -> dict[str, Any]:
        with open(prompt_metadata_path, encoding="utf-8") as f:
            prompt_metadata = json.load(f)
        with open(target_metadata_path, encoding="utf-8") as f:
            target_metadata = json.load(f)
        if not prompt_metadata:
            raise ValueError("prompt_metadata is empty")
        if not target_metadata:
            raise ValueError("target_metadata is empty")

        prompt_meta = metadata_processor.process(prompt_metadata[0], audio_path)
        return {
            "kind": SOULX_SVS_KIND,
            "prompt_meta": prompt_meta,
            "target_meta_list": target_metadata,
        }

    def build_svs_payload_from_audio(
        self,
        *,
        prompt_audio: str | tuple[np.ndarray, int],
        target_audio: str | tuple[np.ndarray, int],
        metadata_processor,
        language: str = "Mandarin",
        vocal_sep: bool | None = None,
        prompt_vocal_sep: bool | None = None,
        target_vocal_sep: bool | None = None,
        prompt_max_merge_duration_ms: int | None = None,
        target_max_merge_duration_ms: int | None = None,
    ) -> dict[str, Any]:
        p_sep = prompt_vocal_sep if prompt_vocal_sep is not None else vocal_sep
        t_sep = target_vocal_sep if target_vocal_sep is not None else vocal_sep
        p_merge = (
            prompt_max_merge_duration_ms if prompt_max_merge_duration_ms is not None else self.max_merge_duration_ms
        )
        t_merge = (
            target_max_merge_duration_ms if target_max_merge_duration_ms is not None else self.max_merge_duration_ms
        )
        prompt_list = self.run_svs_metadata(
            prompt_audio,
            language=language,
            vocal_sep=p_sep,
            max_merge_duration_ms=p_merge,
        )
        target_list = self.run_svs_metadata(
            target_audio,
            language=language,
            vocal_sep=t_sep,
            max_merge_duration_ms=t_merge,
        )
        if not prompt_list or not target_list:
            raise ValueError("SVS preprocess produced empty metadata")

        # Prompt wav is trimmed to mel2note length, not the metadata time window.
        if isinstance(prompt_audio, str):
            prompt_meta = metadata_processor.process(prompt_list[0], prompt_audio)
        else:
            prompt_meta = metadata_processor.process(prompt_list[0], None)
            prompt_vocal, prompt_sr = self._extract_vocal(prompt_audio, vocal_sep=vocal_sep)
            if prompt_sr != metadata_processor.sample_rate:
                prompt_vocal = resample_mono(
                    prompt_vocal,
                    orig_sr=prompt_sr,
                    target_sr=metadata_processor.sample_rate,
                )
            max_samples = prompt_meta["mel2note"].shape[1] * metadata_processor.hop_size
            segment_wav = np.asarray(prompt_vocal[:max_samples], dtype=np.float32)
            prompt_meta["wav"] = torch.from_numpy(segment_wav).unsqueeze(0).float().to(metadata_processor.device)

        return {
            "kind": SOULX_SVS_KIND,
            "prompt_meta": prompt_meta,
            "target_meta_list": target_list,
        }

    @staticmethod
    def build_svc_payload_from_paths(
        *,
        prompt_wav_path: str,
        target_wav_path: str,
        prompt_f0_path: str,
        target_f0_path: str,
        sample_rate: int,
        device: torch.device | str,
    ) -> dict[str, Any]:
        prompt_wav = load_wav(prompt_wav_path, sample_rate).to(device)
        target_wav = load_wav(target_wav_path, sample_rate).to(device)
        prompt_f0 = torch.from_numpy(np.load(prompt_f0_path)).unsqueeze(0).to(device)
        target_f0 = torch.from_numpy(np.load(target_f0_path)).unsqueeze(0).to(device)
        return {
            "kind": SOULX_SVC_KIND,
            "prompt_wav": prompt_wav,
            "target_wav": target_wav,
            "prompt_f0": prompt_f0,
            "target_f0": target_f0,
        }

    def build_svc_payload_from_audio(
        self,
        *,
        prompt_audio: str | tuple[np.ndarray, int],
        target_audio: str | tuple[np.ndarray, int],
        sample_rate: int,
        device: torch.device | str,
        vocal_sep: bool | None = None,
    ) -> dict[str, Any]:
        prompt_vocal, prompt_sr = self._extract_vocal(prompt_audio, vocal_sep=vocal_sep)
        target_vocal, target_sr = self._extract_vocal(target_audio, vocal_sep=vocal_sep)

        if prompt_sr != sample_rate:
            prompt_vocal = resample_mono(prompt_vocal, orig_sr=prompt_sr, target_sr=sample_rate)
        if target_sr != sample_rate:
            target_vocal = resample_mono(target_vocal, orig_sr=target_sr, target_sr=sample_rate)

        prompt_f0 = self.extract_f0(prompt_vocal, sample_rate)
        target_f0 = self.extract_f0(target_vocal, sample_rate)

        return {
            "kind": SOULX_SVC_KIND,
            "prompt_wav": torch.from_numpy(prompt_vocal).unsqueeze(0).to(device),
            "target_wav": torch.from_numpy(target_vocal).unsqueeze(0).to(device),
            "prompt_f0": torch.from_numpy(prompt_f0).unsqueeze(0).to(device),
            "target_f0": torch.from_numpy(target_f0).unsqueeze(0).to(device),
        }


# Backward-compatible alias for callers that still use the old name.
SoulXPreprocessPipeline = SoulXPreprocessModule
