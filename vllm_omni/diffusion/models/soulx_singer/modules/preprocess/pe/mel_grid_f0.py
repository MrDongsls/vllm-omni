"""Resample RMVPE F0 (16 kHz hop) onto the SoulX mel frame grid."""

import math
import time
from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import interp1d
from vllm.logger import init_logger

from ..utils import load_mono_audio
from .rmvpe.constants import SAMPLE_RATE

if TYPE_CHECKING:
    from .rmvpe.inference import RMVPE

logger = init_logger(__name__)

RMVPE_HOP = 160


def interpolate_f0_to_mel_grid(
    f0_16k: np.ndarray,
    original_length: int,
    original_sr: int,
    *,
    target_sr: int,
    hop_size: int,
    max_duration: float = 300.0,
) -> np.ndarray:
    """Linearly resample RMVPE frames onto ``target_sr`` / ``hop_size`` mel frames."""
    batch_max_length = int(max_duration * target_sr / hop_size)
    duration_in_seconds = original_length / original_sr
    effective_target_length = int(duration_in_seconds * target_sr)
    original_frames = math.ceil(effective_target_length / hop_size)
    target_frames = min(original_frames, batch_max_length)

    t_16k = np.arange(len(f0_16k)) * (RMVPE_HOP / float(original_sr))
    t_target = np.arange(target_frames) * (hop_size / float(target_sr))

    if len(f0_16k) > 0:
        f_interp = interp1d(
            t_16k,
            f0_16k,
            kind="linear",
            bounds_error=False,
            fill_value=0.0,
            assume_sorted=True,
        )
        f0 = f_interp(t_target)
    else:
        f0 = np.zeros(target_frames)

    if len(f0) != target_frames:
        f0 = f0[:target_frames] if len(f0) > target_frames else np.pad(f0, (0, target_frames - len(f0)))
    return f0.astype(np.float32, copy=False)


def extract_f0_file(
    rmvpe: "RMVPE",
    wav_path: str,
    *,
    target_sr: int,
    hop_size: int,
    f0_path: str | None = None,
    thread: float = 0.03,
    max_duration: float = 300.0,
    verbose: bool = False,
) -> np.ndarray:
    """Run shared RMVPE on a wav file and return F0 on the mel hop grid."""
    if verbose:
        logger.info("[f0] extracting from %s", wav_path)
        t0 = time.perf_counter()

    audio, _ = load_mono_audio(wav_path, target_sr=SAMPLE_RATE)
    f0_16k = rmvpe.forward(audio, sample_rate=SAMPLE_RATE, thread=thread).detach().cpu().numpy()
    f0 = interpolate_f0_to_mel_grid(
        f0_16k,
        original_length=audio.shape[-1],
        original_sr=SAMPLE_RATE,
        target_sr=target_sr,
        hop_size=hop_size,
        max_duration=max_duration,
    )

    if verbose:
        voiced_ratio = float(np.mean(f0 > 0)) if len(f0) else 0.0
        logger.info(
            "[f0] done: frames=%s voiced_ratio=%.3f time=%.3fs",
            len(f0),
            voiced_ratio,
            time.perf_counter() - t0,
        )
    if f0_path is not None:
        np.save(f0_path, f0)
    return f0
