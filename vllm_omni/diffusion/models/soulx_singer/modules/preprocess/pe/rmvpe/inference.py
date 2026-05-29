from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample
from vllm.logger import init_logger

from ...utils import load_pt_state_dict, resample_align_curve
from .constants import CONST, MEL_FMAX, MEL_FMIN, N_CLASS, N_MELS, SAMPLE_RATE, WINDOW_LENGTH
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_f0

logger = init_logger(__name__)


class RMVPE(nn.Module):
    def __init__(
        self,
        model_path: str | Path,
        hop_length: int = 160,
        device: str | torch.device | None = None,
    ):
        super().__init__()
        self.hop_length = hop_length
        self._model_path = str(model_path)
        self.encoder = E2E0(4, 1, (2, 2))
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        )
        self._resamplers = nn.ModuleDict()
        cents_mapping = 20 * np.arange(N_CLASS) + CONST
        self.register_buffer(
            "cents_mapping",
            torch.from_numpy(np.pad(cents_mapping, (4, 4))).float(),
            persistent=False,
        )
        if model_path:
            self.load_checkpoint(str(model_path), device=device)
        elif device is not None:
            self.to(device)

    def load_checkpoint(
        self,
        model_path: str | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        path = model_path or self._model_path
        load_pt_state_dict(self.encoder, path, state_key="model", strict=False)
        self.eval()
        if device is not None:
            self.to(device)
        logger.info("Loaded RMVPE checkpoint from %s", path)

    def _resample_audio(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if sample_rate == SAMPLE_RATE:
            return audio
        key = str(sample_rate)
        if key not in self._resamplers:
            self._resamplers[key] = Resample(sample_rate, SAMPLE_RATE, lowpass_filter_width=128)
        return self._resamplers[key].to(audio.device)(audio)

    @torch.no_grad()
    def mel2hidden(self, mel: torch.Tensor) -> torch.Tensor:
        n_frames = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="constant")
        return self.encoder(mel)[:, :n_frames]

    def decode_hidden(self, hidden: torch.Tensor, *, thred: float = 0.03) -> np.ndarray:
        """Match upstream ``RMVPE.decode`` / ``to_local_average_cents`` in f0_extraction.py."""
        salience = hidden.detach().float().cpu().numpy()
        if salience.ndim == 3:
            salience = salience[0]
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center = center + 4

        todo_salience = []
        todo_cents_mapping = []
        cents_mapping = self.cents_mapping.detach().cpu().numpy()
        for idx in range(salience.shape[0]):
            start = center[idx] - 4
            end = center[idx] + 5
            todo_salience.append(salience[idx, start:end])
            todo_cents_mapping.append(cents_mapping[start:end])

        todo_salience = np.asarray(todo_salience)
        todo_cents_mapping = np.asarray(todo_cents_mapping)
        product_sum = np.sum(todo_salience * todo_cents_mapping, axis=1)
        weight_sum = np.sum(todo_salience, axis=1)
        cents = product_sum / np.maximum(weight_sum, 1e-8)

        maxx = np.max(salience, axis=1)
        cents[maxx <= thred] = 0.0
        f0 = 10 * (2 ** (cents / 1200))
        f0[f0 == 10] = 0
        return f0.astype(np.float32)

    @staticmethod
    def postprocess(
        f0: np.ndarray,
        *,
        fmin: float = 50,
        fmax: float = 1000,
        min_gap: int = 2,
    ) -> np.ndarray:
        f0 = f0.copy()
        f0[f0 < fmin] = 0
        f0[f0 > fmax] = 0
        for idx in range(f0.shape[0] - min_gap - 1):
            if f0[idx] == 0 and f0[idx + min_gap + 1] == 0 and np.sum(f0[idx : idx + min_gap + 2]) > 0:
                f0[idx : idx + min_gap + 2] = 0
        return f0

    @torch.no_grad()
    def forward(
        self,
        audio: torch.Tensor | np.ndarray,
        *,
        sample_rate: int = SAMPLE_RATE,
        thred: float = 0.03,
    ) -> torch.Tensor:
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(np.asarray(audio, dtype=np.float32))
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        audio = audio.float().to(next(self.parameters()).device)
        mel = self.mel_extractor(self._resample_audio(audio, sample_rate), center=True)
        f0 = self.decode_hidden(self.mel2hidden(mel), thred=thred)
        return torch.from_numpy(f0).squeeze(0)

    @torch.no_grad()
    def get_pitch_batch(
        self,
        waveforms: torch.Tensor,
        sample_rate: int,
        hop_size: int,
        lengths: list[int],
        *,
        fmin: float = 50,
        fmax: float = 1000,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        waveforms = waveforms.float().to(next(self.parameters()).device)
        mel = self.mel_extractor(self._resample_audio(waveforms, sample_rate), center=True)
        f0 = to_local_average_f0(self.mel2hidden(mel))

        f0s_res: list[np.ndarray] = []
        uvs_res: list[np.ndarray] = []
        time_step = hop_size / sample_rate
        for idx in range(f0.shape[0]):
            frame = self.postprocess(f0[idx], fmin=fmin, fmax=fmax, min_gap=6)
            uv = frame == 0
            length = lengths[idx]
            f0_res = resample_align_curve(frame, 0.01, time_step, length)
            uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
            f0_res[uv_res] = 0
            f0s_res.append(f0_res)
            uvs_res.append(uv_res)
        return f0s_res, uvs_res
