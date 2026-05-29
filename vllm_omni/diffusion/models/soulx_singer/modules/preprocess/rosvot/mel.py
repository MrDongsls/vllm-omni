import math

import numpy as np
import torch
import torch.nn as nn

from vllm_omni.utils.audio import mel_filter_bank


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log10(torch.clamp(x, min=clip_val) * C)


class MelNet(nn.Module):
    def __init__(self, hparams) -> None:
        super().__init__()
        self.n_fft = hparams["fft_size"]
        self.num_mels = hparams["audio_num_mel_bins"]
        self.sampling_rate = hparams["audio_sample_rate"]
        self.hop_size = hparams["hop_size"]
        self.win_size = hparams["win_size"]
        self.fmin = hparams["fmin"]
        self.fmax = hparams["fmax"]

        mel = mel_filter_bank(
            sr=self.sampling_rate,
            n_fft=self.n_fft,
            n_mels=self.num_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        ).numpy()
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())
        self.register_buffer("hann_window", torch.hann_window(self.win_size))

    def forward(self, y: torch.Tensor | np.ndarray, center: bool = False) -> torch.Tensor:
        if isinstance(y, np.ndarray):
            y = torch.as_tensor(y, dtype=torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        y = y.clamp(min=-1.0, max=1.0)

        pad_length = math.ceil(y.shape[1] / self.hop_size) * self.hop_size - y.shape[1]
        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            [int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2 + pad_length)],
            mode="reflect",
        ).squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(self.mel_basis, spec)
        spec = dynamic_range_compression_torch(spec)
        return spec.transpose(1, 2)
