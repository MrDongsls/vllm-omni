import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks


def _htk_mel_filter_bank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: float | None,
) -> torch.Tensor:
    """HTK mel filterbank used by RMVPE (``librosa.filters.mel(..., htk=True)``)."""
    if fmax is None:
        fmax = float(sr) / 2.0
    return melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=float(fmin),
        f_max=float(fmax),
        n_mels=n_mels,
        sample_rate=sr,
        mel_scale="htk",
        norm=None,
    ).T


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = _htk_mel_filter_bank(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        ).numpy()
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.register_buffer("hann_window", torch.hann_window(win_length))
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.clamp = clamp

    def forward(self, audio, center=True):
        if center:
            pad_left = self.win_length // 2
            pad_right = (self.win_length + 1) // 2
            audio = F.pad(audio, (pad_left, pad_right))

        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.hann_window.to(audio.device),
            center=False,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.square() + fft.imag.square())
        mel_basis = self.mel_basis.to(audio.device)
        mel_output = torch.matmul(mel_basis, magnitude)
        return torch.log(torch.clamp(mel_output, min=self.clamp))
