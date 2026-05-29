from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from vllm.logger import init_logger
from vllm.multimodal.media.audio import load_audio

from ..utils import load_roformer_yaml_config
from .mel_band_roformer import MelBandRoformer

logger = init_logger(__name__)


class MelBandRoformerSeparator(nn.Module):
    """Chunked vocal separation wrapper around MelBandRoformer."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path | None = None,
        *,
        chunk_length_sec: float = 5,
        use_half: bool = True,
    ):
        super().__init__()
        config: DictConfig = load_roformer_yaml_config(config_path)
        config = OmegaConf.merge(
            config,
            {
                "inference": {
                    "chunk_size": int(chunk_length_sec * config.audio.sample_rate),
                    "use_amp": False,
                }
            },
        )

        model_cfg = OmegaConf.to_container(config.model, resolve=True)
        if not isinstance(model_cfg, dict):
            raise TypeError(f"Expected dict model config, got {type(model_cfg)}")
        self.roformer = MelBandRoformer(**model_cfg)
        self._checkpoint_path = str(checkpoint_path) if checkpoint_path else None
        if checkpoint_path is not None:
            self.load_checkpoint(str(checkpoint_path), use_half=use_half)
        else:
            self.roformer.eval()

        self.config = config
        self.sample_rate = int(config.audio.sample_rate)
        chunk_size = int(config.inference.get("chunk_size", config.audio.chunk_size))
        fade = chunk_size // 10
        window = torch.ones(chunk_size)
        window[:fade] = torch.linspace(0, 1, fade)
        window[-fade:] = torch.linspace(1, 0, fade)
        self.register_buffer("_window", window, persistent=False)

    def load_checkpoint(self, checkpoint_path: str, *, use_half: bool = True) -> None:
        self.roformer.load_checkpoint(checkpoint_path)
        if use_half:
            self.roformer.half()
        self.roformer.eval()
        self._checkpoint_path = checkpoint_path
        logger.info("Loaded MelBandRoformerSeparator checkpoint from %s", checkpoint_path)

    @property
    def chunk_size(self) -> int:
        return int(self.config.inference.get("chunk_size", self.config.audio.chunk_size))

    @property
    def num_overlap(self) -> int:
        return int(self.config.inference.num_overlap)

    @property
    def use_amp(self) -> bool:
        return bool(self.config.inference.get("use_amp", True))

    @property
    def normalize_input(self) -> bool:
        return bool(self.config.inference.get("normalize"))

    @property
    def num_channels(self) -> int:
        return int(self.config.audio.get("num_channels", 1))

    def _prepare_mix(self, mix: np.ndarray) -> tuple[np.ndarray, tuple[float, float] | None]:
        if mix.ndim == 1:
            mix = mix[np.newaxis, :]
        if self.num_channels == 2 and mix.shape[0] == 1:
            mix = np.repeat(mix, 2, axis=0)
        if not self.normalize_input:
            return mix, None
        mono = mix.mean(0)
        mean, std = float(mono.mean()), float(mono.std())
        return (mix - mean) / std, (mean, std)

    @staticmethod
    def _select_vocal_stem(recon: torch.Tensor) -> torch.Tensor:
        """Return the vocals stem as ``[channels, time]``."""
        if recon.ndim == 4:
            return recon[0, 0]
        if recon.ndim == 3:
            return recon[0]
        if recon.ndim == 2:
            return recon
        raise ValueError(f"Unexpected Roformer output shape: {tuple(recon.shape)}")

    @torch.inference_mode()
    def forward(self, mix: torch.Tensor) -> torch.Tensor:
        if mix.ndim == 1:
            mix = mix.unsqueeze(0)
        if self.num_channels == 2 and mix.shape[0] == 1:
            mix = mix.repeat(2, 1)

        chunk_size = self.chunk_size
        step = chunk_size // self.num_overlap
        fade = chunk_size // 10
        border = chunk_size - step
        n_samples = mix.shape[-1]
        window = self._window.to(mix.device)

        if n_samples > 2 * border and border > 0:
            mix = nn.functional.pad(mix, (border, border), mode="reflect")

        out = torch.zeros_like(mix)
        weight = torch.zeros(mix.shape[-1], device=mix.device, dtype=mix.dtype)
        pos = 0
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            while pos < mix.shape[-1]:
                part = mix[:, pos : pos + chunk_size]
                seg_len = part.shape[-1]
                pad_mode = "reflect" if seg_len > chunk_size // 2 else "constant"
                part = nn.functional.pad(part, (0, chunk_size - seg_len), mode=pad_mode, value=0)
                stem = self._select_vocal_stem(self.roformer(part.unsqueeze(0)))
                win = window.clone()
                if pos == 0:
                    win[:fade] = 1
                elif pos + step >= n_samples:
                    win[-fade:] = 1
                out[:, pos : pos + seg_len] += stem[:, :seg_len] * win[:seg_len]
                weight[pos : pos + seg_len] += win[:seg_len]
                pos += step

        out = out / weight
        if n_samples > 2 * border and border > 0:
            out = out[..., border:-border]
        return out

    @torch.inference_mode()
    def separate_mono(self, mix: np.ndarray, device: torch.device) -> np.ndarray:
        mix_np, norm = self._prepare_mix(mix)
        vocal = self.forward(torch.as_tensor(mix_np, dtype=torch.float32, device=device))
        if norm is not None:
            mean, std = norm
            vocal = vocal * std + mean
        if vocal.ndim == 2:
            vocal = vocal.mean(0)
        return vocal.float().cpu().numpy().astype(np.float32)

    def separate_path(self, input_path: str, device: torch.device) -> tuple[np.ndarray, int]:
        mix, _ = load_audio(input_path, sr=self.sample_rate, mono=False)
        mix = np.asarray(mix, dtype=np.float32)
        return self.separate_mono(mix, device), self.sample_rate
