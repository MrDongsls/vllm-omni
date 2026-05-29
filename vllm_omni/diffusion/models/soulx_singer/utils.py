import json
from pathlib import Path

import numpy as np
import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
from vllm.logger import init_logger

logger = init_logger(__name__)

# ---------------- utils for model loading ----------------


def _patch_torchaudio_load() -> None:
    """Borrowed from MOSS-TTS-Nano. Patch torchaudio.load to use soundfile
    if torchcodec is unavailable.
    """
    try:
        import torchaudio

        torchaudio  # noqa
        import torchcodec  # noqa: F401

        return
    except Exception:
        pass

    import soundfile as sf

    def _soundfile_load(path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None):
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if frame_offset > 0:
            data = data[frame_offset:]
        if num_frames > 0:
            data = data[:num_frames]
        waveform = torch.from_numpy(data)
        if channels_first:
            waveform = waveform.T
        return waveform, sr

    def _soundfile_save(path, src, sample_rate, channels_first=True, **kwargs):
        wav = src.detach().cpu().float().numpy()
        if channels_first and wav.ndim == 2:
            wav = wav.T
        sf.write(str(path), wav, sample_rate)

    try:
        import torchaudio

        torchaudio.load = _soundfile_load
        torchaudio.save = _soundfile_save
        logger.info("Patched torchaudio.load/save to use soundfile (torchcodec unavailable)")
    except Exception as e:
        logger.warning("Could not patch torchaudio: %s", e)


def load_config(config_path: str | Path) -> DictConfig:
    """
    Load a configuration file and optionally merges it with a base configuration.

    Args:
    config_path (Path): Path to the configuration file.
    """
    # Load the initial configuration from the given path
    config = OmegaConf.load(str(config_path))

    # Check if there is a base configuration specified and merge if necessary
    if config.get("base_config", None) is not None:
        base_config = OmegaConf.load(str(config.get("base_config")))
        config = OmegaConf.merge(base_config, config)

    return config


# ---------------- utils for data processing ----------------


def load_wav(wav_path: str, sample_rate: int) -> torch.Tensor:
    """Load wav file and resample to target sample rate.

    Args:
        wav_path (str): Path to wav file.
        sample_rate (int): Target sample rate.

    Returns:
        torch.Tensor: Waveform tensor with shape (1, T).
    """
    _patch_torchaudio_load()
    waveform, sr = torchaudio.load(wav_path)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    if len(waveform.shape) > 1 and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    return waveform


def f0_to_coarse(f0, f0_bin=361, f0_min=32.7031956625, f0_shift=0):
    """
    Convert continuous F0 values to discrete F0 bins (SIL and C1 - B6, 361 bins).
    args:
        f0: continuous F0 values
        f0_bin: number of F0 bins
        f0_min: minimum F0 value
        f0_shift: shift value for F0 bins
    returns:
        f0_coarse: discrete F0 bins
    """
    is_torch = isinstance(f0, torch.Tensor)
    uv_mask = f0 <= 0

    if is_torch:
        f0_min_tensor = f0.new_tensor(f0_min)
        f0_safe = torch.maximum(f0, f0_min_tensor)
        f0_cents = 1200 * torch.log2(f0_safe / f0_min_tensor)
    else:
        f0_safe = np.maximum(f0, f0_min)
        f0_cents = 1200 * np.log2(f0_safe / f0_min)

    f0_coarse = (f0_cents / 20) + 1

    if is_torch:
        f0_coarse = torch.round(f0_coarse).long()
        f0_coarse = torch.clamp(f0_coarse, min=1, max=f0_bin - 1)
    else:
        f0_coarse = np.rint(f0_coarse).astype(int)
        f0_coarse = np.clip(f0_coarse, 1, f0_bin - 1)

    f0_coarse[uv_mask] = 0

    if f0_shift != 0:
        if is_torch:
            voiced = f0_coarse > 0
            if voiced.any():
                shifted = f0_coarse[voiced] + f0_shift
                f0_coarse[voiced] = torch.clamp(shifted, 1, f0_bin - 1)
        else:
            voiced = f0_coarse > 0
            if np.any(voiced):
                shifted = f0_coarse[voiced] + f0_shift
                f0_coarse[voiced] = np.clip(shifted, 1, f0_bin - 1)

    return f0_coarse


def resolve_pitch_shift(
    *,
    auto_shift: bool,
    manual_shift: int,
    prompt_f0: torch.Tensor | None = None,
    target_f0: torch.Tensor | None = None,
    prompt_note_pitch: torch.Tensor | None = None,
    target_note_pitch: torch.Tensor | None = None,
) -> int:
    """Resolve semitone shift for auto_shift; ``f0_to_coarse(..., f0_shift=shift * 5)`` consumes it."""
    if not auto_shift or manual_shift != 0:
        return int(manual_shift)

    if prompt_note_pitch is not None and target_note_pitch is not None:
        target_pitched = target_note_pitch[target_note_pitch >= 1]
        prompt_pitched = prompt_note_pitch[prompt_note_pitch >= 1]
        if target_pitched.numel() > 0 and prompt_pitched.numel() > 0:
            shift = torch.round(prompt_pitched.median() - target_pitched.median())
            if torch.isfinite(shift):
                return int(shift.item())

    if prompt_f0 is not None and target_f0 is not None:
        target_voiced = target_f0[target_f0 > 0]
        prompt_voiced = prompt_f0[prompt_f0 > 0]
        if target_voiced.numel() > 0 and prompt_voiced.numel() > 0:
            shift = torch.round(
                torch.log2(prompt_voiced.median() / target_voiced.median()) * 1200 / 100
            )
            if torch.isfinite(shift):
                return int(shift.item())

    return 0


class MetadataProcessor:
    """Data processor for SoulX-Singer"""

    def __init__(
        self,
        hop_size: int,
        sample_rate: int,
        phoneset_path: str = "soulxsinger/utils/phoneme/phone_set.json",
        device: str = "cuda",
    ):
        """Initialize data processor.

        Args:
            hop_size (int): Hop size in samples.
            sample_rate (int): Sample rate in Hz.
            phoneset_path (str): Path to phoneme set JSON file.
            device (str): Device to use for tensor operations.
        """
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.device = device
        self.load_phoneme_id_map(phoneset_path)

    def load_phoneme_id_map(self, phoneset_path: str):
        with open(phoneset_path, encoding="utf-8") as f:
            phoneset = json.load(f)
        self.phone2idx = {ph: idx for idx, ph in enumerate(phoneset)}

    def merge_phoneme(self, meta):
        merged_items = []

        duration = [float(x) for x in meta["duration"].split()]
        phoneme = [str(x).replace("<AP>", "<SP>") for x in meta["phoneme"].split()]
        note_pitch = [int(x) for x in meta["note_pitch"].split()]
        note_type = [int(x) if phoneme[i] != "<SP>" else 1 for i, x in enumerate(meta["note_type"].split())]

        for i in range(len(phoneme)):
            if (
                i > 0
                and phoneme[i] == phoneme[i - 1] == "<SP>"
                and note_type[i] == note_type[i - 1]
                and note_pitch[i] == note_pitch[i - 1]
            ):
                merged_items[-1][1] += duration[i]
            else:
                merged_items.append([phoneme[i], duration[i], note_pitch[i], note_type[i]])

        meta["phoneme"] = [x[0] for x in merged_items]
        meta["duration"] = [x[1] for x in merged_items]
        meta["note_pitch"] = [x[2] for x in merged_items]
        meta["note_type"] = [x[3] for x in merged_items]

        return meta

    def preprocess(
        self,
        note_duration: list[float],
        phonemes: list[str],
        note_pitch: list[int],
        note_type: list[int],
    ):
        """
        Insert <BOW> and <EOW> for each note.
        Get aligned indices for each frame.

        Args:
            note_duration: Duration of each note in seconds
            phonemes: Phoneme sequence for each note
            note_pitch: Pitch value for each note
            note_type: Type value for each note

        """
        sample_rate = self.sample_rate
        hop_size = self.hop_size
        duration = sum(note_duration) * sample_rate / hop_size
        mel2note = torch.zeros(int(duration), dtype=torch.long, device=self.device)

        ph_locations = []  # idx at mel scale and length
        new_phonemes = []
        dur_sum = 0

        note2origin = []

        for ph_idx in range(len(phonemes)):
            dur = int(np.round(dur_sum * sample_rate / hop_size))
            dur = min(dur, len(mel2note) - 1)
            new_phonemes.append("<BOW>")
            note2origin.append(ph_idx)
            if phonemes[ph_idx][:3] == "en_":
                en_phs = ["en_" + x for x in phonemes[ph_idx][3:].split("-")] + [
                    "<SEP>"
                ]  # <sep> between en words in one note
                ph_locations.append([dur, max(1, len(en_phs))])
                new_phonemes.extend(en_phs)
                note2origin.extend([ph_idx] * len(en_phs))
            else:
                ph_locations.append([dur, 1])
                new_phonemes.append(phonemes[ph_idx])
                note2origin.append(ph_idx)
            new_phonemes.append("<EOW>")
            note2origin.append(ph_idx)
            dur_sum += note_duration[ph_idx]

        ph_idx = 1
        for idx, (i, j) in enumerate(ph_locations):
            next_phoneme_start = ph_locations[idx + 1][0] if idx < len(ph_locations) - 1 else len(mel2note)
            if i >= len(mel2note) or i + j > len(mel2note):
                break
            if i < len(mel2note) and mel2note[i] > 0:
                logger.warning(f"warning: overlap of {idx}: {mel2note[i]}")
                while i < len(mel2note) and mel2note[i] > 0:
                    i += 1
            mel2note[i] = ph_idx
            k = i + 1
            while k + j < next_phoneme_start:
                mel2note[k : k + j] = torch.arange(ph_idx, ph_idx + j, device=self.device) + 1
                k += j
            mel2note[next_phoneme_start - 1] = ph_idx + j + 1
            ph_idx += j + 2  # <BOW> + ph repeats + <EOW>

        new_phonemes = ["<PAD>"] + new_phonemes
        new_note_pitch = [0] + [note_pitch[k] for k in note2origin]
        new_note_type = [1] + [note_type[k] for k in note2origin]

        return {
            "phoneme": torch.tensor([self.phone2idx[x] for x in new_phonemes], device=self.device).unsqueeze(0),
            "note_pitch": torch.tensor(new_note_pitch, device=self.device).unsqueeze(0),
            "note_type": torch.tensor(new_note_type, device=self.device).unsqueeze(0),
            "mel2note": mel2note.unsqueeze(0),
        }

    def process(self, meta: dict, wav_path: str | None = None) -> dict[str, torch.Tensor | None]:
        meta = self.merge_phoneme(meta)

        item = self.preprocess(
            meta["duration"],
            meta["phoneme"],
            meta["note_pitch"],
            meta["note_type"],
        )

        f0 = [float(x) for x in meta.get("f0", "").split()]
        min_frame = min(item["mel2note"].shape[1], len(f0)) if len(f0) > 0 else item["mel2note"].shape[1]
        item["f0"] = torch.tensor(f0, device=self.device)[:min_frame].unsqueeze(0).float() if len(f0) > 0 else None
        item["mel2note"] = item["mel2note"][:, :min_frame]

        if wav_path is not None:
            waveform = load_wav(wav_path, self.sample_rate)
            item["wav"] = waveform.to(self.device)[:, : min_frame * self.hop_size]

        return item
