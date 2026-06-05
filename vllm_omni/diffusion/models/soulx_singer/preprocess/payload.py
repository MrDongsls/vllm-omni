"""SoulX-Singer payload schema (guidance/precomputed tables, dummy builders)."""

from typing import Any

import torch

SOULX_PRECOMPUTED_KEYS_BY_KIND: dict[str, tuple[str, ...]] = {
    "svs": ("prompt_metadata_path", "target_metadata_path", "audio_path"),
    "svc": ("prompt_wav_path", "target_wav_path", "prompt_f0_path", "target_f0_path"),
}


def has_precomputed(extra_args: dict[str, Any], kind: str) -> bool:
    keys = SOULX_PRECOMPUTED_KEYS_BY_KIND[kind]
    return all(extra_args.get(key) for key in keys)


def build_dummy_payload(kind: str, device: torch.device) -> dict[str, Any]:
    mel_frames = 4
    hop_size = 480
    n_fft = 1920
    wav_samples = max(mel_frames * hop_size, n_fft)
    if kind == "svc":
        voiced_f0 = torch.full((1, mel_frames), 100.0, device=device, dtype=torch.float64)
        return {
            "kind": "svc",
            "prompt_wav": torch.zeros(1, wav_samples, device=device, dtype=torch.float32),
            "target_wav": torch.zeros(1, wav_samples, device=device, dtype=torch.float32),
            "prompt_f0": voiced_f0.clone(),
            "target_f0": voiced_f0.clone(),
        }
    return {
        "kind": "svs",
        "prompt_meta": {
            "phoneme": torch.zeros(1, 2, device=device, dtype=torch.long),
            "note_pitch": torch.zeros(1, mel_frames, device=device, dtype=torch.long),
            "note_type": torch.zeros(1, mel_frames, device=device, dtype=torch.long),
            "mel2note": torch.zeros(1, mel_frames, device=device, dtype=torch.long),
            "f0": torch.zeros(1, mel_frames, device=device, dtype=torch.float32),
            "wav": torch.zeros(1, wav_samples, device=device, dtype=torch.float32),
        },
        "target_meta_list": [
            {
                "text": "dummy",
                "phoneme": " ".join(["<SP>"] * mel_frames),
                "f0": " ".join(["100"] * mel_frames),
                "language": "Mandarin",
                "note_pitch": " ".join(["0"] * mel_frames),
                "note_type": " ".join(["1"] * mel_frames),
                "duration": " ".join(["1"] * mel_frames),
                "time": [0, max(int(wav_samples / 24000 * 1000), 100)],
            }
        ],
    }


__all__ = ["SOULX_PRECOMPUTED_KEYS_BY_KIND", "has_precomputed", "build_dummy_payload"]
