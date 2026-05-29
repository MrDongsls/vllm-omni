"""SoulX-Singer preprocess payload contract and stage handoff helpers."""


import pickle
from typing import Any

import numpy as np
import torch

SOULX_PREPROCESSED_KEY = "soulx_preprocessed"
SOULX_PREPROCESSED_BLOB_KEY = "soulx_preprocessed_blob"
SOULX_SVC_KIND = "svc"
SOULX_SVS_KIND = "svs"

SOULX_PRECOMPUTED_KEYS: dict[str, tuple[str, ...]] = {
    SOULX_SVS_KIND: ("prompt_metadata_path", "target_metadata_path", "audio_path"),
    SOULX_SVC_KIND: ("prompt_wav_path", "target_wav_path", "prompt_f0_path", "target_f0_path"),
}


def has_precomputed(extra_args: dict[str, Any], kind: str) -> bool:
    return all(extra_args.get(key) for key in SOULX_PRECOMPUTED_KEYS[kind])


def get_soulx_preprocessed_payload(prompt: dict[str, Any]) -> dict[str, Any] | None:
    additional = prompt.get("additional_information") or {}
    payload = additional.get(SOULX_PREPROCESSED_KEY)
    return payload if isinstance(payload, dict) else None


def attach_preprocessed(prompt: dict[str, Any], payload: dict[str, Any]) -> None:
    prompt.setdefault("additional_information", {})[SOULX_PREPROCESSED_KEY] = payload


def build_dummy_payload(kind: str, device: torch.device) -> dict[str, Any]:
    mel_frames = 4
    hop_size = 480
    n_fft = 1920
    wav_samples = max(mel_frames * hop_size, n_fft)
    if kind == SOULX_SVC_KIND:
        voiced_f0 = torch.full((1, mel_frames), 100.0, device=device, dtype=torch.float64)
        return {
            "kind": SOULX_SVC_KIND,
            "prompt_wav": torch.zeros(1, wav_samples, device=device, dtype=torch.float32),
            "target_wav": torch.zeros(1, wav_samples, device=device, dtype=torch.float32),
            "prompt_f0": voiced_f0.clone(),
            "target_f0": voiced_f0.clone(),
        }
    return {
        "kind": SOULX_SVS_KIND,
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


def encode_ipc(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    arr = np.frombuffer(blob, dtype=np.uint8)
    return {SOULX_PREPROCESSED_BLOB_KEY: torch.from_numpy(arr.copy())}


def decode_ipc(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    payload = data.get(SOULX_PREPROCESSED_KEY)
    if isinstance(payload, dict):
        return payload
    blob = data.get(SOULX_PREPROCESSED_BLOB_KEY)
    if isinstance(blob, torch.Tensor):
        return pickle.loads(blob.detach().cpu().numpy().tobytes())
    return None


def consume_payload(
    req: Any,
    expected_kind: str,
    device: torch.device | str,
) -> dict[str, Any]:
    """Read and validate ``soulx_preprocessed`` from the first diffusion prompt."""
    prompts = getattr(req, "prompts", None) or []
    if not prompts:
        raise ValueError(f"SoulX-Singer {expected_kind} request has no prompts.")
    prompt = prompts[0]
    if isinstance(prompt, str):
        raise ValueError(
            f"SoulX-Singer {expected_kind} forward requires pre_process_func output on the prompt."
        )
    payload = get_soulx_preprocessed_payload(prompt)
    if payload is None or payload.get("kind") != expected_kind:
        raise ValueError(
            f"SoulX-Singer {expected_kind} forward requires additional_information['soulx_preprocessed'] "
            "produced by pre_process_func."
        )
    return relocate_tensors(payload, device)


def relocate_tensors(payload: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    def _move(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, dict):
            return {key: _move(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [_move(item) for item in obj]
        return obj

    return _move(payload)


def extract_from_stage_output(source_output: Any) -> dict[str, Any] | None:
    outputs = getattr(source_output, "outputs", None)
    if outputs is None:
        return None
    for item in outputs if isinstance(outputs, list) else [outputs]:
        for field in ("multimodal_output", "data"):
            payload = decode_ipc(getattr(item, field, None))
            if payload is not None:
                return payload
    return None


def attach_to_diffusion_prompt(
    *,
    payload: dict[str, Any],
    original_prompt: Any,
    expected_kind: str,
) -> dict[str, Any]:
    if payload.get("kind") != expected_kind:
        raise ValueError(
            f"Expected preprocess payload kind {expected_kind!r}, got {payload.get('kind')!r}."
        )
    diffusion_prompt = (
        dict(original_prompt)
        if isinstance(original_prompt, dict)
        else {"prompt": str(original_prompt or "soulx-singer")}
    )
    attach_preprocessed(diffusion_prompt, payload)
    return diffusion_prompt
