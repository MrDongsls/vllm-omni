"""Central IPC codec for SoulX-Singer payload handoff (cross-stage / cross-worker / KV metadata)."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch

# These two string keys define the wire protocol for SoulX payload handoff.
SOULX_PREPROCESSED_KEY = "soulx_preprocessed"
SOULX_PREPROCESSED_BLOB_KEY = "soulx_preprocessed_blob"


def encode_ipc(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Serialize a SoulX payload dict into a single tensor blob for IPC."""
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    arr = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    return {SOULX_PREPROCESSED_BLOB_KEY: torch.from_numpy(arr.copy())}


def decode_ipc(data: dict[str, Any] | None) -> dict[str, Any] | None:
    """Best-effort decode of a SoulX payload from any of the shapes vLLM uses.

    Supports:
    - Direct { "soulx_preprocessed": dict }
    - Blob form   { "soulx_preprocessed_blob": tensor }
    - Nested in kv_metadata
    """
    if not isinstance(data, dict):
        return None
    payload = data.get(SOULX_PREPROCESSED_KEY)
    if isinstance(payload, dict):
        return payload
    blob = data.get(SOULX_PREPROCESSED_BLOB_KEY)
    if blob is None:
        kv = data.get("kv_metadata")
        if isinstance(kv, dict):
            blob = kv.get(SOULX_PREPROCESSED_BLOB_KEY)
    if isinstance(blob, torch.Tensor):
        buffer = io.BytesIO(blob.detach().cpu().numpy().tobytes())
        return torch.load(buffer, weights_only=False)
    return None


def _payload_from_field(field_val: Any) -> dict[str, Any] | None:
    payload = decode_ipc(field_val)
    if payload is not None:
        return payload
    if isinstance(field_val, dict):
        direct = field_val.get(SOULX_PREPROCESSED_KEY)
        if isinstance(direct, dict):
            return direct
    return None


def extract_from_stage_output(source_output: Any) -> dict[str, Any] | None:
    """Walk a stage output object and return the first decoded SoulX payload.

    Used by the stage-0 → stage-1 handoff processors.
    Tries common fields where multimodal / diffusion payloads are attached.
    """
    outputs = getattr(source_output, "outputs", None)
    if outputs:
        for item in outputs if isinstance(outputs, list) else [outputs]:
            for field in ("multimodal_output", "data", "custom_output"):
                payload = _payload_from_field(getattr(item, field, None))
                if payload is not None:
                    return payload
    for attr in ("multimodal_output", "custom_output"):
        payload = _payload_from_field(getattr(source_output, attr, None))
        if payload is not None:
            return payload
    return None


def get_soulx_preprocessed_payload(prompt: dict[str, Any]) -> dict[str, Any] | None:
    """Extract attached payload from a diffusion request prompt dict."""
    additional = prompt.get("additional_information") or {}
    payload = additional.get(SOULX_PREPROCESSED_KEY)
    return payload if isinstance(payload, dict) else None


def attach_to_diffusion_prompt(
    *,
    payload: dict[str, Any],
    original_prompt: Any,
    expected_kind: str,
) -> dict[str, Any]:
    """Prepare a prompt dict for the diffusion stage, attaching the payload.

    Validates that the payload kind matches expectation.
    Used by stage input processors when handing off from preprocess to DiT.
    """
    if payload.get("kind") != expected_kind:
        raise ValueError(f"Expected preprocess payload kind {expected_kind!r}, got {payload.get('kind')!r}.")
    diffusion_prompt = (
        dict(original_prompt)
        if isinstance(original_prompt, dict)
        else {"prompt": str(original_prompt or "soulx-singer")}
    )
    diffusion_prompt.setdefault("additional_information", {})[SOULX_PREPROCESSED_KEY] = payload
    return diffusion_prompt


def relocate_tensors(payload: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    """Recursively move all tensors inside the payload to the target device.

    Called after decode when the payload is about to be consumed by the
    diffusion model (which may live on a different device / rank).
    """
    if isinstance(payload, torch.Tensor):
        return payload.to(device, non_blocking=True)
    if isinstance(payload, dict):
        return {key: relocate_tensors(value, device) for key, value in payload.items()}
    if isinstance(payload, list):
        return [relocate_tensors(item, device) for item in payload]
    return payload


def consume_payload(
    req: Any,
    expected_kind: str,
    device: torch.device | str,
) -> dict[str, Any]:
    """Read the payload attached by pre_process_func, validate the kind,
    and relocate tensors to the compute device.
    """
    prompts = getattr(req, "prompts", None) or []
    if not prompts:
        raise ValueError(f"SoulX-Singer {expected_kind} request has no prompts.")
    prompt = prompts[0]
    if isinstance(prompt, str):
        raise ValueError(f"SoulX-Singer {expected_kind} forward requires pre_process_func output on the prompt.")
    payload = get_soulx_preprocessed_payload(prompt)
    if payload is None or payload.get("kind") != expected_kind:
        raise ValueError(
            f"SoulX-Singer {expected_kind} forward requires additional_information['soulx_preprocessed'] "
            "produced by pre_process_func."
        )
    return relocate_tensors(payload, device)
