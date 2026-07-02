# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.outputs import OmniRequestOutput

DREAMZERO_EXTRA_BODY_PARAMS: frozenset[str] = frozenset(
    {
        "robot_obs",
        "reset",
        "session_id",
    }
)
DREAMZERO_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()

ACTION_HORIZON = 24
RELATIVE_OFFSETS = (-23, -16, -8, 0)
CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}
DEFAULT_NUM_CHUNKS = 2


def _load_video_frames(video_path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError("DreamZero observation loading requires opencv-python.") from exc

    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)


def _build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    chunks: list[list[int]] = []
    current = 23
    for _ in range(num_chunks):
        indices = [max(current + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            break
        chunks.append(indices)
        current += ACTION_HORIZON
    return chunks


def _make_obs(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    *,
    prompt: str,
) -> dict[str, Any]:
    obs: dict[str, Any] = {}
    for key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]
        obs[key] = selected[0] if len(frame_indices) == 1 else selected
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    return obs


def build_observations(source: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read 3 camera MP4s and yield DreamZero AR extra_args dicts.

    Yields a sequence (→ autoregressive mode). The first item carries
    ``reset=True``; all share one ``session_id``.
    """
    data_dir = source.get("data_dir")
    if data_dir is None:
        raise ValueError("DreamZero requires source['data_dir'] with 3 camera MP4 files.")
    data_dir = Path(data_dir)
    task = source.get("task", "")
    num_chunks = int(source.get("num_chunks", DEFAULT_NUM_CHUNKS))
    session_id = source.get("session_id")

    camera_frames: dict[str, np.ndarray] = {}
    for camera_key, file_name in CAMERA_FILES.items():
        path = data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Camera video not found: {path}")
        camera_frames[camera_key] = _load_video_frames(path)

    total = min(f.shape[0] for f in camera_frames.values())
    schedule = [[0]] + _build_frame_schedule(total, num_chunks)

    extra_args = []
    for index, frame_indices in enumerate(schedule):
        obs = _make_obs(camera_frames, frame_indices, prompt=task)
        obs["session_id"] = session_id
        extra_args.append(
            {
                "reset": index == 0,
                "robot_obs": obs,
                "session_id": session_id,
                "prompt": task,
            }
        )
    return extra_args, {}


def process_robot_actions(
    output: OmniRequestOutput,
    **kwargs,
) -> dict[str, Any]:
    """Extract actions from DreamZero's DiffusionOutput.

    DreamZero returns ``DiffusionOutput(output={"actions": np.ndarray, "video": ...})``.
    This processor extracts the actions array and passes through any extra metadata.

    Args:
        output: The raw DiffusionOutput from pipeline.forward().

    Returns:
        A dict with at least ``{"actions": np.ndarray, "metadata": {...}}``.
    """
    action_output = output.multimodal_output.get("actions")
    action_array = np.asarray(action_output)

    if not output.images:
        raise RuntimeError("DreamZero output does not contain video latents in `images`.")
    latents = output.images[0]
    if not isinstance(latents, torch.Tensor):
        raise TypeError(f"Expected tensor latents, got {type(latents)!r}")

    latents = latents.detach().cpu()
    if latents.dim() == 4:
        latents = latents.unsqueeze(0)
    if latents.dim() != 5:
        raise ValueError(f"Unexpected latent shape: {tuple(latents.shape)}")

    if latents.shape[1] < latents.shape[2]:
        latents = latents.transpose(1, 2).contiguous()
    return {"actions": action_array, "metadata": {"video_latents": latents}}
