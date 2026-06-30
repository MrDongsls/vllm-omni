# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
import torchvision

from vllm_omni.diffusion.models.internvla_a1 import (
    InternVLAA1Config,
    InternVLAA1TrainMetadata,
)
from vllm_omni.diffusion.models.internvla_a1.config import OBS_IMAGES, OBS_STATE, OBS_TASK

INTERNVLA_A1_EXTRA_BODY_PARAMS: frozenset[str] = frozenset(
    {
        "batch_inputs",
        "noise",
    }
)
INTERNVLA_A1_EXTRA_OUTPUT_PARAMS: frozenset[str] = frozenset()

CAMERA_KEYS = ("image0", "image1", "image2")


def _normalize_vector(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    denom = np.where(std == 0, np.ones_like(std), std)
    return (values - mean) / denom


def _unnormalize_vector(values: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return values * std + mean


def _load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _stack_stats(stats: dict[str, Any], keys: list[str]) -> dict[str, torch.Tensor]:
    result = {}
    for stat_name in ("mean", "std"):
        values = []
        for key in keys:
            values.extend(stats[key][stat_name])
        result[stat_name] = torch.tensor(values, dtype=torch.float32)
    return result


def _clamp_index(index: int, start: int, end: int) -> int:
    return max(start, min(end - 1, index))


class TorchvisionVideoReaderCache:
    def __init__(self, backend: str = "pyav") -> None:
        self.backend = backend
        self._readers: dict[str, Any] = {}
        torchvision.set_video_backend(backend)

    def get(self, path: str) -> Any:
        reader = self._readers.get(path)
        if reader is None:
            reader = torchvision.io.VideoReader(path, "video")
            self._readers[path] = reader
        return reader

    def decode_frames(self, path: str, timestamps: list[float], tolerance_s: float = 1e-4) -> torch.Tensor:
        reader = self.get(path)
        first_ts = min(timestamps)
        last_ts = max(timestamps)
        reader.seek(first_ts, keyframes_only=self.backend == "pyav")

        loaded_frames: list[torch.Tensor] = []
        loaded_ts: list[float] = []
        for frame in reader:
            current_ts = float(frame["pts"])
            loaded_frames.append(frame["data"])
            loaded_ts.append(current_ts)
            if current_ts >= last_ts:
                break

        query_ts = torch.tensor(timestamps, dtype=torch.float32)
        loaded_ts_tensor = torch.tensor(loaded_ts, dtype=torch.float32)
        distances = torch.cdist(query_ts[:, None], loaded_ts_tensor[:, None], p=1)
        min_dist, argmin = distances.min(dim=1)
        if not torch.all(min_dist < tolerance_s):
            raise RuntimeError(
                f"Video timestamps are outside tolerance: query={query_ts.tolist()} "
                f"loaded={loaded_ts_tensor.tolist()} path={path}"
            )
        return torch.stack([loaded_frames[i] for i in argmin]).float() / 255.0


@dataclass
class A2DOpenLoopSample:
    index: int
    episode_index: int
    task: str
    state_raw: torch.Tensor
    action_raw: torch.Tensor
    inputs: dict[str, torch.Tensor]


class A2DOpenLoopDataset:
    image_keys = [
        "observation.images.head",
        "observation.images.hand_left",
        "observation.images.hand_right",
    ]
    state_keys = [
        "observation.states.joint.position",
        "observation.states.effector.position",
    ]
    action_keys = [
        "actions.joint.position",
        "actions.effector.position",
    ]

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        config: InternVLAA1Config,
        train_stats: dict[str, Any],
        image_offsets: tuple[int, int] = (-15, 0),
        tolerance_s: float = 1e-4,
    ) -> None:
        self.root = Path(dataset_root)
        self.config = config
        self.info = _load_json(self.root / "meta" / "info.json")
        self.dataset_stats = _load_json(self.root / "meta" / "stats.json")
        self.data_rows = _load_parquet_rows(self.root / "data" / "chunk-000" / "file-000.parquet")
        self.episode_rows = _load_parquet_rows(self.root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
        self.task_rows = _load_parquet_rows(self.root / "meta" / "tasks.parquet")

        self.state_stats = _stack_stats(train_stats, self.state_keys)
        self.action_stats = _stack_stats(train_stats, self.action_keys)
        self.image_offsets = image_offsets
        self.tolerance_s = tolerance_s
        self.video_reader = TorchvisionVideoReaderCache(backend="pyav")

    @property
    def num_episodes(self) -> int:
        return len(self.episode_rows)

    @property
    def physical_action_dim(self) -> int:
        return 16

    def episode_start_indices(self, max_episodes: int | None = None) -> list[tuple[int, list[int]]]:
        rows = self.episode_rows if max_episodes is None else self.episode_rows[:max_episodes]
        result = []
        for ep in rows:
            start = int(ep["dataset_from_index"])
            end = int(ep["dataset_to_index"])
            result.append((int(ep["episode_index"]), list(range(start, end, self.config.chunk_size))))
        return result

    def _task_text(self, task_index: int) -> str:
        return self.task_rows[task_index]["__index_level_0__"]

    def _episode_for_index(self, row: dict[str, Any]) -> dict[str, Any]:
        return self.episode_rows[int(row["episode_index"])]

    def _state_vector(self, row: dict[str, Any]) -> torch.Tensor:
        return torch.tensor(row[self.state_keys[0]] + row[self.state_keys[1]], dtype=torch.float32)

    def _action_vector(self, row: dict[str, Any]) -> torch.Tensor:
        return torch.tensor(row[self.action_keys[0]] + row[self.action_keys[1]], dtype=torch.float32)

    def _query_rows(self, idx: int, deltas: list[int]) -> list[dict[str, Any]]:
        row = self.data_rows[idx]
        episode = self._episode_for_index(row)
        start = int(episode["dataset_from_index"])
        end = int(episode["dataset_to_index"])
        return [self.data_rows[_clamp_index(idx + delta, start, end)] for delta in deltas]

    def _decode_camera_history(
        self, episode: dict[str, Any], camera_key: str, rows: list[dict[str, Any]]
    ) -> torch.Tensor:
        timestamps = [float(r["timestamp"]) for r in rows]
        shifted = [float(episode[f"videos/{camera_key}/from_timestamp"]) + ts for ts in timestamps]
        chunk_idx = int(episode[f"videos/{camera_key}/chunk_index"])
        file_idx = int(episode[f"videos/{camera_key}/file_index"])
        path = self.root / self.info["video_path"].format(
            video_key=camera_key,
            chunk_index=chunk_idx,
            file_index=file_idx,
        )
        frames = self.video_reader.decode_frames(str(path), shifted, tolerance_s=self.tolerance_s)
        return frames

    def get_sample(self, idx: int) -> A2DOpenLoopSample:
        row = self.data_rows[idx]
        episode = self._episode_for_index(row)
        image_rows = self._query_rows(idx, list((-15, 0)))
        action_rows = self._query_rows(idx, list(range(self.config.chunk_size)))
        camera_images = [self._decode_camera_history(episode, camera_key, image_rows) for camera_key in self.image_keys]
        state_raw = self._state_vector(row)
        state_norm = _normalize_vector(state_raw, self.state_stats)
        action_raw = torch.stack([self._action_vector(action_row) for action_row in action_rows], dim=0)
        task = self._task_text(int(row["task_index"]))
        inputs = {
            OBS_STATE: state_norm,
            OBS_TASK: task,
            f"{OBS_IMAGES}.image0": camera_images[0],
            f"{OBS_IMAGES}.image1": camera_images[1],
            f"{OBS_IMAGES}.image2": camera_images[2],
            f"{OBS_IMAGES}.image0_mask": torch.tensor(True),
            f"{OBS_IMAGES}.image1_mask": torch.tensor(True),
            f"{OBS_IMAGES}.image2_mask": torch.tensor(True),
        }
        return A2DOpenLoopSample(
            index=idx,
            episode_index=int(row["episode_index"]),
            task=task,
            state_raw=state_raw,
            action_raw=action_raw,
            inputs=inputs,
        )


def collate_open_loop_samples(
    samples: list[A2DOpenLoopSample],
    *,
    device: str,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    first = samples[0]
    batch_inputs: dict[str, torch.Tensor] = {}
    for key in first.inputs:
        values = [sample.inputs[key] for sample in samples]
        if isinstance(values[0], torch.Tensor):
            tensor = torch.stack(values, dim=0)
            if tensor.dtype in (torch.int64, torch.bool):
                batch_inputs[key] = tensor.to(device=device)
            else:
                batch_inputs[key] = tensor.to(device=device, dtype=dtype)
        else:
            batch_inputs[key] = values

    metadata = {
        "indices": [sample.index for sample in samples],
        "episode_indices": [sample.episode_index for sample in samples],
        "tasks": [sample.task for sample in samples],
        "state_raw": torch.stack([sample.state_raw for sample in samples], dim=0),
        "action_raw": torch.stack([sample.action_raw for sample in samples], dim=0),
    }
    return batch_inputs, metadata


def build_observations(source: dict[str, Any]) -> dict[str, Any]:
    """Build one InternVLA-A1 extra_args dict from the A2D dataset.

    Returns a single dict (→ single-shot mode). ``_meta`` carries the
    denorm stats the action processor needs and is stripped before the
    request is built.
    """
    dataset_dir = source.get("dataset_dir")
    if dataset_dir is None or not Path(dataset_dir).exists():
        raise ValueError(f"InternVLA-A1 requires source['dataset_dir']. Got: {dataset_dir}.")

    model_dir = Path(source.get("model_dir") or source.get("model", ""))
    device = source.get("device", "cuda")
    dtype_name = source.get("dtype", "bfloat16")
    seed = int(source.get("seed", 42))
    sample_index = int(source.get("sample_index", 0))

    config = InternVLAA1Config.from_pretrained(model_dir)
    config.device, config.dtype = device, dtype_name
    train_meta = InternVLAA1TrainMetadata.from_pretrained(model_dir)
    with open(model_dir / "stats.json") as f:
        train_stats = json.load(f)["a2d"]

    dataset = A2DOpenLoopDataset(Path(dataset_dir), config=config, train_stats=train_stats)
    sample = dataset.get_sample(sample_index)

    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float32
    batch_inputs, meta = collate_open_loop_samples([sample], device=device, dtype=dtype)
    noise = torch.randn(
        (1, config.chunk_size, config.max_action_dim),
        generator=torch.Generator(device="cpu").manual_seed(seed),
        dtype=torch.float32,
    )

    return {
        "batch_inputs": batch_inputs,
        "noise": noise,
        "_meta": {
            "physical_action_dim": dataset.physical_action_dim,
            "action_stats": dataset.action_stats,
            "action_mode": train_meta.action_mode,
            "state_raw": meta["state_raw"],
        },
    }


def process_robot_actions(
    output: Any,
    *,
    physical_action_dim: int = 16,
    joint_dims: int = 14,
    unnormalize_mean: np.ndarray | None = None,
    unnormalize_std: np.ndarray | None = None,
    raw_state: np.ndarray | None = None,
    action_mode: str | None = None,
) -> dict[str, Any]:
    """Extract actions from InternVLA-A1 pipeline output.

    InternVLA-A1 returns ``DiffusionOutput(output=tensor)`` where the tensor
    has shape ``(1, chunk_size=50, max_action_dim=32)``. This processor slices
    to the physical action dim and optionally unnormalizes and applies delta
    correction.

    Args:
        output: The raw ``DiffusionOutput`` from ``pipeline.forward()``.
        physical_action_dim: Number of action dimensions to keep (default 16).
        joint_dims: Number of joint dimensions within the action (default 14).
        unnormalize_mean: Per-dimension mean for action unnormalization.
            Shape ``(physical_action_dim,)``.
        unnormalize_std: Per-dimension std for action unnormalization.
            Shape ``(physical_action_dim,)``.
        raw_state: Raw (unnormalized) state for delta-mode correction.
            Shape ``(max_state_dim,)``. Required when ``action_mode="delta"``.
        action_mode: ``"delta"`` to apply delta correction, ``None`` otherwise.

    Returns:
        A dict with ``{"actions": np.ndarray, "metadata": {...}}``.
    """
    output_tensor = getattr(output, "output", output)

    if isinstance(output_tensor, torch.Tensor):
        actions = output_tensor[:, :, :physical_action_dim].to(torch.float32)  # (1, 50, 16)
    else:
        raise TypeError(f"InternVLA-A1 output must be a torch.Tensor, got {type(output_tensor)}")

    if unnormalize_mean is not None and unnormalize_std is not None:
        mean_t = torch.tensor(unnormalize_mean, dtype=torch.float32)
        std_t = torch.tensor(unnormalize_std, dtype=torch.float32)
        actions = _unnormalize_vector(actions, mean_t, std_t)

    if action_mode == "delta" and raw_state is not None:
        raw = np.asarray(raw_state, dtype=np.float32).flatten()[:joint_dims]
        raw_t = torch.from_numpy(raw).unsqueeze(0).unsqueeze(0)  # (1, 1, joint_dims)
        actions[:, :, :joint_dims] += raw_t

    actions_np = actions.squeeze(0).cpu().numpy()  # (50, physical_action_dim)
    return {"actions": actions_np, "metadata": {}}
