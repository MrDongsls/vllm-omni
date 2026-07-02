# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Robot policy inference — shared task for robot-policy DiT models.

The script is model-agnostic: it selects an *inference mode* and an
*observation loader* declared in ``vllm_omni/model_extras/<model>.py``. To add a
new robot-policy model, register ``observation_loader`` + ``robot_obs_builder``
+ ``action_output_processor`` — no edits here. Inference mode can be automatically
detected by checking the Iterable[dict] input type.

Examples:
    # DreamZero (autoregressive)
    python robot_policy.py --model GEAR-Dreams/DreamZero-DROID \\
        --deploy-config vllm_omni/deploy/dreamzero_tp1_cfg2.yaml \\
        --video-dir outputs/dreamzero/assets --task "Move the pan forward"

    # InternVLA-A1 (single_shot)
    python robot_policy.py --model /path/to/internvla_a1 \\
        --dataset-dir /path/to/a2d --task "pick up the cube"
"""

from __future__ import annotations

import argparse
import functools
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig
from vllm_omni.diffusion.utils.param_utils import apply_declared_extra_args
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.model_extras import (
    build_robot_observations,
    get_extra_body_params,
    get_model_class_name,
    process_robot_actions,
)

DECODE_VIDEO_SUPPORTS = [
    "dreamzero",
]


def parse_json_object(value: str, flag_name: str = "argument") -> dict[str, Any]:
    """Parse a CLI value as a JSON object, attributing errors to ``flag_name``."""
    try:
        config = json.loads(value)
    except json.JSONDecodeError as e:
        raise argparse.ArgumentTypeError(f"{flag_name} must be valid JSON: {e}") from e
    if not isinstance(config, dict):
        raise argparse.ArgumentTypeError(f"{flag_name} must be a JSON object")
    return config


parse_profiler_config = functools.partial(parse_json_object, flag_name="--profiler-config")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robot policy inference, action sequence planning from an image or numpy array."
    )
    parser.add_argument(
        "--model",
        default="GEAR-Dreams/DreamZero-DROID",
        help="Diffusers Robot policy model ID or local path (Dreamzero, Internvla-a1, ...)",
    )
    parser.add_argument("--model-class-name", default=None, help="Override model class name.")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--deploy-config", default=None)
    parser.add_argument("--data-dir", type=Path, help="Directory containing organized assets needed by examples.")
    parser.add_argument("--task", default="", help="Task prompt string controls the robot trajectory planning.")
    parser.add_argument("--num-chunks", type=int, default=2)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=Path("robot_policy_output.npz"))
    parser.add_argument(
        "--vae-use-slicing",
        action="store_true",
        help="Enable VAE slicing for memory optimization.",
    )
    parser.add_argument(
        "--vae-use-tiling",
        action="store_true",
        help="Enable VAE tiling for memory optimization.",
    )
    parser.add_argument(
        "--enable-cpu-offload",
        action="store_true",
        help="Enable CPU offloading for diffusion models.",
    )
    parser.add_argument(
        "--enable-layerwise-offload",
        action="store_true",
        help="Enable layerwise (blockwise) offloading on DiT modules.",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile and force eager execution.",
    )
    parser.add_argument(
        "--audio-sample-rate",
        type=int,
        default=24000,
        help="Sample rate for audio output when saved (default: 24000).",
    )
    parser.add_argument(
        "--cache-backend",
        type=str,
        default=None,
        choices=["cache_dit", "tea_cache"],
        help=(
            "Cache backend to use for acceleration. "
            "Options: 'cache_dit' (DBCache + SCM + TaylorSeer), 'tea_cache' (Timestep Embedding Aware Cache). "
            "Default: None (no cache acceleration)."
        ),
    )
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable diffusion pipeline profiler to display stage durations.",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default=None,
        choices=["fp8", "mxfp8", "mxfp4", "mxfp4_dualscale", "int8", "gguf"],
        help="Quantization method for the transformer. mxfp8: W8A8 MXFP8 (NPU). mxfp4: W4A4 MXFP4 (NPU). mxfp4_dualscale: W4A4 MXFP4 dual-scale + BF16 fallback mixed (NPU). fp8: online FP8 (GPU).",
    )

    # Distributed and parallel execution
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ulysses sequence parallelism.",
    )
    parser.add_argument(
        "--ring-degree",
        type=int,
        default=1,
        help="Number of GPUs used for ring sequence parallelism.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for tensor parallelism (TP) inside the DiT.",
    )
    parser.add_argument(
        "--cfg-parallel-size",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs used for classifier free guidance parallel size.",
    )
    parser.add_argument(
        "--vae-patch-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used for VAE patch/tile parallelism (decode).",
    )
    parser.add_argument(
        "--use-hsdp",
        action="store_true",
        help=("Enable Hybrid Sharded Data Parallel to shard model weights across GPUs. "),
    )
    parser.add_argument(
        "--hsdp-shard-size",
        type=int,
        default=-1,
        help=(
            "Number of GPUs to shard model weights across within each replica group. "
            "-1 (default) auto-calculates as world_size / replicate_size. "
        ),
    )
    parser.add_argument(
        "--hsdp-replicate-size",
        type=int,
        default=1,
        help=(
            "Number of replica groups for HSDP. Each replica holds a full sharded copy. "
            "Default 1 means pure sharding (no replication). "
        ),
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="Number of pipeline parallel stages.",
    )
    parser.add_argument(
        "--profiler-config",
        type=parse_profiler_config,
        default=None,
        help='JSON profiler config for torch/cuda profiling, e.g. \'{"profiler":"torch","torch_profiler_dir":"./perf"}\'.',
    )
    parser.add_argument(
        "--extra-body",
        type=functools.partial(parse_json_object, flag_name="--extra-body"),
        default=None,
        help=(
            "Model-specific generation params as a JSON object. Keys are filtered "
            "against the model's declared extra_body_params (see vllm_omni/model_extras), "
            "so unknown keys for the chosen model are silently dropped. "
            'Cosmos3 V2V example: \'{"condition_frame_indexes_vision": [0, 1], '
            '"condition_video_keep": "first", "flow_shift": 10.0, '
            '"max_sequence_length": 4096, "guardrails": false}\'.'
        ),
    )
    return parser.parse_args()


def _write_mp4(model_class_name: str, frames: np.ndarray, fps: int) -> None:
    import cv2

    omni_root = Path(__file__).expanduser().parents[3]
    video_write_path = omni_root / f"example_video_{model_class_name}.mp4"
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(
        video_write_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_write_path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def normalize_extra_body_params(
    sampling_params,
    extra_body: dict,
    declared_extra_body: dict,
):
    if declared_extra_body:
        apply_declared_extra_args(sampling_params, declared_extra_body, extra_body)
    elif extra_body:
        sampling_params.extra_args.update({k: v for k, v in extra_body.items() if v is not None})
    return sampling_params


def run_inference(omni: Omni, model_class_name, observations, extra_body, declared_extra_body) -> list[dict[str, Any]]:
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    metadata = extra_body.get("metadata", {})

    results = []
    for index, extra_args in enumerate(observations):
        prompt = extra_args.get("prompt", "")
        sp = normalize_extra_body_params(
            OmniDiffusionSamplingParams(extra_args=extra_args), extra_body, declared_extra_body
        )
        raw = omni.generate(prompt, sampling_params_list=[sp])
        if not raw:
            raise RuntimeError(f"No output for step {index}")
        results.append(process_robot_actions(model_class_name, raw[0], **metadata))
    return results


def main() -> None:
    args = parse_args()
    model_class_name = args.model_class_name

    print(f"[robot_policy] model={args.model} class={model_class_name}")

    # Configure cache based on backend type
    cache_config = None
    if args.cache_backend == "cache_dit":
        cache_config = {
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "max_warmup_steps": 4,
            "residual_diff_threshold": 0.24,
            "max_continuous_cached_steps": 3,
            "enable_taylorseer": False,
            "taylorseer_order": 1,
            "scm_steps_mask_policy": None,
            "scm_steps_policy": "dynamic",
        }
    elif args.cache_backend == "tea_cache":
        cache_config = {
            "rel_l1_thresh": 0.2,
        }

    profiler_enabled = args.profiler_config is not None
    parallel_config = DiffusionParallelConfig(
        ulysses_degree=args.ulysses_degree,
        ring_degree=args.ring_degree,
        cfg_parallel_size=args.cfg_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        vae_patch_parallel_size=args.vae_patch_parallel_size,
        use_hsdp=args.use_hsdp,
        hsdp_shard_size=args.hsdp_shard_size,
        hsdp_replicate_size=args.hsdp_replicate_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )
    omni_kwargs = dict(
        model=args.model,
        enable_layerwise_offload=args.enable_layerwise_offload,
        vae_use_slicing=args.vae_use_slicing,
        vae_use_tiling=args.vae_use_tiling,
        enable_cpu_offload=args.enable_cpu_offload,
        parallel_config=parallel_config,
        enforce_eager=args.enforce_eager,
        model_class_name=model_class_name,
        cache_backend=args.cache_backend,
        cache_config=cache_config,
        enable_diffusion_pipeline_profiler=args.enable_diffusion_pipeline_profiler,
        profiler_config=args.profiler_config,
    )
    if args.quantization is not None:
        omni_kwargs["quantization"] = args.quantization
    declared_extra_body_params = get_extra_body_params(model_class_name)

    omni = Omni(**omni_kwargs)
    model_class_name = get_model_class_name(omni) or model_class_name
    print(f"[Robot Policy] Using model_class_name - {model_class_name}")

    if profiler_enabled:
        print("[Profiler] Starting profiling...")
        omni.start_profile()

    # NOTE: Reconsider the key parameters (interface def)
    source = {
        "model": args.model,
        "model_dir": args.model_dir,
        "task": args.task,
        "data_dir": args.data_dir,
        "num_chunks": args.num_chunks,
        "session_id": args.session_id or str(uuid.uuid4()),
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
    }

    # print task specific configuration
    # TODO: Undone
    print(f"\n{'=' * 60}")
    print("Robot policy configuration")
    print("")
    print(f"{'=' * 60}\n")

    extra_body = dict(args.extra_body or {})
    observation, metadata = build_robot_observations(model_class_name, source, **extra_body)
    if extra_body.get("metadata"):
        print("[Warning] CLI arguments have already provided metadata, please check if there is any confliction.")
    extra_body["metadata"] = metadata

    # Return type drives the mode: a single dict → single-shot,
    # any other iterable → autoregressive.
    observations = [observation] if isinstance(observation, dict) else observation
    results = run_inference(omni, model_class_name, observations, extra_body, declared_extra_body_params)

    if not results:
        print("[robot_policy] No actions produced.")
        return

    actions = [r["actions"] for r in results]
    stacked = np.stack(actions, axis=0) if len(actions) > 1 else actions[0]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, actions=stacked, num_steps=len(results))
    print(f"[robot_policy] saved {stacked.shape} → {out}")

    if args.export_video:
        if model_class_name not in DECODE_VIDEO_SUPPORTS:
            print(f"[Warning] {model_class_name} doesn't support video export.")
        else:
            video_latents = [r["metadata"].get("video_latents") for r in results]
            print(f"Decoding {len(video_latents)} steps...")
            if video_latents[0] is None:
                print("[Warning] Video latents not found in output. Please check the model output.")

            # NOTE: only example done this way (dreamzero), need more examples to formalize
            full_latents = torch.cat(video_latents, dim=2)
            stage_client = omni.engine.stage_clients[0]
            engine = getattr(stage_client, "_engine", None)
            if engine is None:
                raise RuntimeError("DreamZero export requires inline diffusion stage access.")

            decoded = engine.executor.collective_rpc(
                "decode_video_latents_to_uint8",
                args=(full_latents,),
                unique_reply_rank=0,
                exec_all_ranks=True,
            )

        if isinstance(decoded, torch.Tensor):
            frames = decoded.numpy()
        if not isinstance(frames, np.ndarray):
            raise TypeError(f"Unexpected decoded output type: {type(decoded)!r}")
        _write_mp4(model_class_name, frames, fps=5)


if __name__ == "__main__":
    main()
