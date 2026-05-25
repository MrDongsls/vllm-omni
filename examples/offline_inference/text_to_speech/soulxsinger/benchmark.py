"""Repeatable SoulX-Singer offline benchmark (RTF + optional stage timings).

Usage:
    python benchmark.py --model /path/to/SoulX-Singer-svs --svs \\
        --prompt-metadata-path ... --target-metadata-path ... --audio-path ... \\
        --output benchmark_svs.wav

    python benchmark.py --model /path/to/SoulX-Singer-svc \\
        --prompt-wav-path ... --target-wav-path ... --prompt-f0-path ... --target-f0-path ... \\
        --output benchmark_svc.wav

    python benchmark.py ... --enable-diffusion-pipeline-profiler
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from vllm_omni.engine.arg_utils import nullify_stage_engine_defaults
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ASSETS = REPO_ROOT / "tests" / "assets" / "soulxsinger"
_SAMPLE_RATE = 24000


def _require_paths(paths: dict[str, str | None]) -> None:
    missing = [name for name, p in paths.items() if not p or not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Missing file(s): "
            + ", ".join(f"{k}={paths[k]!r}" for k in missing)
            + ". Run upstream SoulX-Singer preprocess first."
        )


def _extract_audio(outputs) -> tuple[np.ndarray, int]:
    ro = outputs[0].request_output
    mm = getattr(ro, "multimodal_output", None) if ro is not None else None
    if not mm and ro is not None and ro.outputs:
        mm = getattr(ro.outputs[0], "multimodal_output", None)
    if not mm or "audio" not in mm:
        raise RuntimeError("No audio in multimodal_output")
    audio = mm["audio"]
    sr = int(mm.get("audio_sample_rate") or mm.get("sr") or _SAMPLE_RATE)
    if hasattr(audio, "cpu"):
        audio_np = audio.detach().cpu().numpy().squeeze()
    else:
        audio_np = np.asarray(audio).squeeze()
    return audio_np.astype(np.float32, copy=False), sr


def _audio_duration_sec(outputs) -> float:
    audio_np, sr = _extract_audio(outputs)
    return audio_np.size / sr


def _stage_durations_from_output(outputs) -> dict[str, float]:
    omni_out = outputs[0]
    durations = getattr(omni_out, "stage_durations", None) or {}
    if durations:
        return dict(durations)
    ro = getattr(omni_out, "request_output", None)
    if ro is not None:
        return dict(getattr(ro, "stage_durations", None) or {})
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoulX-Singer offline benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--svs", action="store_true")
    parser.add_argument("--control", type=str, default="score", choices=["score", "melody"])
    parser.add_argument("--prompt-metadata-path", type=str, default=str(DEFAULT_ASSETS / "zh_prompt.json"))
    parser.add_argument("--target-metadata-path", type=str, default=str(DEFAULT_ASSETS / "music.json"))
    parser.add_argument("--audio-path", type=str, default=str(DEFAULT_ASSETS / "zh_prompt.mp3"))
    parser.add_argument("--prompt-wav-path", type=str, default=str(DEFAULT_ASSETS / "zh_prompt.mp3"))
    parser.add_argument("--target-wav-path", type=str, default=str(DEFAULT_ASSETS / "music.mp3"))
    parser.add_argument("--prompt-f0-path", type=str, default=str(DEFAULT_ASSETS / "zh_prompt_f0.npy"))
    parser.add_argument("--target-f0-path", type=str, default=str(DEFAULT_ASSETS / "music_f0.npy"))
    parser.add_argument("--auto-shift", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pitch-shift", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (excluded from stats)")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs")
    parser.add_argument(
        "--enable-diffusion-pipeline-profiler",
        action="store_true",
        help="Enable low-overhead stage timing in DiffusionOutput.stage_durations",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable torch.compile on diff_estimator (eager baseline for A/B)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["bfloat16", "bf16", "float16", "fp16", "half"],
        help="DiT trunk dtype (default: framework bfloat16). Use float16 to match upstream --fp16.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Save WAV from the last measured run (after warmup) for listening / AB comparison",
    )
    parser.add_argument(
        "--save-each-run",
        action="store_true",
        help="With --output, also write measured runs as <stem>_run1.wav, _run2.wav, ...",
    )
    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


def _build_run_config(args: argparse.Namespace) -> tuple[str, dict, OmniDiffusionSamplingParams]:
    if args.svs:
        model_class = "SoulXSingerPipeline"
        _require_paths(
            {
                "prompt_metadata_path": args.prompt_metadata_path,
                "target_metadata_path": args.target_metadata_path,
                "audio_path": args.audio_path,
            }
        )
        extra_args = {
            "prompt_metadata_path": os.path.abspath(args.prompt_metadata_path),
            "target_metadata_path": os.path.abspath(args.target_metadata_path),
            "audio_path": os.path.abspath(args.audio_path),
            "control": args.control,
            "auto_shift": args.auto_shift,
            "pitch_shift": args.pitch_shift,
        }
    else:
        model_class = "SoulXSingerSVCPipeline"
        _require_paths(
            {
                "prompt_wav_path": args.prompt_wav_path,
                "target_wav_path": args.target_wav_path,
                "prompt_f0_path": args.prompt_f0_path,
                "target_f0_path": args.target_f0_path,
            }
        )
        extra_args = {
            "prompt_wav_path": os.path.abspath(args.prompt_wav_path),
            "target_wav_path": os.path.abspath(args.target_wav_path),
            "prompt_f0_path": os.path.abspath(args.prompt_f0_path),
            "target_f0_path": os.path.abspath(args.target_f0_path),
            "auto_shift": args.auto_shift,
            "pitch_shift": args.pitch_shift,
        }

    sampling = OmniDiffusionSamplingParams(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        extra_args=extra_args,
    )
    return model_class, extra_args, sampling


def main() -> None:
    args = parse_args()
    model_class, _, sampling = _build_run_config(args)

    omni_kwargs: dict = {"model": args.model, "model_class_name": model_class}
    if args.enable_diffusion_pipeline_profiler:
        omni_kwargs["enable_diffusion_pipeline_profiler"] = True
    if args.enforce_eager:
        omni_kwargs["enforce_eager"] = True
    if args.dtype is not None:
        omni_kwargs["dtype"] = args.dtype

    compile_mode = "eager" if args.enforce_eager else "torch.compile (regional on LlamaNARDecoderLayer)"
    dtype_label = args.dtype or "from deploy/framework"
    print(f"Loading SoulX-Singer ({model_class}) from {args.model} [{compile_mode}, dtype={dtype_label}]")
    omni = Omni(**omni_kwargs)
    prompts = {"prompt": "soulx-singer-benchmark"}

    latencies_ms: list[float] = []
    rtfs: list[float] = []
    last_stages: dict[str, float] = {}
    last_measured_outputs = None

    total_runs = args.warmup + args.runs
    for run_idx in range(total_runs):
        is_warmup = run_idx < args.warmup
        label = "warmup" if is_warmup else f"run {run_idx - args.warmup + 1}/{args.runs}"
        t0 = time.perf_counter()
        outputs = list(omni.generate(prompts, sampling_params_list=[sampling]))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        audio_sec = _audio_duration_sec(outputs)
        rtf = (elapsed_ms / 1000.0) / audio_sec if audio_sec > 0 else float("inf")
        print(f"[{label}] client={elapsed_ms:.1f} ms, audio={audio_sec:.2f}s, RTF={rtf:.3f}")
        if not is_warmup:
            latencies_ms.append(elapsed_ms)
            rtfs.append(rtf)
            last_stages = _stage_durations_from_output(outputs)
            last_measured_outputs = outputs
            if args.output and args.save_each_run:
                measured_idx = run_idx - args.warmup + 1
                out_path = Path(args.output)
                run_path = out_path.with_name(f"{out_path.stem}_run{measured_idx}{out_path.suffix}")
                audio_np, sr = _extract_audio(outputs)
                sf.write(str(run_path), audio_np, sr)
                print(f"  saved {run_path} ({sr} Hz, {audio_np.size / sr:.2f}s)")

    omni.close()

    if args.output and last_measured_outputs is not None:
        audio_np, sr = _extract_audio(last_measured_outputs)
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio_np, sr)
        print(f"\nSaved last measured run → {out_path} ({sr} Hz, {audio_np.size / sr:.2f}s)")

    if latencies_ms:
        print("\n=== Summary (measured runs) ===")
        print(f"client_ms: mean={statistics.mean(latencies_ms):.1f}, stdev={statistics.pstdev(latencies_ms):.1f}")
        print(f"RTF:       mean={statistics.mean(rtfs):.3f}, stdev={statistics.pstdev(rtfs):.3f}")
        if last_stages:
            print("\nStage durations (last measured run):")
            for name, value in sorted(last_stages.items()):
                if name.endswith("_ms"):
                    print(f"  {name}: {value:.1f} ms")
                else:
                    print(f"  {name}: {value * 1000.0:.1f} ms")


if __name__ == "__main__":
    main()
