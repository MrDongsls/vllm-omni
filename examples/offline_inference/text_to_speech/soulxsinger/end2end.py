"""Offline SoulX-Singer SVS / SVC inference via vLLM-Omni.

Requires **preprocessed** metadata (upstream ``SoulX-Singer/preprocess/``). Omni does not
run preprocess in this script.

Usage:
    # SVS (singing voice synthesis) — set config.json architectures to SoulXSingerPipeline
    python end2end.py --model /path/to/SoulX-Singer --svs \\
        --prompt-metadata-path /path/to/zh_prompt.json \\
        --target-metadata-path /path/to/music.json \\
        --audio-path /path/to/zh_prompt.mp3

    # SVC (singing voice conversion) — architectures SoulXSingerSVCPipeline
    python end2end.py --model /path/to/SoulX-Singer \\
        --prompt-wav-path /path/to/zh_prompt.mp3 \\
        --target-wav-path /path/to/music.mp3 \\
        --prompt-f0-path /path/to/zh_prompt_f0.npy \\
        --target-f0-path /path/to/music_f0.npy
"""

from __future__ import annotations

import argparse
import os
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SoulX-Singer offline SVS / SVC")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="local directory of the SoulX-Singer model weight",
    )
    parser.add_argument(
        "--svs",
        action="store_true",
        help="SVS mode (metadata paths). Default pipeline when omitted is SVC.",
    )
    parser.add_argument(
        "--control",
        type=str,
        default="score",
        choices=["score", "melody"],
        help="SVS control mode (score = MIDI pitch, melody = F0)",
    )
    parser.add_argument(
        "--prompt-metadata-path",
        type=str,
        default=str(DEFAULT_ASSETS / "zh_prompt.json"),
        help="SVS: prompt metadata JSON from preprocess",
    )
    parser.add_argument(
        "--target-metadata-path",
        type=str,
        default=str(DEFAULT_ASSETS / "music.json"),
        help="SVS: target metadata JSON (may contain multiple segments for long songs)",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=str(DEFAULT_ASSETS / "zh_prompt.mp3"),
        help="SVS: prompt alignment wav",
    )
    parser.add_argument(
        "--prompt-wav-path",
        type=str,
        default=str(DEFAULT_ASSETS / "zh_prompt.mp3"),
        help="SVC: reference timbre wav",
    )
    parser.add_argument(
        "--target-wav-path",
        type=str,
        default=str(DEFAULT_ASSETS / "music.mp3"),
        help="SVC: target content wav",
    )
    parser.add_argument(
        "--prompt-f0-path",
        type=str,
        default=str(DEFAULT_ASSETS / "zh_prompt_f0.npy"),
        help="SVC: prompt F0 .npy",
    )
    parser.add_argument(
        "--target-f0-path",
        type=str,
        default=str(DEFAULT_ASSETS / "music_f0.npy"),
        help="SVC: target F0 .npy",
    )
    parser.add_argument(
        "--auto-shift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable automatic pitch shift between prompt and target",
    )
    parser.add_argument(
        "--pitch-shift",
        type=int,
        default=0,
        help="Manual pitch shift in semitones (overrides auto when non-zero)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=32,
        help="Diffusion sampling steps",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.wav",
        help="Output wav path",
    )
    nullify_stage_engine_defaults(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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

    print(f"Loading SoulX-Singer ({model_class}) from {args.model}")
    omni = Omni(model=args.model, model_class_name=model_class)

    prompts = {"prompt": "soulx-singer"}
    sampling = OmniDiffusionSamplingParams(
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        extra_args=extra_args,
    )

    print("Running generate (single request; long SVS merges segments inside pipeline)...")
    outputs = list(omni.generate(prompts, sampling_params_list=[sampling]))
    if not outputs:
        raise RuntimeError("No output from omni.generate")

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

    sf.write(args.output, audio_np, sr)
    print(f"Saved {args.output} ({sr} Hz, {len(audio_np) / sr:.2f}s)")
    omni.close()


if __name__ == "__main__":
    main()
