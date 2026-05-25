#!/usr/bin/env python3
"""SoulX-Singer OpenAI-compatible chat client (SVS / SVC).

Uses ``POST /v1/chat/completions`` with ``extra_body.num_inference_steps``,
``extra_body.guidance_scale``, and pipeline fields under ``extra_body.extra_args``.
Paths must exist on the **server** filesystem (use absolute paths when client and
server share a host).

Usage:
  # SVS (preprocessed metadata + prompt alignment wav)
  python openai_chat_client.py --mode svs \\
      --prompt-metadata-path /abs/zh_prompt.json \\
      --target-metadata-path /abs/music.json \\
      --audio-path /abs/zh_prompt.mp3 \\
      -o out.wav

  # SVC (wav + F0 npy)
  python openai_chat_client.py --mode svc \\
      --prompt-wav-path /abs/zh_prompt.mp3 \\
      --target-wav-path /abs/music.mp3 \\
      --prompt-f0-path /abs/zh_prompt_f0.npy \\
      --target-f0-path /abs/music_f0.npy \\
      -o out_svc.wav

  # Quick smoke test with repo fixtures (server must see the same paths)
  python openai_chat_client.py --mode svs --use-bundled-assets -o svs.wav
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path

import requests
import soundfile
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
BUNDLED_ASSETS = REPO_ROOT / "tests" / "assets" / "soulxsinger"


def _abs_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def _save_wav(audio: torch.Tensor, path: Path, sample_rate: int) -> None:
    audio = audio.to(torch.float32)
    peak = audio.abs().max()
    if peak > 0:
        audio = audio / peak.clamp(min=1e-8)
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), audio.clamp(-1.0, 1.0).cpu().T.numpy(), sample_rate, subtype="PCM_16")


def _decode_audio_from_response(body: dict) -> tuple[torch.Tensor, int]:
    for choice in body.get("choices", []):
        audio_obj = choice.get("message", {}).get("audio")
        if not (isinstance(audio_obj, dict) and audio_obj.get("data")):
            continue
        data, sr = soundfile.read(
            io.BytesIO(base64.b64decode(audio_obj["data"])),
            dtype="float32",
            always_2d=True,
        )
        return torch.from_numpy(data).transpose(0, 1), int(sr)
    brief = {k: v for k, v in body.items() if k != "choices"}
    raise RuntimeError(f"no audio in response message.audio: {brief}")


def _bundled_svs_extra_args() -> dict:
    return {
        "prompt_metadata_path": _abs_path(str(BUNDLED_ASSETS / "zh_prompt.json")),
        "target_metadata_path": _abs_path(str(BUNDLED_ASSETS / "music.json")),
        "audio_path": _abs_path(str(BUNDLED_ASSETS / "zh_prompt.mp3")),
        "control": "score",
        "auto_shift": True,
        "pitch_shift": 0,
    }


def _bundled_svc_extra_args() -> dict:
    return {
        "prompt_wav_path": _abs_path(str(BUNDLED_ASSETS / "zh_prompt.mp3")),
        "target_wav_path": _abs_path(str(BUNDLED_ASSETS / "music.mp3")),
        "prompt_f0_path": _abs_path(str(BUNDLED_ASSETS / "zh_prompt_f0.npy")),
        "target_f0_path": _abs_path(str(BUNDLED_ASSETS / "music_f0.npy")),
        "auto_shift": True,
        "pitch_shift": 0,
    }


def _build_extra_args(args: argparse.Namespace) -> dict:
    if args.use_bundled_assets:
        return _bundled_svs_extra_args() if args.mode == "svs" else _bundled_svc_extra_args()

    if args.mode == "svs":
        paths = {
            "prompt_metadata_path": args.prompt_metadata_path,
            "target_metadata_path": args.target_metadata_path,
            "audio_path": args.audio_path,
        }
        missing = [k for k, p in paths.items() if not p]
        if missing:
            raise SystemExit(f"SVS requires: {', '.join(missing)} (or --use-bundled-assets)")
        return {
            "prompt_metadata_path": _abs_path(paths["prompt_metadata_path"]),
            "target_metadata_path": _abs_path(paths["target_metadata_path"]),
            "audio_path": _abs_path(paths["audio_path"]),
            "control": args.control,
            "auto_shift": args.auto_shift,
            "pitch_shift": args.pitch_shift,
        }

    paths = {
        "prompt_wav_path": args.prompt_wav_path,
        "target_wav_path": args.target_wav_path,
        "prompt_f0_path": args.prompt_f0_path,
        "target_f0_path": args.target_f0_path,
    }
    missing = [k for k, p in paths.items() if not p]
    if missing:
        raise SystemExit(f"SVC requires: {', '.join(missing)} (or --use-bundled-assets)")
    return {
        "prompt_wav_path": _abs_path(paths["prompt_wav_path"]),
        "target_wav_path": _abs_path(paths["target_wav_path"]),
        "prompt_f0_path": _abs_path(paths["prompt_f0_path"]),
        "target_f0_path": _abs_path(paths["target_f0_path"]),
        "auto_shift": args.auto_shift,
        "pitch_shift": args.pitch_shift,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="SoulX-Singer chat completions client")
    p.add_argument("--mode", choices=["svs", "svc"], default="svs", help="SVS or SVC pipeline")
    p.add_argument(
        "--use-bundled-assets",
        action="store_true",
        help=f"Use fixtures under {BUNDLED_ASSETS} (server must read the same paths)",
    )
    p.add_argument("--prompt", "-p", default="Generate singing voice", help="Chat message text")
    p.add_argument("--output", "-o", default="soulx_out.wav")
    p.add_argument("--server", "-s", default="http://localhost:8091")
    p.add_argument("--model", default="Soul-AILab/SoulX-Singer", help="Model id passed to the API")
    p.add_argument("--steps", type=int, default=32, help="num_inference_steps")
    p.add_argument("--guidance-scale", type=float, default=3.0)
    p.add_argument("--timeout", type=float, default=1800.0, help="HTTP timeout seconds (long songs)")
    # SVS
    p.add_argument("--prompt-metadata-path", default=str(BUNDLED_ASSETS / "zh_prompt.json"))
    p.add_argument("--target-metadata-path", default=str(BUNDLED_ASSETS / "music.json"))
    p.add_argument("--audio-path", default=str(BUNDLED_ASSETS / "zh_prompt.mp3"))
    p.add_argument("--control", choices=["score", "melody"], default="score")
    # SVC
    p.add_argument("--prompt-wav-path")
    p.add_argument("--target-wav-path")
    p.add_argument("--prompt-f0-path")
    p.add_argument("--target-f0-path")
    # Shared
    p.add_argument("--auto-shift", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pitch-shift", type=int, default=0)
    args = p.parse_args()

    extra_args = _build_extra_args(args)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "extra_body": {
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "extra_args": extra_args,
        },
    }

    url = f"{args.server.rstrip('/')}/v1/chat/completions"
    print(f"POST {url}  mode={args.mode}  steps={args.steps}  cfg={args.guidance_scale}")
    r = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=args.timeout)
    if not r.ok:
        print(r.text, file=sys.stderr)
    r.raise_for_status()
    body = r.json()
    for choice in body.get("choices", []):
        metrics = choice.get("metrics")
        if metrics:
            print(f"metrics: {metrics}")
    audio, sr = _decode_audio_from_response(body)
    _save_wav(audio, Path(args.output), sr)
    dur = audio.shape[-1] / sr
    print(f"saved {args.output}  sr={sr}Hz  duration={dur:.2f}s  channels={audio.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
