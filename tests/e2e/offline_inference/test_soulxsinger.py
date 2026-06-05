"""E2E offline inference tests for SoulX-Singer (single-stage, preprocess inline)."""

import functools
import importlib
import os
from pathlib import Path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

PROMPT_AUDIO = get_asset_path("soulxsinger/zh_prompt.mp3")
TARGET_AUDIO = get_asset_path("soulxsinger/music.mp3")
SAMPLE_RATE = 24_000

if not PROMPT_AUDIO.is_file() or not TARGET_AUDIO.is_file():
    pytest.skip(
        f"Missing SoulX-Singer audio assets: {PROMPT_AUDIO.name}, {TARGET_AUDIO.name}",
        allow_module_level=True,
    )

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, pytest.mark.tts]

_CASES = (
    pytest.param(
        "SoulXSingerPipeline",
        "soulxsinger_svs.yaml",
        {
            "language": "Mandarin",
            "vocal_sep": False,
            "control": "score",
            "auto_shift": False,
            "pitch_shift": 0,
        },
        ("g2pM", "g2p_en"),
        id="svs",
    ),
    pytest.param(
        "SoulXSingerSVCPipeline",
        "soulxsinger_svc.yaml",
        {"vocal_sep": False, "auto_shift": False, "pitch_shift": 0},
        (),
        id="svc",
    ),
)


@functools.lru_cache(maxsize=1)
def _resolve_weights() -> tuple[Path, Path]:
    for key in ("SOULXSINGER_MODEL_DIR", "SOULXSINGER_BASE_MODEL_DIR"):
        if raw := os.environ.get(key):
            base = Path(raw).expanduser().resolve()
            if (base / "config.yaml").is_file():
                break
    else:
        from huggingface_hub import snapshot_download

        base = Path(snapshot_download("Soul-AILab/SoulX-Singer", allow_patterns=["*"]))

    if raw := os.environ.get("SOULX_PREPROCESS_WEIGHTS_DIR"):
        pre = Path(raw).expanduser().resolve()
        if (pre / "rmvpe" / "rmvpe.pt").is_file():
            return base, pre

    from huggingface_hub import snapshot_download

    pre = Path(snapshot_download("Soul-AILab/SoulX-Singer-Preprocess", allow_patterns=["*"]))
    return base, pre


@pytest.fixture(scope="session")
def soulx_weights() -> tuple[Path, Path]:
    try:
        return _resolve_weights()
    except Exception as exc:
        pytest.skip(f"Set SOULXSINGER_MODEL_DIR / SOULX_PREPROCESS_WEIGHTS_DIR. ({exc})")


def _flatten_audio(audio_val) -> np.ndarray:
    import torch

    if isinstance(audio_val, list):
        chunks = [c.detach().cpu().float().numpy().reshape(-1) for c in audio_val if c is not None]
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    if isinstance(audio_val, torch.Tensor):
        return audio_val.detach().cpu().float().numpy().reshape(-1)
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("architecture,deploy_yaml,extra_args,py_deps", _CASES)
def test_soulxsinger_multistage_from_audio(
    soulx_weights: tuple[Path, Path],
    architecture: str,
    deploy_yaml: str,
    extra_args: dict,
    py_deps: tuple[str, ...],
) -> None:
    for mod in py_deps:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            pytest.fail(f"SoulX SVS requires {mod}: {exc}")

    base_dir, preprocess_dir = soulx_weights

    # SVS mode requires phone_set.json in the model directory
    if architecture == "SoulXSingerPipeline":
        if not (base_dir / "phoneme" / "phone_set.json").is_file() and not (base_dir / "phone_set.json").is_file():
            pytest.skip(
                "SoulX-Singer SVS test requires phoneme/phone_set.json. "
                "Copy it from github.com/Soul-AILab/SoulX-Singer into the model dir. "
                "See `examples/offline_inference/text_to_speech/README.md` for details."
            )

    model = str(base_dir)
    with OmniRunner(
        model,
        stage_configs_path=get_deploy_config_path(deploy_yaml),
        async_chunk=False,
    ) as runner:
        sampling = OmniDiffusionSamplingParams(
            num_inference_steps=4,
            guidance_scale=3.0,
            seed=42,
            extra_args={
                "prompt_audio": str(PROMPT_AUDIO),
                "target_audio": str(TARGET_AUDIO),
                "preprocess_weights_dir": str(preprocess_dir),
                **extra_args,
            },
        )
        prompt = {"prompt_token_ids": [0]}
        outputs = runner.generate([prompt], sampling)

    assert outputs and outputs[0].error is None, outputs[0].error if outputs else "no output"
    mm = outputs[0].multimodal_output
    assert isinstance(mm, dict) and "audio" in mm
    audio = _flatten_audio(mm["audio"])
    assert 12_000 <= audio.size
    assert np.isfinite(audio).all() and float(np.max(np.abs(audio))) > 1e-4
    duration_s = audio.size / SAMPLE_RATE
    assert 50.0 <= duration_s <= 52.0, f"duration={duration_s:.1f}s"
