"""E2E offline multistage tests for SoulX-Singer (preprocess → SVS/SVC)."""

import functools
import importlib
import json
import os
from pathlib import Path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path
from tests.helpers.runtime import OmniRunner
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.diffusion.models.soulx_singer.preprocess.prompt import prepare_multistage_prompt
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

PROMPT_AUDIO = get_asset_path("soulxsinger/zh_prompt.mp3")
TARGET_AUDIO = get_asset_path("soulxsinger/music.mp3")
PHONE_SET = get_asset_path("soulxsinger/phoneme/phone_set.json")
SAMPLE_RATE = 24_000

if not PROMPT_AUDIO.is_file() or not TARGET_AUDIO.is_file():
    pytest.skip(
        f"Missing SoulX-Singer audio assets: {PROMPT_AUDIO.name}, {TARGET_AUDIO.name}",
        allow_module_level=True,
    )

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, pytest.mark.tts]

_CASES = (
    pytest.param(
        "svs",
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
        "svc",
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


def _model_view(base: Path, name: str, architecture: str) -> str:
    view = base / ".pytest_soulx_views" / name
    view.mkdir(parents=True, exist_ok=True)
    for fname in ("config.yaml", "model.pt", "model-svc.pt"):
        src, dst = base / fname, view / fname
        if src.is_file() and not dst.exists():
            dst.symlink_to(src.resolve())
    phoneset = base / "phoneme" / "phone_set.json"
    if not phoneset.is_file():
        phoneset = PHONE_SET
    if phoneset.is_file():
        (view / "phoneme").mkdir(parents=True, exist_ok=True)
        dst = view / "phoneme" / "phone_set.json"
        if not dst.exists():
            dst.symlink_to(phoneset.resolve())
    (view / "config.json").write_text(json.dumps({"model_type": "soulxsinger", "architectures": [architecture]}) + "\n")
    return str(view.resolve())


def _flatten_audio(audio_val) -> np.ndarray:
    import torch

    if isinstance(audio_val, list):
        chunks = [c.detach().cpu().float().numpy().reshape(-1) for c in audio_val if c is not None]
        return np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
    if isinstance(audio_val, torch.Tensor):
        return audio_val.detach().cpu().float().numpy().reshape(-1)
    return np.asarray(audio_val, dtype=np.float32).reshape(-1)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("view,architecture,deploy_yaml,extra_args,py_deps", _CASES)
def test_soulxsinger_multistage_from_audio(
    soulx_weights: tuple[Path, Path],
    view: str,
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
    model = _model_view(base_dir, view, architecture)
    with OmniRunner(
        model,
        stage_configs_path=get_deploy_config_path(deploy_yaml),
        async_chunk=False,
    ) as runner:
        sampling = runner.get_default_sampling_params_list()
        sampling[0] = SamplingParams(max_tokens=1)
        sampling[1] = OmniDiffusionSamplingParams(
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
        prompt = prepare_multistage_prompt({"prompt_token_ids": [0]}, sampling)
        outputs = runner.generate([prompt], sampling)

    assert outputs and outputs[0].error is None, outputs[0].error if outputs else "no output"
    mm = outputs[0].multimodal_output
    assert isinstance(mm, dict) and "audio" in mm
    audio = _flatten_audio(mm["audio"])
    assert 12_000 <= audio.size
    assert np.isfinite(audio).all() and float(np.max(np.abs(audio))) > 1e-4
    duration_s = audio.size / SAMPLE_RATE
    assert 15.0 <= duration_s <= 70.0, f"duration={duration_s:.1f}s"
