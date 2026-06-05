"""Resolve SoulX preprocess checkpoint locations."""

import os
from pathlib import Path

from huggingface_hub import snapshot_download
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig

logger = init_logger(__name__)

_PREPROCESS_WEIGHTS_REPO = "Soul-AILab/SoulX-Singer-Preprocess"
_RMVPE_REL = Path("rmvpe/rmvpe.pt")


def resolve_preprocess_weights_root(
    od_config: OmniDiffusionConfig,
    extra_args: dict | None = None,
) -> Path:
    """Locate preprocess weights on disk or download from Hugging Face."""
    extra_args = extra_args or {}
    candidates: list[Path] = []

    if env_override := os.environ.get("SOULX_PREPROCESS_WEIGHTS_DIR"):
        candidates.append(Path(env_override).expanduser())

    # assume model weight dir and preprocess weight dir share the same parent directory
    model_dir = Path(od_config.model).expanduser()
    candidates.extend(
        [
            model_dir / "preprocess",
            model_dir.parent / "SoulX-Singer-Preprocess",
        ]
    )

    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / _RMVPE_REL).is_file():
            logger.info("Using SoulX preprocess weights from %s", resolved)
            return resolved

    logger.info(
        "SoulX preprocess weights not found locally; downloading %s",
        _PREPROCESS_WEIGHTS_REPO,
    )
    downloaded = snapshot_download(_PREPROCESS_WEIGHTS_REPO, allow_patterns=["*"])
    return Path(downloaded).resolve()


def preprocess_weight_paths(weights_root: Path) -> dict[str, str]:
    """Map upstream relative paths to absolute paths under ``weights_root``."""
    root = weights_root.resolve()
    return {
        "rmvpe": str(root / "rmvpe/rmvpe.pt"),
        "sep_ckpt": str(root / "mel-band-roformer-karaoke/mel_band_roformer_karaoke_becruily.ckpt"),
        "sep_config": str(root / "mel-band-roformer-karaoke/config_karaoke_becruily.yaml"),
        "asr_zh": str(root / "speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
        "asr_en": str(root / "parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo"),
        "rosvot": str(root / "rosvot/rosvot/model.pt"),
    }
