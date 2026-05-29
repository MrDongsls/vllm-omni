"""Stage-0 preprocess → stage-1 SoulX DiT diffusion handoff."""

from typing import Any

from vllm.logger import init_logger

from vllm_omni.diffusion.models.soulx_singer.preprocess.payload import (
    SOULX_PREPROCESSED_KEY,
    SOULX_SVC_KIND,
    SOULX_SVS_KIND,
    attach_to_diffusion_prompt,
    extract_from_stage_output,
)

logger = init_logger(__name__)


def _preprocess2diffusion(
    source_outputs: list[Any],
    prompt: Any,
    *,
    expected_kind: str,
) -> list[dict[str, Any]]:
    diffusion_inputs: list[dict[str, Any]] = []
    for source_output in source_outputs:
        payload = extract_from_stage_output(source_output)
        if payload is None:
            logger.warning(
                "Missing %s in stage-0 output for request %s",
                SOULX_PREPROCESSED_KEY,
                getattr(source_output, "request_id", "?"),
            )
            continue
        diffusion_inputs.append(
            attach_to_diffusion_prompt(
                payload=payload,
                original_prompt=prompt,
                expected_kind=expected_kind,
            )
        )
    return diffusion_inputs


def preprocess2svs(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Attach stage-0 payload for ``PipelineSoulXSingerSVS``."""
    del _requires_multimodal_data
    return _preprocess2diffusion(source_outputs, prompt, expected_kind=SOULX_SVS_KIND)


def preprocess2svc(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[dict[str, Any]]:
    """Attach stage-0 payload for ``PipelineSoulXSingerSVC``."""
    del _requires_multimodal_data
    return _preprocess2diffusion(source_outputs, prompt, expected_kind=SOULX_SVC_KIND)
