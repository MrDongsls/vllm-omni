"""Shared ``pre_process_func`` helpers for SoulX-Singer pipelines."""


from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.soulx_singer.modules.preprocess.module import SoulXPreprocessModule
from vllm_omni.diffusion.models.soulx_singer.preprocess.payload import (
    SOULX_PRECOMPUTED_KEYS,
    SOULX_SVC_KIND,
    SOULX_SVS_KIND,
    attach_preprocessed,
    build_dummy_payload,
    get_soulx_preprocessed_payload,
    has_precomputed,
)
from vllm_omni.diffusion.models.soulx_singer.utils import MetadataProcessor, load_config
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniTextPrompt


def is_warmup_request(request: OmniDiffusionRequest) -> bool:
    request_ids = getattr(request, "request_ids", None) or ()
    return len(request_ids) == 1 and request_ids[0] == "dummy_req_id"


def normalize_prompt(prompt: Any) -> dict[str, Any]:
    if isinstance(prompt, str):
        return OmniTextPrompt(prompt=prompt)
    return prompt


def prepare_runtime_inputs(runtime_info: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize stage-0 runtime buffer into prompt + extra_args."""
    if runtime_info is None:
        info: dict[str, Any] = {}
    elif isinstance(runtime_info, dict):
        info = dict(runtime_info)
    elif isinstance(runtime_info, list):
        info = dict(runtime_info[0]) if runtime_info else {}
    else:
        info = {}

    extra_args = dict(info.get("extra_args") or {})
    prompt = dict(info.get("prompt") or {})
    if not prompt.get("multi_modal_data"):
        mm = info.get("multi_modal_data") or {}
        prompt_audio = mm.get("audio") if isinstance(mm, dict) else None
        if prompt_audio is None:
            prompt_audio = extra_args.get("prompt_audio")
        if prompt_audio is not None:
            prompt.setdefault("multi_modal_data", {})["audio"] = prompt_audio
    return prompt, extra_args


def build_preprocess_payload(
    kind: str,
    *,
    prompt: dict[str, Any],
    extra_args: dict[str, Any],
    preprocess: nn.Module,
    metadata_processor,
    device,
    sample_rate: int,
) -> dict[str, Any]:
    mm = prompt.get("multi_modal_data") or {}
    prompt_audio = mm.get("audio") if isinstance(mm, dict) else None
    if prompt_audio is None:
        prompt_audio = extra_args.get("prompt_audio")
    target_audio = extra_args.get("target_audio")

    if kind == SOULX_SVC_KIND:
        if has_precomputed(extra_args, SOULX_SVC_KIND):
            return SoulXPreprocessModule.build_svc_payload_from_paths(
                prompt_wav_path=str(extra_args["prompt_wav_path"]),
                target_wav_path=str(extra_args["target_wav_path"]),
                prompt_f0_path=str(extra_args["prompt_f0_path"]),
                target_f0_path=str(extra_args["target_f0_path"]),
                sample_rate=sample_rate,
                device=device,
            )
        if prompt_audio is None or target_audio is None:
            raise ValueError(
                "SoulX-Singer SVC preprocess requires precomputed paths or "
                "multi_modal_data['audio'] + extra_args['target_audio']."
            )
        return preprocess.build_svc_payload_from_audio(
            prompt_audio=prompt_audio,
            target_audio=target_audio,
            sample_rate=sample_rate,
            device=device,
            vocal_sep=extra_args.get("vocal_sep"),
        )

    if has_precomputed(extra_args, SOULX_SVS_KIND):
        return SoulXPreprocessModule.build_svs_payload_from_paths(
            prompt_metadata_path=str(extra_args["prompt_metadata_path"]),
            target_metadata_path=str(extra_args["target_metadata_path"]),
            audio_path=str(extra_args["audio_path"]),
            metadata_processor=metadata_processor,
        )
    if prompt_audio is None or target_audio is None:
        raise ValueError(
            "SoulX-Singer SVS preprocess requires precomputed metadata paths or "
            "multi_modal_data['audio'] + extra_args['target_audio']."
        )
    return preprocess.build_svs_payload_from_audio(
        prompt_audio=prompt_audio,
        target_audio=target_audio,
        metadata_processor=metadata_processor,
        language=str(extra_args.get("language", "Mandarin")),
        vocal_sep=extra_args.get("vocal_sep"),
        prompt_vocal_sep=extra_args.get("prompt_vocal_sep"),
        target_vocal_sep=extra_args.get("target_vocal_sep"),
        prompt_max_merge_duration_ms=extra_args.get("prompt_max_merge_duration"),
        target_max_merge_duration_ms=extra_args.get("target_max_merge_duration"),
    )


def attach_preprocess_for_diffusion_request(
    request: OmniDiffusionRequest,
    *,
    kind: str,
    metadata_processor=None,
    sample_rate: int | None = None,
    device=None,
) -> OmniDiffusionRequest:
    """Resolve preprocess payload for stage-1 diffusion (warmup / IPC / precomputed paths)."""
    extra_args = dict(getattr(request.sampling_params, "extra_args", None) or {})
    for i, prompt in enumerate(request.prompts):
        prompt = normalize_prompt(prompt)

        if is_warmup_request(request):
            request.sampling_params.num_inference_steps = 1
            if kind == SOULX_SVS_KIND:
                if metadata_processor is None:
                    raise ValueError("SVS warmup requires metadata_processor")
                payload = build_dummy_payload(SOULX_SVS_KIND, torch.device("cpu"))
                dummy_prompt = payload["prompt_meta"]
                processed = metadata_processor.process(dict(payload["target_meta_list"][0]))
                payload["prompt_meta"] = {
                    key: value.clone() if isinstance(value, torch.Tensor) else value
                    for key, value in processed.items()
                }
                if isinstance(dummy_prompt.get("wav"), torch.Tensor):
                    payload["prompt_meta"]["wav"] = dummy_prompt["wav"].clone()
            else:
                if device is None or sample_rate is None:
                    raise ValueError("SVC warmup requires device and sample_rate")
                payload = build_dummy_payload(SOULX_SVC_KIND, device)
        elif get_soulx_preprocessed_payload(prompt):
            request.prompts[i] = prompt
            continue
        elif has_precomputed(extra_args, kind):
            if kind == SOULX_SVS_KIND:
                payload = SoulXPreprocessModule.build_svs_payload_from_paths(
                    prompt_metadata_path=str(extra_args["prompt_metadata_path"]),
                    target_metadata_path=str(extra_args["target_metadata_path"]),
                    audio_path=str(extra_args["audio_path"]),
                    metadata_processor=metadata_processor,
                )
            else:
                payload = SoulXPreprocessModule.build_svc_payload_from_paths(
                    prompt_wav_path=str(extra_args["prompt_wav_path"]),
                    target_wav_path=str(extra_args["target_wav_path"]),
                    prompt_f0_path=str(extra_args["prompt_f0_path"]),
                    target_f0_path=str(extra_args["target_f0_path"]),
                    sample_rate=int(sample_rate),
                    device=device,
                )
        else:
            raise ValueError(
                f"SoulX-Singer {kind} requires precomputed paths "
                f"{list(SOULX_PRECOMPUTED_KEYS[kind])}, an attached soulx_preprocessed payload "
                f"(multi-stage preprocess stage), or run with pipeline soulxsinger_{kind}."
            )

        if payload.get("kind") != kind:
            raise ValueError(f"Invalid {kind} preprocess payload kind: {payload.get('kind')}")
        attach_preprocessed(prompt, payload)
        request.prompts[i] = prompt

    if kind == SOULX_SVS_KIND:
        extra_args.setdefault("control", extra_args.get("control", "score"))
    request.sampling_params.extra_args = extra_args
    return request


def build_metadata_processor(od_config: OmniDiffusionConfig):
    from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_base import _resolve_phoneset_path

    model_dir = od_config.model
    config_path = Path(model_dir) / "config.yaml"
    hf_config = load_config(str(config_path))
    audio_config = hf_config.audio

    return MetadataProcessor(
        hop_size=audio_config.hop_size,
        sample_rate=audio_config.sample_rate,
        phoneset_path=_resolve_phoneset_path(model_dir),
        device=str(get_local_device()),
    )
