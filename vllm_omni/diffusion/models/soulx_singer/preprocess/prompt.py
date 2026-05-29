"""Build stage-0 preprocess prompts for multistage SoulX-Singer serving."""


from collections.abc import Sequence
from typing import Any


def prepare_preprocess_prompt(
    prompt: Any,
    stage0_params: Any = None,
    downstream_params: Any = None,
) -> None:
    """Merge stage sampling ``extra_args`` into a stage-0 prompt dict in place."""
    items = prompt if isinstance(prompt, list) else [prompt]
    for item in items:
        if not isinstance(item, dict):
            continue

        merged_extra_args: dict[str, Any] = {}
        for source in (item, stage0_params, downstream_params):
            if isinstance(source, dict):
                extra_args = source.get("extra_args")
                if isinstance(extra_args, dict):
                    merged_extra_args.update(extra_args)
            elif source is not None:
                extra_args = getattr(source, "extra_args", None)
                if isinstance(extra_args, dict):
                    merged_extra_args.update(extra_args)

        if not merged_extra_args and not item.get("multi_modal_data"):
            continue

        info = item.setdefault("additional_information", {})
        if not isinstance(info, dict):
            continue

        existing_extra = dict(info.get("extra_args") or {})
        existing_extra.update(merged_extra_args)
        info["extra_args"] = existing_extra

        snapshot: dict[str, Any] = dict(info.get("prompt") or {})
        if "prompt" in item and "prompt" not in snapshot:
            snapshot["prompt"] = item["prompt"]
        multi_modal_data = item.get("multi_modal_data")
        if isinstance(multi_modal_data, dict) and multi_modal_data:
            snapshot["multi_modal_data"] = dict(multi_modal_data)
        if snapshot:
            info["prompt"] = snapshot

        if merged_extra_args:
            top_extra = dict(item.get("extra_args") or {})
            top_extra.update(merged_extra_args)
            item["extra_args"] = top_extra


def resolve_model_stage(stage_cfg: Any) -> str | None:
    """Best-effort ``model_stage`` lookup across dict/object stage configs."""
    if isinstance(stage_cfg, dict):
        if stage_cfg.get("model_stage"):
            return str(stage_cfg["model_stage"])
        engine_args = stage_cfg.get("engine_args") or {}
        if isinstance(engine_args, dict) and engine_args.get("model_stage"):
            return str(engine_args["model_stage"])
    model_stage = getattr(stage_cfg, "model_stage", None)
    if model_stage:
        return str(model_stage)
    engine_args = getattr(stage_cfg, "engine_args", None)
    if isinstance(engine_args, dict) and engine_args.get("model_stage"):
        return str(engine_args["model_stage"])
    if engine_args is not None:
        nested = getattr(engine_args, "model_stage", None)
        if nested:
            return str(nested)
    return None


def prepare_multistage_prompt(
    prompt: Any,
    sampling_params_list: Sequence[Any],
) -> Any:
    """Prepare stage-0 prompt from a full multistage ``sampling_params_list``."""
    stage0_params = sampling_params_list[0] if sampling_params_list else None
    downstream_params = sampling_params_list[1] if len(sampling_params_list) > 1 else None
    prepare_preprocess_prompt(prompt, stage0_params, downstream_params)
    return prompt


def prepare_multistage_prompt_if_needed(
    prompt: Any,
    sampling_params_list: Sequence[Any],
    stage_configs: Sequence[Any] | None = None,
) -> Any:
    """Merge downstream ``extra_args`` into stage-0 for SoulX preprocess pipelines."""
    if len(sampling_params_list) < 2:
        return prompt
    if stage_configs:
        if resolve_model_stage(stage_configs[0]) != "soulx_preprocess":
            return prompt
    prepare_multistage_prompt(prompt, sampling_params_list)
    return prompt
