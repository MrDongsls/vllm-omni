"""SoulX-Singer multi-stage pipeline topologies (preprocess → DiT)."""

from dataclasses import replace

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.soulx_singer"

_SOULX_STAGE0 = StagePipelineConfig(
    stage_id=0,
    model_stage="soulx_preprocess",
    execution_type=StageExecutionType.LLM_GENERATION,
    input_sources=(),
    final_output=False,
    engine_output_type="latent",
    owns_tokenizer=True,
    requires_multimodal_data=True,
    model_arch="SoulXSingerModel",
)

SOULX_SINGER_SVS_PIPELINE = PipelineConfig(
    model_type="soulxsinger_svs",
    model_arch="SoulXSingerModel",
    hf_architectures=("SoulXSingerPipeline", "SoulXSingerModel"),
    stages=(
        _SOULX_STAGE0,
        StagePipelineConfig(
            stage_id=1,
            model_stage="soulx_svs",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            requires_multimodal_data=True,
            sync_process_input_func=f"{_PROC}.preprocess2svs",
            extras={"model_class_name": "SoulXSingerPipeline"},
        ),
    ),
)

SOULX_SINGER_SVC_PIPELINE = PipelineConfig(
    model_type="soulxsinger_svc",
    model_arch="SoulXSingerModel",
    hf_architectures=("SoulXSingerSVCPipeline", "SoulXSingerModel"),
    stages=(
        replace(_SOULX_STAGE0, extras={"soulx_mode": "svc"}),
        StagePipelineConfig(
            stage_id=1,
            model_stage="soulx_svc",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            requires_multimodal_data=True,
            sync_process_input_func=f"{_PROC}.preprocess2svc",
            extras={"model_class_name": "SoulXSingerSVCPipeline"},
        ),
    ),
)
