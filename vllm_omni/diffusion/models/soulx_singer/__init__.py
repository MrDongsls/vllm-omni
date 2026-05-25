"""SoulX-Singer SVS / SVC flow-matching singing models for vLLM-Omni."""

from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_base import (
    FlowMatchingAudioPipeline,
    get_soulxsinger_post_process_func,
)
from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_svc import (
    PipelineSoulXSingerSVC,
    get_soulxsinger_svc_pre_process_func,
)
from vllm_omni.diffusion.models.soulx_singer.pipeline_soulx_singer_svs import (
    PipelineSoulXSingerSVS,
    get_soulxsinger_pre_process_func,
)

__all__ = [
    "FlowMatchingAudioPipeline",
    "PipelineSoulXSingerSVS",
    "PipelineSoulXSingerSVC",
    "get_soulxsinger_post_process_func",
    "get_soulxsinger_pre_process_func",
    "get_soulxsinger_svc_pre_process_func",
]
