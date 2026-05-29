"""SoulX-Singer preprocess stage wrapped for ``LLM_GENERATION`` runtime."""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.soulx_singer.modules.preprocess.module import SoulXPreprocessModule
from vllm_omni.diffusion.models.soulx_singer.preprocess.payload import (
    build_dummy_payload,
    encode_ipc,
)
from vllm_omni.diffusion.models.soulx_singer.preprocess.pre_process import (
    build_metadata_processor,
    build_preprocess_payload,
    prepare_runtime_inputs,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput


class SoulXSingerPreprocessForGeneration(nn.Module):
    """One-shot preprocess stage: audio in → ``soulx_preprocessed`` payload out."""

    have_multimodal_outputs = True
    enable_update_additional_information = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        del prefix
        self.config = vllm_config.model_config.hf_config
        self.model_stage = "soulx_preprocess"
        self.soulx_mode = str(getattr(self.config, "soulx_mode", "svs")).lower()
        self.hidden_size = int(getattr(self.config, "hidden_size", 128))

        self._od_config = OmniDiffusionConfig(model=vllm_config.model_config.model)
        self._preprocess: nn.Module | None = None
        self._metadata_processor = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_preprocess(self, extra_args: dict[str, Any]):
        if self._preprocess is None:
            self._preprocess = SoulXPreprocessModule(
                self._od_config,
                vocal_sep=bool(extra_args.get("vocal_sep", True)),
                midi_transcribe=self.soulx_mode == "svs",
                max_merge_duration_ms=int(extra_args.get("max_merge_duration", 60000)),
                verbose=bool(extra_args.get("preprocess_verbose", False)),
                extra_args=extra_args,
            )
        return self._preprocess

    def _get_metadata_processor(self):
        if self._metadata_processor is None:
            self._metadata_processor = build_metadata_processor(self._od_config)
        return self._metadata_processor

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        del weights
        return set()

    def get_dummy_runtime_additional_information(self, num_reqs: int) -> list[dict[str, Any]]:
        return [{"_is_dummy": True}] * num_reqs

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        del kwargs
        return torch.zeros(
            (input_ids.shape[0], self.hidden_size),
            device=input_ids.device,
            dtype=torch.float32,
        )

    def _omni_output(self, input_ids: torch.Tensor, payload: dict[str, Any]) -> OmniOutput:
        hidden = torch.zeros(
            (max(1, input_ids.shape[0]), self.hidden_size),
            device=input_ids.device,
            dtype=torch.float32,
        )
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=encode_ipc(payload),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> OmniOutput:
        del positions, intermediate_tensors, inputs_embeds
        runtime_info = kwargs.get("model_intermediate_buffer")
        if runtime_info is None:
            runtime_info = kwargs.get("runtime_additional_information", [])

        is_dummy = (isinstance(runtime_info, dict) and runtime_info.get("_is_dummy")) or (
            isinstance(runtime_info, list)
            and bool(runtime_info)
            and isinstance(runtime_info[0], dict)
            and runtime_info[0].get("_is_dummy")
        )
        if is_dummy:
            return self._omni_output(input_ids, build_dummy_payload(self.soulx_mode, input_ids.device))

        prompt, extra_args = prepare_runtime_inputs(runtime_info)
        payload = build_preprocess_payload(
            self.soulx_mode,
            prompt=prompt,
            extra_args=extra_args,
            preprocess=self._get_preprocess(extra_args),
            metadata_processor=self._get_metadata_processor(),
            device=self._device,
            sample_rate=int(getattr(self.config, "sample_rate", 24000)),
        )
        return self._omni_output(input_ids, payload)

    def compute_logits(
        self,
        hidden_states: torch.Tensor | OmniOutput,
        sampling_metadata: Any = None,
    ) -> torch.Tensor:
        """One-shot preprocess: finish after the first forward step."""
        del sampling_metadata
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            hidden_states = torch.zeros((1, 1), device=device, dtype=torch.float32)
        vocab_size = int(getattr(self.config, "vocab_size", 2))
        logits = torch.full(
            (hidden_states.shape[0], vocab_size),
            fill_value=-1e9,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        logits[:, 0] = 0.0
        return logits


class SoulXSingerModel(SoulXSingerPreprocessForGeneration):
    """Registry entrypoint for stage-0 SoulX preprocess."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if vllm_config.model_config.model_stage != "soulx_preprocess":
            raise ValueError(
                "SoulXSingerModel only supports model_stage='soulx_preprocess', "
                f"got {vllm_config.model_config.model_stage!r}."
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
