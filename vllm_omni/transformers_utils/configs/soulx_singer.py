"""Minimal HuggingFace config for SoulX-Singer multi-stage preprocess."""

from transformers.configuration_utils import PretrainedConfig


class SoulXSingerConfig(PretrainedConfig):
    """Config for ``SoulXSingerModel`` stage-0 preprocess (generation runtime)."""

    model_type = "soulxsinger"

    def __init__(
        self,
        *,
        soulx_mode: str = "svs",
        hidden_size: int = 128,
        sample_rate: int = 24000,
        hop_size: int = 480,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.soulx_mode = soulx_mode
        self.hidden_size = hidden_size
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.num_attention_heads = 1
        self.num_hidden_layers = 1
        self.vocab_size = 2
