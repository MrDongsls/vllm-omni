from torch import nn

from .espnet_positional_embedding import RelPositionalEncoding
from .espnet_transformer_attn import RelPositionMultiHeadedAttention
from .layers import ConvolutionModule, EncoderLayer, MultiLayeredConv1d, Swish


class ConformerLayers(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_layers,
        kernel_size=9,
        dropout=0.0,
        num_heads=4,
        use_last_norm=True,
        save_hidden=False,
    ):
        super().__init__()
        self.use_last_norm = use_last_norm
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (hidden_size, hidden_size * 4, 1, dropout)
        self.pos_embed = RelPositionalEncoding(hidden_size, dropout)
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    hidden_size,
                    RelPositionMultiHeadedAttention(num_heads, hidden_size, 0.0),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    ConvolutionModule(hidden_size, kernel_size, Swish()),
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )
        if self.use_last_norm:
            self.layer_norm = nn.LayerNorm(hidden_size)
        else:
            self.layer_norm = nn.Linear(hidden_size, hidden_size)
        self.save_hidden = save_hidden
        if save_hidden:
            self.hiddens = []

    def forward(self, x, padding_mask=None):
        self.hiddens = []
        nonpadding_mask = x.abs().sum(-1) > 0
        x = self.pos_embed(x)
        for layer in self.encoder_layers:
            x, mask = layer(x, nonpadding_mask[:, None, :])
            if self.save_hidden:
                self.hiddens.append(x[0])
        x = x[0]
        x = self.layer_norm(x) * nonpadding_mask.float()[:, :, None]
        return x
