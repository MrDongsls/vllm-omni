"""ROSVOT conv/norm building blocks (inference-only; weights from checkpoint)."""

# ruff: noqa: N803

import torch
import torch.nn as nn


class LayerNorm(torch.nn.LayerNorm):
    """LayerNorm with optional channel dimension (for [B, C, T] tensors)."""

    def __init__(self, nout: int, dim: int = -1, eps: float = 1e-5):
        super().__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == -1:
            return super().forward(x)
        return super().forward(x.transpose(1, -1)).transpose(1, -1)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def get_norm_builder(norm_type: str, channels: int, ln_eps: float = 1e-6):
    if norm_type == "ln":
        return lambda: LayerNorm(channels, dim=1, eps=ln_eps)
    return lambda: nn.Identity()


def get_act_builder(act_type: str):
    if act_type == "leakyrelu":
        return lambda: nn.LeakyReLU(negative_slope=0.01, inplace=True)
    return lambda: nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        dilation,
        n=2,
        norm_type="ln",
        c_multiple=2,
        ln_eps=1e-12,
        act_type="leakyrelu",
    ):
        super().__init__()
        norm_builder = get_norm_builder(norm_type, channels, ln_eps)
        act_builder = get_act_builder(act_type)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    norm_builder(),
                    nn.Conv1d(
                        channels,
                        c_multiple * channels,
                        kernel_size,
                        dilation=dilation,
                        padding=(dilation * (kernel_size - 1)) // 2,
                    ),
                    LambdaLayer(lambda x, ks=kernel_size: x * ks**-0.5),
                    act_builder(),
                    nn.Conv1d(c_multiple * channels, channels, 1, dilation=dilation),
                )
                for _ in range(n)
            ]
        )

    def forward(self, x):
        nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        for block in self.blocks:
            x = (x + block(x)) * nonpadding
        return x


class ConvBlocks(nn.Module):
    def __init__(
        self,
        hidden_size,
        out_dims,
        dilations=None,
        kernel_size=3,
        norm_type="ln",
        layers_in_block=2,
        c_multiple=2,
        ln_eps=1e-5,
        is_BTC=True,
        num_layers=None,
        post_net_kernel=3,
        act_type="leakyrelu",
    ):
        super().__init__()
        self.is_BTC = is_BTC
        if num_layers is not None:
            dilations = [1] * num_layers
        elif dilations is None:
            dilations = [1]
        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(
                    hidden_size,
                    kernel_size,
                    d,
                    n=layers_in_block,
                    norm_type=norm_type,
                    c_multiple=c_multiple,
                    ln_eps=ln_eps,
                    act_type=act_type,
                )
                for d in dilations
            ],
        )
        self.last_norm = get_norm_builder(norm_type, hidden_size, ln_eps)()
        self.post_net1 = nn.Conv1d(
            hidden_size,
            out_dims,
            kernel_size=post_net_kernel,
            padding=post_net_kernel // 2,
        )

    def forward(self, x, nonpadding=None):
        if self.is_BTC:
            x = x.transpose(1, 2)
        if nonpadding is None:
            nonpadding = (x.abs().sum(1) > 0).float()[:, None, :]
        elif self.is_BTC:
            nonpadding = nonpadding.transpose(1, 2)
        x = self.res_blocks(x) * nonpadding
        x = self.last_norm(x) * nonpadding
        x = self.post_net1(x) * nonpadding
        if self.is_BTC:
            x = x.transpose(1, 2)
        return x
