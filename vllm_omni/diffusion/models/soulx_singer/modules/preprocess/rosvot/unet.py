import numpy as np
import torch
import torch.nn as nn

from .conv import ConvBlocks, ResidualBlock, get_act_builder, get_norm_builder


class UnetDown(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_layers,
        kernel_size,
        down_rates,
        channel_multiples=None,
        is_BTC=True,
        constant_channels=False,
    ):
        super().__init__()
        assert n_layers == len(down_rates)
        down_rates = [int(i) for i in down_rates]
        self.n_layers = n_layers
        self.is_BTC = is_BTC
        channel_multiples = channel_multiples if channel_multiples is not None else down_rates
        self.layers = nn.ModuleList()
        self.downs = nn.ModuleList()
        in_channels = hidden_size
        for i in range(self.n_layers):
            out_channels = int(in_channels * channel_multiples[i]) if not constant_channels else in_channels
            self.layers.append(
                nn.Sequential(
                    ResidualBlock(
                        in_channels,
                        kernel_size,
                        dilation=1,
                        n=1,
                        norm_type="ln",
                        c_multiple=1,
                        ln_eps=1e-5,
                        act_type="leakyrelu",
                    ),
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
                    ResidualBlock(
                        out_channels,
                        kernel_size,
                        dilation=1,
                        n=1,
                        norm_type="ln",
                        c_multiple=1,
                        ln_eps=1e-5,
                        act_type="leakyrelu",
                    ),
                )
            )
            self.downs.append(nn.AvgPool1d(down_rates[i]))
            in_channels = out_channels
        self.last_norm = get_norm_builder("ln", out_channels)()
        self.post_net = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x, **kwargs):
        if self.is_BTC:
            x = x.transpose(1, 2)
        skip_xs = []
        for i in range(self.n_layers):
            skip_x = self.layers[i](x)
            x = self.downs[i](skip_x)
            skip_xs.append(skip_x.transpose(1, 2) if self.is_BTC else skip_x)
        x = self.post_net(self.last_norm(x))
        if self.is_BTC:
            x = x.transpose(1, 2)
        return x, skip_xs


class UnetMid(nn.Module):
    def __init__(
        self,
        hidden_size,
        kernel_size,
        n_layers=None,
        in_dims=None,
        out_dims=None,
        is_BTC=True,
        net=None,
    ):
        super().__init__()
        in_dims = in_dims if in_dims is not None else hidden_size
        out_dims = out_dims if out_dims is not None else hidden_size
        self.pre = nn.Conv1d(in_dims, hidden_size, kernel_size, padding=kernel_size // 2)
        self.post = nn.Conv1d(hidden_size, out_dims, kernel_size, padding=kernel_size // 2)
        self.is_BTC = is_BTC
        self.net = net or ConvBlocks(
            hidden_size,
            out_dims=hidden_size,
            dilations=None,
            kernel_size=kernel_size,
            layers_in_block=2,
            c_multiple=2,
            num_layers=n_layers,
            post_net_kernel=3,
            act_type="leakyrelu",
            is_BTC=is_BTC,
        )

    def forward(self, x, cond=None, **kwargs):
        if self.is_BTC:
            x = self.pre(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.pre(x)
        x = self.net(x + (0 if cond is None else cond))
        if self.is_BTC:
            return self.post(x.transpose(1, 2)).transpose(1, 2)
        return self.post(x)


class UnetUp(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_layers,
        kernel_size,
        up_rates,
        channel_multiples=None,
        is_BTC=True,
        constant_channels=False,
        use_skip_layer=False,
        skip_scale=1.0,
    ):
        super().__init__()
        assert n_layers == len(up_rates)
        up_rates = [int(i) for i in up_rates]
        self.n_layers = n_layers
        self.is_BTC = is_BTC
        self.skip_scale = skip_scale
        channel_multiples = channel_multiples if channel_multiples is not None else up_rates
        self.in_channels_lst = (
            (np.cumprod([1] + channel_multiples) * hidden_size).astype(int)
            if not constant_channels
            else [hidden_size for _ in range(self.n_layers + 1)]
        )
        in_channels = self.in_channels_lst[-1]
        self.ups = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        self.layers = nn.ModuleList()
        for i in range(self.n_layers - 1, -1, -1):
            out_channels = self.in_channels_lst[i] if not constant_channels else in_channels
            self.ups.append(
                nn.Sequential(
                    nn.ConvTranspose1d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        stride=up_rates[i],
                        padding=kernel_size // 2,
                        output_padding=up_rates[i] - 1,
                    ),
                    get_norm_builder("ln", in_channels)(),
                    get_act_builder("leakyrelu")(),
                )
            )
            self.layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels * 2, out_channels, kernel_size, padding=(kernel_size - 1) // 2),
                    ResidualBlock(
                        out_channels,
                        kernel_size,
                        dilation=1,
                        n=1,
                        norm_type="ln",
                        c_multiple=1,
                        ln_eps=1e-5,
                        act_type="leakyrelu",
                    ),
                )
            )
            if use_skip_layer:
                self.skip_layers.append(
                    ResidualBlock(
                        in_channels,
                        kernel_size,
                        dilation=1,
                        n=1,
                        norm_type="ln",
                        c_multiple=1,
                        ln_eps=1e-5,
                        act_type="leakyrelu",
                    )
                )
            else:
                self.skip_layers.append(nn.Identity())
            in_channels = out_channels
        self.out_channels = out_channels
        self.last_norm = get_norm_builder("ln", out_channels)()
        self.post_net = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x, skips, **kwargs):
        if self.is_BTC:
            x = x.transpose(1, 2)
        for i in range(self.n_layers):
            x = self.ups[i](x)
            skip_x = skips[self.n_layers - i - 1]
            if self.is_BTC:
                skip_x = skip_x.transpose(1, 2)
            skip_x = self.skip_layers[i](skip_x) * self.skip_scale
            x = torch.cat((x, skip_x), dim=1)
            x = self.layers[i](x)
        x = self.post_net(self.last_norm(x))
        if self.is_BTC:
            x = x.transpose(1, 2)
        return x


class Unet(nn.Module):
    def __init__(
        self,
        hidden_size,
        down_layers,
        up_layers,
        kernel_size,
        updown_rates,
        mid_layers=None,
        channel_multiples=None,
        is_BTC=True,
        constant_channels=False,
        mid_net=None,
        use_skip_layer=False,
        skip_scale=1.0,
    ):
        super().__init__()
        assert len(updown_rates) == down_layers == up_layers
        if channel_multiples is None:
            channel_multiples = updown_rates
        else:
            assert len(channel_multiples) == len(updown_rates)
        self.down = UnetDown(
            hidden_size,
            down_layers,
            kernel_size,
            updown_rates,
            channel_multiples,
            is_BTC,
            constant_channels,
        )
        down_out_dims = int(np.cumprod(channel_multiples)[-1] * hidden_size) if not constant_channels else hidden_size
        self.mid = UnetMid(
            hidden_size,
            kernel_size,
            mid_layers,
            in_dims=down_out_dims,
            out_dims=down_out_dims,
            is_BTC=is_BTC,
            net=mid_net,
        )
        self.up = UnetUp(
            hidden_size,
            up_layers,
            kernel_size,
            updown_rates,
            channel_multiples,
            is_BTC,
            constant_channels,
            use_skip_layer,
            skip_scale,
        )

    def forward(self, x, mid_cond=None, **kwargs):
        x, skips = self.down(x)
        x = self.mid(x, mid_cond)
        return self.up(x, skips)
