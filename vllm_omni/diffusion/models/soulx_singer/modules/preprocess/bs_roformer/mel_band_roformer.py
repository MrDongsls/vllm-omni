from collections.abc import Callable
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from rotary_embedding_torch import RotaryEmbedding

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention as DiffusionAttention
from vllm_omni.diffusion.models.soulx_singer.modules.preprocess.utils import load_pt_state_dict
from vllm_omni.utils.audio import mel_filter_bank as build_mel_filter_bank

from .attend import Attend

# helper functions


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


# norm


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# attention


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Linear(dim_inner, dim),
        )

    def forward(self, x):
        return self.net(x)


class RoformerSelfAttention(nn.Module):
    """Self-attention block using vLLM-Omni ``DiffusionAttention`` backend."""

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        rotary_embed=None,
        *,
        role: str = "soulx.roformer.self",
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)
        self.attn = DiffusionAttention(
            num_heads=heads,
            head_size=dim_head,
            softmax_scale=dim_head**-0.5,
            causal=False,
            num_kv_heads=heads,
            role=role,
        )

    def forward(self, x):
        x_normed = self.norm(x)
        q, k, v = rearrange(
            self.to_qkv(x_normed),
            "b n (qkv h d) -> qkv b h n d",
            qkv=3,
            h=self.heads,
            d=self.dim_head,
        )

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        q = rearrange(q, "b h n d -> b n h d")
        k = rearrange(k, "b h n d -> b n h d")
        v = rearrange(v, "b h n d -> b n h d")
        out = self.attn(q, k, v, attn_metadata=AttentionMetadata())

        gates = self.to_gates(x_normed)
        out = out * rearrange(gates, "b n h -> b n h 1").sigmoid()
        out = rearrange(out, "b n h d -> b n (h d)")
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    this flavor of linear attention proposed in https://arxiv.org/abs/2106.09681 by El-Nouby et al.
    """

    def __init__(
        self,
        *,
        dim,
        dim_head=32,
        heads=8,
        scale=8,
        flash=False,
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange("b n (qkv h d) -> qkv b h d n", qkv=3, h=heads),
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.attend = Attend(scale=scale, flash=flash)
        self.to_out = nn.Sequential(
            Rearrange("b h d n -> b n (h d)"),
            nn.Linear(dim_inner, dim, bias=False),
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)


class Transformer(nn.Module):
    _repeated_blocks = ["layers"]

    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
        linear_attn=False,
        attn_role: str = "soulx.roformer.self",
        **_ignored,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for layer_idx in range(depth):
            if linear_attn:
                attn = LinearAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    flash=flash_attn,
                )
            else:
                attn = RoformerSelfAttention(
                    dim=dim,
                    dim_head=dim_head,
                    heads=heads,
                    rotary_embed=rotary_embed,
                    role=f"{attn_role}.{layer_idx}",
                )

            self.layers.append(
                nn.ModuleList(
                    [
                        attn,
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# bandsplit module


class BandSplit(nn.Module):
    def __init__(self, dim, dim_inputs: tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(RMSNorm(dim_in), nn.Linear(dim_in, dim))

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(dim_in, dim_out, dim_hidden=None, depth=1, activation=nn.Tanh):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(nn.Module):
    def __init__(self, dim, dim_inputs: tuple[int, ...], depth, mlp_expansion_factor=4):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = nn.ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth), nn.GLU(dim=-1))

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)


# main class


class MelBandRoformer(nn.Module):
    _repeated_blocks = ["layers"]
    _layerwise_offload_blocks_attr = "layers"

    def __init__(
        self,
        dim,
        *,
        depth,
        stereo=False,
        num_stems=1,
        time_transformer_depth=2,
        freq_transformer_depth=2,
        linear_transformer_depth=0,
        num_bands=60,
        dim_head=64,
        heads=8,
        flash_attn=True,
        dim_freqs_in=1025,
        sample_rate=44100,
        stft_n_fft=2048,
        stft_hop_length=512,
        stft_win_length=2048,
        stft_normalized=False,
        stft_window_fn: Callable | None = None,
        mask_estimator_depth=1,
        match_input_audio_length=False,
        mlp_expansion_factor=4,
        skip_connection=False,
        **_ignored,
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.skip_connection = skip_connection

        self.layers = nn.ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            flash_attn=flash_attn,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for block_idx in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                tran_modules.append(
                    Transformer(
                        depth=linear_transformer_depth,
                        linear_attn=True,
                        attn_role=f"soulx.roformer.linear.{block_idx}",
                        **transformer_kwargs,
                    )
                )
            tran_modules.append(
                Transformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    attn_role=f"soulx.roformer.time.{block_idx}",
                    **transformer_kwargs,
                )
            )
            tran_modules.append(
                Transformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    attn_role=f"soulx.roformer.freq.{block_idx}",
                    **transformer_kwargs,
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft, hop_length=stft_hop_length, win_length=stft_win_length, normalized=stft_normalized
        )

        freqs = torch.stft(
            torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_n_fft), return_complex=True
        ).shape[1]

        # create mel filter bank
        # with librosa.filters.mel as in section 2 of paper

        mel_filter_bank_numpy = build_mel_filter_bank(
            sr=sample_rate,
            n_fft=stft_n_fft,
            n_mels=num_bands,
        ).numpy()

        mel_filter_bank_tensor = torch.from_numpy(mel_filter_bank_numpy)

        # for some reason, it doesn't include the first freq? just force a value for now

        mel_filter_bank_tensor[0][0] = 1.0

        # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
        # so let's force a positive value

        mel_filter_bank_tensor[-1, -1] = 1.0

        # binary as in paper (then estimated masks are averaged for overlapping regions)

        freqs_per_band = mel_filter_bank_tensor > 0
        assert freqs_per_band.any(dim=0).all(), "all frequencies need to be covered by all bands for now"

        repeated_freq_indices = repeat(torch.arange(freqs), "f -> b f", b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, "f -> f s", s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, "f s -> (f s)")

        self.register_buffer("freq_indices", freq_indices, persistent=False)
        self.register_buffer("freqs_per_band", freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, "b f -> b", "sum")
        num_bands_per_freq = reduce(freqs_per_band, "b f -> f", "sum")

        self.register_buffer("num_freqs_per_band", num_freqs_per_band, persistent=False)
        self.register_buffer("num_bands_per_freq", num_bands_per_freq, persistent=False)

        # band split and mask estimator

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())

        self.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )

            self.mask_estimators.append(mask_estimator)

        self.match_input_audio_length = match_input_audio_length

    def load_checkpoint(self, checkpoint_path: str) -> None:
        load_pt_state_dict(self, checkpoint_path, strict=False)
        self.eval()

    @torch.inference_mode()
    def forward(self, raw_audio: torch.Tensor) -> torch.Tensor:
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, "b t -> b 1 t")

        batch, channels, raw_audio_length = raw_audio.shape

        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), (
            "stereo needs to be set to True if passing in audio signal that is stereo "
            "(channel dimension of 2). also need to be False if mono (channel dimension of 1)"
        )

        # to stft

        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, "* t")

        stft_window = self.stft_window_fn(device=device)

        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, "* f t c")

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        stft_repr = rearrange(stft_repr, "b s f t c -> b (f s) t c")

        # index out all frequencies for all frequency ranges across bands ascending in one go

        batch_arange = torch.arange(batch, device=device)[..., None]

        # account for stereo

        x = stft_repr[batch_arange, self.freq_indices]

        # fold the complex (real and imag) into the frequencies dimension

        x = rearrange(x, "b f t c -> b t (f c)")

        x = self.band_split(x)

        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):
            if len(transformer_block) == 3:
                linear_transformer, time_transformer, freq_transformer = transformer_block
                x, ft_ps = pack([x], "b * d")
                x = linear_transformer(x)
                (x,) = unpack(x, ft_ps, "b * d")
            else:
                time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            x = rearrange(x, "b t f d -> b f t d")
            x, ps = pack([x], "* t d")
            x = time_transformer(x)
            (x,) = unpack(x, ps, "* t d")
            x = rearrange(x, "b f t d -> b t f d")
            x, ps = pack([x], "* f d")
            x = freq_transformer(x)
            (x,) = unpack(x, ps, "* f d")

            if self.skip_connection:
                store[i] = x

        num_stems = len(self.mask_estimators)
        masks = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        masks = rearrange(masks, "b n t (f c) -> b n f t c", c=2)

        # modulate frequency representation

        stft_repr = rearrange(stft_repr, "b f t c -> b 1 f t c")

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks)

        masks = masks.type(stft_repr.dtype)

        # need to average the estimated mask for the overlapped frequencies

        scatter_indices = repeat(self.freq_indices, "f -> b n f t", b=batch, n=num_stems, t=stft_repr.shape[-1])

        stft_repr_expanded_stems = repeat(stft_repr, "b 1 ... -> b n ...", n=num_stems)
        masks_summed = torch.zeros_like(stft_repr_expanded_stems).scatter_add_(2, scatter_indices, masks)

        denom = repeat(self.num_bands_per_freq, "f -> (f r) 1", r=channels)

        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        # modulate stft repr with estimated mask

        stft_repr = stft_repr * masks_averaged

        # istft

        stft_repr = rearrange(stft_repr, "b n (f s) t -> (b n s) f t", s=self.audio_channels)

        recon_audio = torch.istft(
            stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=istft_length
        )

        recon_audio = rearrange(recon_audio, "(b n s) t -> b n s t", b=batch, s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, "b 1 s t -> b s t")

        return recon_audio
