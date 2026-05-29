from collections import namedtuple

import torch
import torch.nn.functional as F
from packaging import version
from torch import einsum, nn

FlashAttentionConfig = namedtuple("FlashAttentionConfig", ["enable_flash", "enable_math", "enable_mem_efficient"])


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


class Attend(nn.Module):
    """Kernel attention for MelBandRoformer linear-attention blocks (inference-only)."""

    def __init__(self, flash=False, scale=None):
        super().__init__()
        self.scale = scale
        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse("2.0.0")), (
            "flash attention requires pytorch 2.0+"
        )

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None
        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device("cuda"))
        device_version = version.parse(f"{device_properties.major}.{device_properties.minor}")
        if device_version >= version.parse("8.0"):
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        config = self.cuda_config if q.is_cuda else self.cpu_config
        with torch.backends.cuda.sdp_kernel(**config._asdict()):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    def forward(self, q, k, v):
        if self.flash:
            return self.flash_attn(q, k, v)

        scale = default(self.scale, q.shape[-1] ** -0.5)
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        attn = sim.softmax(dim=-1)
        return einsum("b h i j, b h j d -> b h i d", attn, v)
