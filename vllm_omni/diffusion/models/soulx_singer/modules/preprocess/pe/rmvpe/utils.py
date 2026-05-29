import numpy as np
import torch

from .constants import CONST, N_CLASS


def to_local_average_f0(hidden: torch.Tensor, thred: float = 0.03) -> np.ndarray:
    hidden = hidden.detach().float().cpu()
    idx = torch.arange(N_CLASS)[None, None, :]
    idx_cents = idx * 20 + CONST
    center = torch.argmax(hidden, dim=2, keepdim=True)
    start = torch.clip(center - 4, min=0)
    end = torch.clip(center + 5, max=N_CLASS)
    idx_mask = (idx >= start) & (idx < end)
    weights = hidden * idx_mask
    product_sum = torch.sum(weights * idx_cents, dim=2)
    weight_sum = torch.sum(weights, dim=2)
    cents = product_sum / (weight_sum + (weight_sum == 0))
    f0 = 10 * 2 ** (cents / 1200)
    uv = hidden.max(dim=2)[0] < thred
    return (f0 * ~uv).numpy()
