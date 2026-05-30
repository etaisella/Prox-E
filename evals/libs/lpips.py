"""LPIPS perceptual distance wrapper used by Identity Preservation.

image tensors are normalised to ``[-1, 1]`` before being passed to ``lpips.LPIPS``.
Default backbone is VGG (more aligned with human perception than ``alex`` for
shape-edit evaluation; pass ``net='alex'`` to match the legacy default).
"""

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import lpips as _lpips
import torch
from PIL import Image


@lru_cache(maxsize=2)
def load(net: str = "vgg", device: str = "cuda:0"):
    return _lpips.LPIPS(net=net).to(device).eval()


def _load_image(path: Path, image_size: int) -> torch.Tensor:
    img = (
        Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    )
    arr = torch.as_tensor(list(img.getdata()), dtype=torch.float32).reshape(
        image_size, image_size, 3
    )
    arr = (
        arr.permute(2, 0, 1).unsqueeze(0) / 255.0 * 2.0 - 1.0
    )  # (1, 3, H, W) in [-1, 1]
    return arr


@torch.no_grad()
def distance(
    pred_paths: Sequence[Path],
    gt_paths: Sequence[Path],
    *,
    device: str = "cuda:0",
    image_size: int = 512,
    net: str = "vgg",
) -> float:
    """Per-view LPIPS, averaged. ``pred_paths`` and ``gt_paths`` must align 1:1."""
    if len(pred_paths) != len(gt_paths):
        raise ValueError(
            f"len(pred_paths)={len(pred_paths)} != len(gt_paths)={len(gt_paths)}"
        )
    model = load(net=net, device=device)
    pred = torch.cat([_load_image(p, image_size) for p in pred_paths]).to(device)
    gt = torch.cat([_load_image(p, image_size) for p in gt_paths]).to(device)
    return float(model(pred, gt).mean().item())
