"""DINOv2 image feature extractor used by DINO-I (identity preservation).

Wraps ``facebook/dinov2-base`` via HuggingFace transformers. Per-image
features come from the mean of ``last_hidden_state``; similarity is the
cosine of L2-normalised features.
"""

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image


@lru_cache(maxsize=2)
def load(device: str = "cuda:0", model_name: str = "facebook/dinov2-base"):
    from transformers import AutoImageProcessor, Dinov2Model

    model = Dinov2Model.from_pretrained(model_name).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(model_name)
    return model, processor


def _open(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


@torch.no_grad()
def features(
    paths: Sequence[Path], *, device: str = "cuda:0", batch_size: int = 16
) -> torch.Tensor:
    """Returns (len(paths), feature_dim) L2-normalised feature tensor on ``device``."""
    if not paths:
        raise ValueError("dino.features got no paths")
    model, processor = load(device=device)
    out: list[torch.Tensor] = []
    for i in range(0, len(paths), batch_size):
        imgs = [_open(p) for p in paths[i : i + batch_size]]
        batch = processor(images=imgs, return_tensors="pt").to(device)
        feats = model(**batch).last_hidden_state.mean(dim=1)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out.append(feats)
    return torch.cat(out, dim=0)


def cosine_pairs(
    pred_paths: Sequence[Path],
    gt_paths: Sequence[Path],
    *,
    device: str = "cuda:0",
    batch_size: int = 16,
) -> list[float]:
    """Pairwise cosine similarity between aligned pred/gt images."""
    if len(pred_paths) != len(gt_paths):
        raise ValueError(
            f"len(pred_paths)={len(pred_paths)} != len(gt_paths)={len(gt_paths)}"
        )
    f_pred = features(pred_paths, device=device, batch_size=batch_size)
    f_gt = features(gt_paths, device=device, batch_size=batch_size)
    sims = (f_pred * f_gt).sum(dim=-1)
    return [float(x) for x in sims.cpu().tolist()]
