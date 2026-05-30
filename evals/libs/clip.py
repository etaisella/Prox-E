"""OpenAI CLIP wrapper used by the Edit Fidelity metrics."""

from functools import lru_cache
from pathlib import Path
from typing import List, Sequence

import clip
import torch
from PIL import Image


@lru_cache(maxsize=2)
def load(model_name: str = "ViT-B/32", device: str = "cuda:0"):
    """Cached loader; returns (model, preprocess). One instance per (name, device)."""
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


@torch.no_grad()
def encode_image_paths(
    paths: Sequence[Path], model, preprocess, device: str
) -> torch.Tensor:
    """Returns a (len(paths), feature_dim) tensor of un-normalised CLIP image features."""
    if not paths:
        raise ValueError("encode_image_paths got no paths")
    batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths]).to(
        device
    )
    return model.encode_image(batch).float()


@torch.no_grad()
def encode_text(text: str, model, device: str) -> torch.Tensor:
    """Returns a (1, feature_dim) tensor of un-normalised CLIP text features."""
    tokens = clip.tokenize([text]).to(device)
    return model.encode_text(tokens).float()


def render_paths_for(render_root: Path, sample_id: str) -> List[Path]:
    """Resolve render paths for a sample. Supports both single-view ({id}.png) and
    multi-view ({id}/view_NN.png) layouts produced by :mod:`evals.preprocessing`."""
    single = render_root / f"{sample_id}.png"
    if single.exists():
        return [single]
    folder = render_root / sample_id
    if folder.is_dir():
        return sorted(folder.glob("*.png"))
    return []
