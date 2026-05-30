"""PointNet++ classifier feature extractor used by PFD.

Loads the Point-E PointNet checkpoint
(``https://openaipublic.azureedge.net/main/point-e/pointnet.pt``) into a
``width_mult=2``, ``num_class=40``, ``normal_channel=False`` model and exposes
``extract(points: (N, K, 3)) -> (N, feature_dim)`` where feature_dim is 512.

The checkpoint is auto-downloaded on first use to ``evals/checkpoints/pointnet.pt``.
"""

from __future__ import annotations

import os
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from tqdm import tqdm

from evals.libs.pointnet2_cls_ssg import get_model


POINTNET_URL = "https://openaipublic.azureedge.net/main/point-e/pointnet.pt"
DEFAULT_CKPT = Path(__file__).resolve().parents[1] / "checkpoints" / "pointnet.pt"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        with (
            open(tmp, "wb") as f,
            tqdm(
                total=total, unit="iB", unit_scale=True, desc=f"download {dest.name}"
            ) as bar,
        ):
            while True:
                chunk = r.read(1 << 14)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(tmp, dest)


def _normalize_point_clouds(pc: np.ndarray) -> np.ndarray:
    """Center each cloud at origin and scale by max radius."""
    centroids = np.mean(pc, axis=1, keepdims=True)
    pc = pc - centroids
    m = np.max(np.sqrt(np.sum(pc**2, axis=-1, keepdims=True)), axis=1, keepdims=True)
    return pc / np.clip(m, 1e-8, None)


@lru_cache(maxsize=2)
def load(device: str = "cuda:0", ckpt_path: str | None = None) -> tuple:
    """Returns the loaded eval-mode model. Cached per (device, ckpt)."""
    path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    if not path.exists():
        print(f"[pointnet_cls] checkpoint missing; downloading to {path}")
        _download(POINTNET_URL, path)
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    model = get_model(num_class=40, normal_channel=False, width_mult=2)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract(
    points: np.ndarray, *, device: str = "cuda:0", batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (features, softmax_probs) for an array of point clouds.

    Args:
        points: ``(N, K, 3)`` float array.

    Returns:
        features: ``(N, feature_dim)`` float32 array (penultimate layer).
        probs:    ``(N, num_classes)`` float32 array of softmax probabilities.
    """
    model = load(device=device)
    pts = _normalize_point_clouds(points.astype(np.float32))

    feats_out: list[np.ndarray] = []
    probs_out: list[np.ndarray] = []
    for i in range(0, len(pts), batch_size):
        batch = (
            torch.from_numpy(pts[i : i + batch_size])
            .to(device)
            .permute(0, 2, 1)
            .float()
        )
        log_probs, _, feats = model(batch, features=True)
        feats_out.append(feats.cpu().numpy())
        probs_out.append(log_probs.exp().cpu().numpy())
    return np.concatenate(feats_out, axis=0), np.concatenate(probs_out, axis=0)


def load_npz_set(npz_dir: Path, sample_ids: Sequence[str]) -> np.ndarray:
    """Load all ``{npz_dir}/{sample_id}.npz`` into a single (N, K, 3) array, skipping missing files."""
    clouds = []
    missing = []
    for sid in sample_ids:
        p = npz_dir / f"{sid}.npz"
        if not p.exists():
            missing.append(sid)
            continue
        arr = np.load(p)["pointcloud"]  # (1, K, 3)
        clouds.append(arr[0])
    if missing:
        print(
            f"[pointnet_cls] missing {len(missing)} npz files (first 5): {missing[:5]}"
        )
    if not clouds:
        raise FileNotFoundError(f"no npz files found in {npz_dir}")
    return np.stack(clouds, axis=0)
