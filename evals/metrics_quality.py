"""3D Quality metrics: PFD, FID.

PFD: Fréchet distance between PointNet++ classifier features extracted from
      predicted vs. ground-truth point clouds.
FID:  Fréchet distance between InceptionV3 features over rendered images,
      via ``torchmetrics.image.fid.FrechetInceptionDistance``.

Both metrics are distribution-level (no per-sample value).
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from evals.libs import pointnet_cls
from evals.libs.fid import fid_from_features


def _list_render_paths(root: Path, sample_ids: List[str]) -> List[Path]:
    out: List[Path] = []
    for sid in sample_ids:
        single = root / f"{sid}.png"
        if single.exists():
            out.append(single)
            continue
        folder = root / sid
        if folder.is_dir():
            out.extend(sorted(folder.glob("*.png")))
    return out


def _to_uint8_tensor(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path)
    if img.mode != "RGB":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, (0, 0), img if img.mode == "RGBA" else None)
        img = bg.convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    t = T.ToTensor()(img)  # (3, H, W) float in [0, 1]
    return (t * 255).to(torch.uint8)


def _transform_pfd_point_clouds(points: np.ndarray, model_name: str) -> np.ndarray:
    """PFD coordinate transform for generated point clouds."""
    out = points.copy()
    if model_name == "trellis":
        out[..., 0] *= -1.0
        out[..., [0, 2, 1]] = out[..., [0, 1, 2]]
    elif model_name == "proxe":
        out[..., 0] *= -1.0
        out[..., [0, 2, 1]] = out[..., [0, 1, 2]]
        out[..., 2] *= -1.0
    elif model_name == "blendedpc":
        out[..., 0] *= -1.0
        out[..., [2, 0, 1]] = out[..., [0, 1, 2]]
    elif model_name != "none":
        raise ValueError(f"unknown PFD model_name: {model_name}")
    return out


def _transform_pfd_gt_point_clouds(points: np.ndarray, model_name: str) -> np.ndarray:
    """PFD coordinate transform for GT point clouds."""
    out = points.copy()
    if model_name == "trellis":
        out[..., 1] *= -1.0
    elif model_name not in {"proxe", "blendedpc", "none"}:
        raise ValueError(f"unknown PFD model_name: {model_name}")
    return out


def _image_fid(
    pred_paths: List[Path],
    gt_paths: List[Path],
    *,
    device: str,
    image_size: int = 512,
    batch_size: int = 16,
) -> float:
    fid = FrechetInceptionDistance(feature=2048).to(device)
    fid.reset()
    for side, paths, real in (("real", gt_paths, True), ("fake", pred_paths, False)):
        for i in tqdm(range(0, len(paths), batch_size), desc=f"fid:{side}"):
            chunk = paths[i : i + batch_size]
            batch = torch.stack([_to_uint8_tensor(p, image_size) for p in chunk]).to(
                device
            )
            fid.update(batch, real=real)
    return float(fid.compute().item())


def compute(
    *,
    sample_ids: List[str],
    render_dir: Path,
    pcd_dir: Path,
    device: str = "cuda:0",
    pfd_model_name: str = "proxe",
    pfd_src_pc_from: str = "gt",
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Returns (aggregate, per_sample). Per-sample is empty (both are distribution metrics)."""
    pred_pcd_dir = pcd_dir / "pred"
    gt_pcd_dir = pcd_dir / "gt"

    # PFD needs paired point clouds on both sides.
    if pred_pcd_dir.is_dir() and gt_pcd_dir.is_dir() and any(gt_pcd_dir.glob("*.npz")):
        pred_pcds = pointnet_cls.load_npz_set(pred_pcd_dir, sample_ids)
        gt_pcds = pointnet_cls.load_npz_set(gt_pcd_dir, sample_ids)
        if pfd_src_pc_from == "gt":
            pred_pcds = _transform_pfd_point_clouds(pred_pcds, pfd_model_name)
        elif pfd_src_pc_from == "proxe":
            gt_pcds = _transform_pfd_gt_point_clouds(gt_pcds, pfd_model_name)
        elif pfd_src_pc_from != "none":
            raise ValueError(f"unknown PFD src_pc_from: {pfd_src_pc_from}")
        pred_feats, _ = pointnet_cls.extract(pred_pcds, device=device)
        gt_feats, _ = pointnet_cls.extract(gt_pcds, device=device)
        pfd = fid_from_features(pred_feats, gt_feats)
    else:
        print("  [quality] no GT point clouds; PFD is NaN")
        pfd = float("nan")

    pred_imgs = _list_render_paths(render_dir / "pred", sample_ids)
    gt_imgs = _list_render_paths(render_dir / "gt", sample_ids)
    if not pred_imgs or not gt_imgs:
        print("  [quality] no GT renders; FID is NaN")
        fid = float("nan")
    else:
        fid = _image_fid(pred_imgs, gt_imgs, device=device)

    return {"PFD": pfd, "FID": fid}, {}
