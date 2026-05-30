"""Identity Preservation metrics: l-GD, LPIPS, DINO-I.

Per-sample LPIPS and DINO-I are averaged over views and reduced to a single
float per sample. l-GD comes from :mod:`evals.libs.lgd` -- a self-contained
module that segments each point cloud with a per-class ONNX PointNet
(auto-downloaded from ``ailia-models``) and computes masked Chamfer on the
non-edited region.

For users without ``obj_class``/``part`` annotations, l-GD is skipped and the
remaining identity metrics still run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from evals.libs import lpips as lpips_mod, dino as dino_mod, lgd as lgd_mod


def _transform_pred_pcd(points: np.ndarray, model_name: str) -> np.ndarray:
    """Coordinate transforms for generated point clouds."""
    out = points.copy()
    if model_name == "default":
        out[:, 0] *= -1.0
        out[:, [0, 2, 1]] = out[:, [0, 1, 2]]
    elif model_name == "proxe":
        out[:, 0] *= -1.0
        out[:, [0, 2, 1]] = out[:, [0, 1, 2]]
        out[:, 2] *= -1.0
    elif model_name == "blendedpc":
        out[:, 0] *= -1.0
        out[:, [2, 0, 1]] = out[:, [0, 1, 2]]
    elif model_name != "none":
        raise ValueError(f"unknown l-GD pred point-cloud transform: {model_name}")
    return out


def _aligned_views(
    pred_root: Path, gt_root: Path, sid: str
) -> Tuple[List[Path], List[Path]]:
    """Return aligned (pred_view_paths, gt_view_paths) for one sample.
    Single view: ``{root}/{sid}.png``. Multi-view: ``{root}/{sid}/view_NN.png``.
    """
    pred_single = pred_root / f"{sid}.png"
    gt_single = gt_root / f"{sid}.png"
    if pred_single.exists() and gt_single.exists():
        return [pred_single], [gt_single]
    pred_dir = pred_root / sid
    gt_dir = gt_root / sid
    if not pred_dir.is_dir() or not gt_dir.is_dir():
        return [], []
    pred_views = sorted(pred_dir.glob("*.png"))
    gt_views = sorted(gt_dir.glob("*.png"))
    if len(pred_views) != len(gt_views):
        return [], []
    return pred_views, gt_views


def _load_pcd(npz_path: Path) -> np.ndarray:
    arr = np.load(npz_path)["pointcloud"]
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float32)


def _build_lgd_samples(
    sample_ids: List[str],
    pcd_dir: Path,
    instructions: Dict,
) -> List[lgd_mod.LgdSample]:
    """Convert the unified evaluator's metadata into LgdSample objects."""
    samples: List[lgd_mod.LgdSample] = []
    for sid in sample_ids:
        meta = instructions.get(sid)
        if not isinstance(meta, dict):
            continue
        obj_class = meta.get("obj_class")
        if obj_class is None:
            continue
        part_keyword = meta.get("part_keyword") or meta.get("part")
        if part_keyword is None and "instruction" in meta:
            part_keyword = lgd_mod.auto_resolve_part(meta["instruction"], obj_class)
        if not part_keyword:
            continue
        pred_npz = pcd_dir / "pred" / f"{sid}.npz"
        gt_npz = pcd_dir / "gt" / f"{sid}.npz"
        if not pred_npz.exists() or not gt_npz.exists():
            continue
        samples.append(
            lgd_mod.LgdSample(
                sample_id=sid,
                pred=_load_pcd(pred_npz),
                gt=_load_pcd(gt_npz),
                obj_class=obj_class,
                part_keyword=part_keyword,
            )
        )
    return samples


def compute(
    *,
    sample_ids: List[str],
    render_dir: Path,
    pcd_dir: Path,
    device: str = "cuda:0",
    instructions: Optional[Dict] = None,
    lgd_ckpt_dir: Optional[Path] = None,
    lgd_normalize_pcd: bool = True,
    lgd_pred_pcd_transform: str = "proxe",
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """Compute identity-preservation metrics. ``instructions`` is only consulted
    for l-GD's per-sample ``obj_class`` / ``part_keyword``."""
    pred_root = render_dir / "pred"
    gt_root = render_dir / "gt"

    per_sample: Dict[str, Dict[str, float]] = {sid: {} for sid in sample_ids}
    lpips_vals: List[float] = []
    dino_vals: List[float] = []

    pred_view_lists: List[List[Path]] = []
    gt_view_lists: List[List[Path]] = []
    aligned_ids: List[str] = []
    for sid in sample_ids:
        p, g = _aligned_views(pred_root, gt_root, sid)
        if not p:
            continue
        pred_view_lists.append(p)
        gt_view_lists.append(g)
        aligned_ids.append(sid)

    for sid, p_views, g_views in tqdm(
        list(zip(aligned_ids, pred_view_lists, gt_view_lists)), desc="LPIPS"
    ):
        d = lpips_mod.distance(p_views, g_views, device=device)
        per_sample[sid]["LPIPS"] = d
        lpips_vals.append(d)

    flat_pred = [p for views in pred_view_lists for p in views]
    flat_gt = [p for views in gt_view_lists for p in views]
    if flat_pred:
        flat_sims = dino_mod.cosine_pairs(flat_pred, flat_gt, device=device)
        idx = 0
        for sid, p_views in zip(aligned_ids, pred_view_lists):
            v = flat_sims[idx : idx + len(p_views)]
            idx += len(p_views)
            mean_sim = float(np.mean(v))
            per_sample[sid]["DINO-I"] = mean_sim
            dino_vals.append(mean_sim)

    aggregate: Dict[str, float] = {
        "LPIPS": float(np.mean(lpips_vals)) if lpips_vals else float("nan"),
        "DINO-I": float(np.mean(dino_vals)) if dino_vals else float("nan"),
    }

    if instructions is not None:
        ckpt_dir = lgd_ckpt_dir or (
            pcd_dir.parent.parent / "evals" / "checkpoints" / "lgd"
        )
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        lgd_samples = _build_lgd_samples(sample_ids, pcd_dir, instructions)
        if lgd_samples:
            if lgd_pred_pcd_transform != "none":
                for sample in lgd_samples:
                    sample.pred = _transform_pred_pcd(
                        sample.pred, lgd_pred_pcd_transform
                    )
            per_sample_lgd, per_class, summary = lgd_mod.compute_lgd(
                lgd_samples,
                ckpt_dir=ckpt_dir,
                device=device,
                normalize=lgd_normalize_pcd,
            )
            aggregate["l-GD"] = summary["average"]
            aggregate["l-GD_per_class"] = per_class
            aggregate["l-GD_n_evaluated"] = summary["n_evaluated"]
            aggregate["l-GD_n_skipped"] = summary["n_skipped"]
            for sid, v in per_sample_lgd.items():
                per_sample.setdefault(sid, {})["l-GD"] = v

    return aggregate, per_sample
