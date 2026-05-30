"""l-GD (localized Geometric Difference).

Per-shape Chamfer distance computed on the points NOT belonging to the
language-referenced part(s). The pipeline:

    1. Segment the point cloud with a small PointNet ONNX model trained per
       object class (chair / table / lamp / ...). Auto-downloaded from
       ``ailia-models/pointnet_pytorch/``.
    2. Build a boolean mask of the points labelled as the referenced part.
    3. Compute Chamfer distance between pred and gt restricted to the
       *non-edited* points (i.e. the complement of the mask), scaled by 1000
       to match the ChangeIt3D convention.

Required per sample:
    pred_pcd (N, 3)        gt_pcd (N, 3)
    obj_class              one of CATEGORIES below
    part_keyword           e.g. "leg", "seat", or "unknown" (sample is skipped)

This file is intentionally self-contained — no ``third_party/`` import, no
vendored ChangeIt3D tree. Only depends on numpy, torch, onnxruntime, tqdm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

try:
    import open3d as o3d
except ImportError:  # pragma: no cover - only needed for optional alignment
    o3d = None


# ---------------------------------------------------------------------------
# Part-label tables and instruction parsing live in :mod:`evals.libs.part_words`
# so light-weight callers (e.g. dataset prep scripts) can use them without
# pulling in onnxruntime / torch.
from evals.libs.part_words import CATEGORIES, PART_KEYWORDS, auto_resolve_part  # noqa: F401

ONNX_URL_TEMPLATE = (
    "https://storage.googleapis.com/ailia-models/pointnet_pytorch/{cls}_100.onnx"
)
SCALE_CHAMFER_BY = 1000.0  # matches ChangeIt3D upstream


def _download_onnx(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"[lgd] downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 14)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


@lru_cache(maxsize=32)
def _session(onnx_path: str) -> ort.InferenceSession:
    """Cached ONNX session per file path."""
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, os.cpu_count() // 2 or 1)
    return ort.InferenceSession(
        onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
    )


def load_segmenter(obj_class: str, ckpt_dir: Path) -> ort.InferenceSession:
    """Load (or download) the per-class ONNX segmenter."""
    if obj_class not in CATEGORIES:
        raise KeyError(
            f"no segmenter available for class {obj_class!r}; "
            f"supported: {sorted(CATEGORIES)}"
        )
    onnx_path = ckpt_dir / f"{obj_class}_100.onnx"
    if not onnx_path.exists():
        _download_onnx(ONNX_URL_TEMPLATE.format(cls=obj_class), onnx_path)
    return _session(str(onnx_path))


def _segment_preprocess(pc: np.ndarray) -> np.ndarray:
    """Unit-sphere normalisation expected by the ONNX seg model.

    ChangeIt3D's refined path first calls ``PointCloud.load_shapetalk``, which
    maps raw ShapeTalk coordinates ``(x, y, z) -> (-z, x, y)``. Its
    ``segment_pointcloud`` then applies ``x *= -1`` and ``(x, y, z) ->
    (y, z, x)``, cancelling the loader transform and feeding the segmenter the
    original raw coordinates.
    """
    pts = pc.astype(np.float32).copy()
    pts -= pts.mean(axis=0, keepdims=True)
    r = np.sqrt((pts**2).sum(axis=1)).max()
    if r > 0:
        pts /= r
    return pts.T[None].astype(np.float32)  # (1, 3, N)


def segment(pc: np.ndarray, session: ort.InferenceSession) -> np.ndarray:
    """Run the per-class ONNX segmenter, returning a per-point label ``(N,)`` array."""
    pred, _trans = session.run(None, {"point": _segment_preprocess(pc)})
    return pred[0].argmax(axis=1).astype(np.int64)


def _part_indices(obj_class: str, part_keyword: str) -> List[int]:
    table = CATEGORIES[obj_class]
    if part_keyword not in table:
        return []
    return list(table[part_keyword])


def part_mask(labels: np.ndarray, obj_class: str, part_keyword: str) -> np.ndarray:
    """True where the point belongs to one of the part indices for the given keyword."""
    idxs = _part_indices(obj_class, part_keyword)
    if not idxs:
        return np.zeros_like(labels, dtype=bool)
    return np.isin(labels, idxs)


# ---------------------------------------------------------------------------
# Chamfer + masking
# ---------------------------------------------------------------------------


def _center_in_unit_sphere(pc: np.ndarray) -> np.ndarray:
    """Translate so the bounding box is centred at the origin, then scale by max radius.
    Same convention ChangeIt3D applies to both pred and gt before l-GD.
    """
    out = pc.astype(np.float32).copy()
    bbox_mid = (out.max(axis=0) + out.min(axis=0)) / 2.0
    out -= bbox_mid
    r = np.sqrt((out**2).sum(axis=1)).max()
    if r > 0:
        out /= r
    return out


def align_with_segmentation_masks(
    source_pts: np.ndarray,
    pred_pts: np.ndarray,
    source_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> np.ndarray:
    """Align pred to source using the non-edited anchor masks.

    This mirrors ChangeIt3D's refined ``align_with_segmentation_masks``: centroid
    and scale calibration on anchor regions, followed by point-to-point ICP on
    those anchors, then the transform is applied to the full prediction.
    """
    if o3d is None:
        raise ImportError("open3d is required for l-GD point-cloud alignment")

    src_anchor = source_pts[source_mask]
    pred_anchor = pred_pts[pred_mask]
    if len(src_anchor) == 0 or len(pred_anchor) == 0:
        return pred_pts

    pcd_src_anchor = o3d.geometry.PointCloud()
    pcd_src_anchor.points = o3d.utility.Vector3dVector(src_anchor)

    pcd_pred_anchor = o3d.geometry.PointCloud()
    pcd_pred_anchor.points = o3d.utility.Vector3dVector(pred_anchor)

    pcd_full_pred = o3d.geometry.PointCloud()
    pcd_full_pred.points = o3d.utility.Vector3dVector(pred_pts)

    mean_src = np.mean(src_anchor, axis=0)
    mean_pred = np.mean(pred_anchor, axis=0)
    scale_src = np.mean(np.linalg.norm(src_anchor - mean_src, axis=1))
    scale_pred = np.mean(np.linalg.norm(pred_anchor - mean_pred, axis=1))
    if scale_src <= 0 or scale_pred <= 0:
        return pred_pts
    scale_factor = scale_src / scale_pred

    pcd_full_pred.translate(-mean_pred)
    pcd_pred_anchor.translate(-mean_pred)
    pcd_full_pred.scale(scale_factor, center=(0, 0, 0))
    pcd_pred_anchor.scale(scale_factor, center=(0, 0, 0))
    pcd_full_pred.translate(mean_src)
    pcd_pred_anchor.translate(mean_src)

    threshold = 0.05 * scale_src
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd_pred_anchor,
        pcd_src_anchor,
        threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
    )
    pcd_full_pred.transform(reg_p2p.transformation)
    return np.asarray(pcd_full_pred.points, dtype=np.float32)


@torch.no_grad()
def masked_chamfer(
    a: torch.Tensor,
    b: torch.Tensor,
    keep_a: torch.Tensor,
    keep_b: torch.Tensor,
) -> torch.Tensor:
    """Masked Chamfer distance.

    Args:
        a, b:           (B, N, 3) float tensors.
        keep_a, keep_b: (B, N) bool tensors -- True = include this point.

    Returns:
        (B,) float tensor of per-shape mean distances. Points where the mask is
        False are pushed to ``1e6`` so they self-match (no spurious contribution)
        and are then zero-weighted in the final average.
    """
    LARGE = 1e6
    a_in = a.clone()
    b_in = b.clone()
    a_in[~keep_a] = LARGE
    b_in[~keep_b] = LARGE
    d2 = torch.cdist(a_in, b_in, p=2) ** 2  # (B, N, N)
    cd_a, _ = d2.min(dim=2)  # (B, N)
    cd_b, _ = d2.min(dim=1)  # (B, N)
    cd_a = cd_a * keep_a.float()
    cd_b = cd_b * keep_b.float()
    n_a = keep_a.sum(dim=1).clamp_min(1).float()
    n_b = keep_b.sum(dim=1).clamp_min(1).float()
    return (cd_a.sum(dim=1) + cd_b.sum(dim=1)) / (n_a + n_b)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


class LgdSample:
    __slots__ = ("sample_id", "pred", "gt", "obj_class", "part_keyword")

    def __init__(
        self,
        sample_id: str,
        pred: np.ndarray,
        gt: np.ndarray,
        obj_class: str,
        part_keyword: str,
    ):
        self.sample_id = sample_id
        self.pred = pred
        self.gt = gt
        self.obj_class = obj_class
        self.part_keyword = part_keyword


def compute_lgd(
    samples: Sequence[LgdSample],
    *,
    ckpt_dir: Path,
    device: str = "cuda:0",
    normalize: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Compute l-GD for a list of samples.

    Returns:
        (per_sample, per_class, summary)
            per_sample: {sample_id -> l-GD}
            per_class:  {obj_class -> mean l-GD}
            summary:    {"average": float, "n_evaluated": int, "n_skipped": int}
    """
    if not samples:
        return {}, {}, {"average": float("nan"), "n_evaluated": 0, "n_skipped": 0}

    dev = torch.device(
        device if torch.cuda.is_available() or device == "cpu" else "cpu"
    )
    per_sample: Dict[str, float] = {}
    skipped: List[str] = []

    # Process per class so we load each ONNX once.
    by_class: Dict[str, List[LgdSample]] = {}
    for s in samples:
        if s.part_keyword == "unknown" or not s.part_keyword:
            skipped.append(s.sample_id)
            continue
        if s.obj_class not in CATEGORIES:
            skipped.append(s.sample_id)
            continue
        by_class.setdefault(s.obj_class, []).append(s)

    for cls, cls_samples in by_class.items():
        session = load_segmenter(cls, ckpt_dir)
        for s in tqdm(cls_samples, desc=f"lgd[{cls}]"):
            labels_gt = segment(s.gt, session)
            labels_pred = segment(s.pred, session)
            mask_gt = part_mask(labels_gt, cls, s.part_keyword)
            mask_pred = part_mask(labels_pred, cls, s.part_keyword)
            if not mask_gt.any() and not mask_pred.any():
                skipped.append(s.sample_id)
                continue
            if mask_gt.all() or mask_pred.all():
                skipped.append(s.sample_id)
                continue
            keep_gt = ~mask_gt
            keep_pred = ~mask_pred

            pred = (
                _center_in_unit_sphere(s.pred)
                if normalize
                else s.pred.astype(np.float32)
            )
            gt = _center_in_unit_sphere(s.gt) if normalize else s.gt.astype(np.float32)
            pred = align_with_segmentation_masks(gt, pred, keep_gt, keep_pred)

            t_pred = torch.from_numpy(pred).unsqueeze(0).to(dev)
            t_gt = torch.from_numpy(gt).unsqueeze(0).to(dev)
            t_kp = torch.from_numpy(keep_pred).unsqueeze(0).to(dev)
            t_kg = torch.from_numpy(keep_gt).unsqueeze(0).to(dev)

            d = masked_chamfer(t_gt, t_pred, t_kg, t_kp)
            per_sample[s.sample_id] = float(d.item() * SCALE_CHAMFER_BY)

    per_class: Dict[str, float] = {}
    for cls, cls_samples in by_class.items():
        vs = [per_sample[s.sample_id] for s in cls_samples if s.sample_id in per_sample]
        if vs:
            per_class[cls] = float(np.mean(vs))

    all_vs = list(per_sample.values())
    summary = {
        "average": float(np.mean(all_vs)) if all_vs else float("nan"),
        "n_evaluated": len(per_sample),
        "n_skipped": len(skipped),
    }
    return per_sample, per_class, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_pcd(npz_path: Path) -> np.ndarray:
    arr = np.load(npz_path)["pointcloud"]
    if arr.ndim == 3:
        arr = arr[0]
    return arr.astype(np.float32)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Compute l-GD on a directory of paired pred/gt point clouds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pred_pcd_dir",
        type=Path,
        required=True,
        help="Directory of predicted {sample_id}.npz (key 'pointcloud', (1, K, 3))",
    )
    p.add_argument(
        "--gt_pcd_dir",
        type=Path,
        required=True,
        help="Directory of GT {sample_id}.npz (same format)",
    )
    p.add_argument(
        "--samples_json",
        type=Path,
        required=True,
        help='JSON mapping sample_id -> {"obj_class": str, "part_keyword": str} '
        'or {"obj_class": str, "instruction": str} (part is then auto-resolved)',
    )
    p.add_argument(
        "--ckpt_dir",
        type=Path,
        default=Path("evals/checkpoints/lgd"),
        help="Where the per-class *_100.onnx files live (auto-downloaded if missing)",
    )
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--no_normalize_pcd",
        dest="normalize_pcd",
        action="store_false",
        help="Disable bbox unit-sphere normalization before Chamfer",
    )
    p.set_defaults(normalize_pcd=True)
    args = p.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_meta = json.loads(args.samples_json.read_text())

    samples: List[LgdSample] = []
    missing: List[str] = []
    for sid, meta in samples_meta.items():
        pred_path = args.pred_pcd_dir / f"{sid}.npz"
        gt_path = args.gt_pcd_dir / f"{sid}.npz"
        if not pred_path.exists() or not gt_path.exists():
            missing.append(sid)
            continue
        obj_class = meta.get("obj_class")
        if obj_class is None:
            missing.append(sid)
            continue
        part_keyword = meta.get("part_keyword")
        if part_keyword is None and "instruction" in meta:
            part_keyword = (
                auto_resolve_part(meta["instruction"], obj_class) or "unknown"
            )
        if part_keyword is None:
            part_keyword = "unknown"
        samples.append(
            LgdSample(
                sid, _load_pcd(pred_path), _load_pcd(gt_path), obj_class, part_keyword
            )
        )

    if missing:
        print(
            f"[lgd] missing inputs for {len(missing)} samples (first 5: {missing[:5]})"
        )

    per_sample, per_class, summary = compute_lgd(
        samples,
        ckpt_dir=args.ckpt_dir,
        device=args.device,
        normalize=args.normalize_pcd,
    )

    (args.output_dir / "lgd_results.json").write_text(
        json.dumps(
            {
                "per_class": per_class,
                **summary,
            },
            indent=2,
        )
    )
    (args.output_dir / "lgd_per_sample.json").write_text(
        json.dumps({sid: {"l-GD": v} for sid, v in per_sample.items()}, indent=2)
    )

    print(json.dumps({"per_class": per_class, **summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
