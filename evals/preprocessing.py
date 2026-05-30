"""Mesh-rendering and mesh-to-point-cloud preprocessing for the unified evaluator.

Both ``render_meshes`` and ``mesh_to_pcd_pairs`` are cache-aware: per-sample
outputs that already exist on disk are skipped.

Rendering delegates to ``prox_e.utils.render_obj_with_blender_sequence`` (the
same Blender helper used by ``inference.py``). Point-cloud sampling: surface-sample with trimesh, voxel-downsample
with open3d, then uniform-subsample to ``num_points``.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh
from tqdm import tqdm

from prox_e.utils import render_obj_with_blender_sequence


Rotation = Optional[Tuple[float, float, float]]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_paths(out_root: Path, sample_id: str, num_views: int) -> List[Path]:
    """Single-view: out_root/{sample_id}.png; multi-view: out_root/{sample_id}/view_NN.png."""
    if num_views == 1:
        return [out_root / f"{sample_id}.png"]
    sub = out_root / sample_id
    return [sub / f"view_{i:02d}.png" for i in range(num_views)]


def _azims_for(num_views: int) -> List[float]:
    """Evenly-spaced azimuths around the object. Single view defaults to azim=70 to match
    the default of ``render_obj_with_blender`` (matches Prox-E's inference-time hero render).
    """
    if num_views == 1:
        return [70.0]
    return [i * (360.0 / num_views) for i in range(num_views)]


def _render_side(
    mesh_paths: Dict[str, Path],
    sample_ids: List[str],
    out_dir: Path,
    num_views: int,
    image_size: int,
    rotation: Rotation,
    label: str,
    render_norm: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    azims = _azims_for(num_views)
    skipped = 0
    rendered = 0
    failed: List[str] = []
    for sid in tqdm(sample_ids, desc=f"render[{label}]"):
        paths = _render_paths(out_dir, sid, num_views)
        if all(p.exists() for p in paths):
            skipped += 1
            continue
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
        try:
            render_obj_with_blender_sequence(
                str(mesh_paths[sid]),
                azims,
                [str(p) for p in paths],
                res_x=image_size,
                res_y=image_size,
                dist=1.5,
                elev=20.0,
                light_energy=3.0,
                invisible_ground=False,
                shade_smooth=True,
                render_norm=render_norm,
                rotation=tuple(rotation) if rotation is not None else None,
            )
            rendered += 1
        except Exception as e:  # noqa: BLE001 — render failures shouldn't kill the run
            failed.append(sid)
            print(f"  [render:{label}] {sid}: {type(e).__name__}: {e}")
    print(
        f"  [render:{label}] rendered={rendered} skipped={skipped} failed={len(failed)}"
    )
    if failed:
        print(f"  [render:{label}] failed ids (first 5): {failed[:5]}")


def render_meshes(
    *,
    sample_ids: List[str],
    pred_paths: Dict[str, Path],
    gt_paths: Dict[str, Path],
    output_dir: Path,
    num_views: int = 1,
    image_size: int = 512,
    pred_rotation: Rotation = None,
    gt_rotation: Rotation = None,
    render_norm: bool = False,
) -> None:
    """Render pred and gt meshes into ``output_dir/{pred,gt}/...``.

    Layout per side:
        num_views == 1:   {output_dir}/{side}/{sample_id}.png
        num_views >  1:   {output_dir}/{side}/{sample_id}/view_NN.png
    """
    _render_side(
        pred_paths,
        sample_ids,
        output_dir / "pred",
        num_views,
        image_size,
        pred_rotation,
        "pred",
        render_norm,
    )
    if gt_paths:
        _render_side(
            gt_paths,
            sample_ids,
            output_dir / "gt",
            num_views,
            image_size,
            gt_rotation,
            "gt",
            render_norm,
        )
    else:
        print(
            "  [render:gt] no --gt_dir provided; relying on cached renders if any "
            "(see --gt_render_dir)"
        )


# ---------------------------------------------------------------------------
# Mesh -> point cloud
# ---------------------------------------------------------------------------


def _load_trimesh(path: Path) -> trimesh.Trimesh:
    scene = trimesh.load(path, force="mesh", process=False)
    if isinstance(scene, trimesh.Trimesh):
        return scene
    mesh = trimesh.Trimesh()
    for geom in scene.geometry.values():
        mesh = trimesh.util.concatenate([mesh, geom])
    return mesh


def _mesh_to_fixed_pcd(
    path: Path, n_samples: int, voxel_size: float = 0.01
) -> Optional[np.ndarray]:
    """Surface-sample → voxel-downsample. Returns float32 ndarray (M, 3) where M >= n_samples typically."""
    try:
        mesh = _load_trimesh(path)
    except Exception as e:  # noqa: BLE001
        print(f"  [mesh2pcd] {path.name}: load failed: {type(e).__name__}: {e}")
        return None
    raw_points, _ = trimesh.sample.sample_surface(mesh, count=n_samples)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_points)
    down = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(down.points, dtype=np.float32)


def _uniform_subsample(
    points: np.ndarray, n_samples: int, random_seed: int
) -> np.ndarray:
    np.random.seed(random_seed)
    replace = n_samples > len(points)
    idx = np.random.choice(len(points), n_samples, replace=replace)
    return points[idx]


def _mesh_to_pcd_side(
    mesh_paths: Dict[str, Path],
    sample_ids: List[str],
    out_dir: Path,
    num_points: int,
    seed: int,
    label: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    failed: List[str] = []
    for sid in tqdm(sample_ids, desc=f"pcd[{label}]"):
        npz_path = out_dir / f"{sid}.npz"
        if npz_path.exists():
            skipped += 1
            continue
        pts = _mesh_to_fixed_pcd(mesh_paths[sid], n_samples=20480)
        if pts is None or len(pts) == 0:
            failed.append(sid)
            continue
        pts = _uniform_subsample(pts, num_points, seed)
        np.savez_compressed(npz_path, pointcloud=pts[None, ...].astype(np.float32))
        saved += 1
    print(f"  [pcd:{label}] saved={saved} skipped={skipped} failed={len(failed)}")
    if failed:
        print(f"  [pcd:{label}] failed ids (first 5): {failed[:5]}")


def mesh_to_pcd_pairs(
    *,
    sample_ids: List[str],
    pred_paths: Dict[str, Path],
    gt_paths: Dict[str, Path],
    output_dir: Path,
    num_points: int = 2048,
    seed: int = 42,
) -> None:
    """Write ``output_dir/{pred,gt}/{sample_id}.npz`` with key ``pointcloud`` of shape (1, K, 3)."""
    _mesh_to_pcd_side(
        pred_paths, sample_ids, output_dir / "pred", num_points, seed, "pred"
    )
    if gt_paths:
        _mesh_to_pcd_side(
            gt_paths, sample_ids, output_dir / "gt", num_points, seed, "gt"
        )
    else:
        print("  [pcd:gt] no --gt_dir provided; skipped (PFD and l-GD will be skipped)")
