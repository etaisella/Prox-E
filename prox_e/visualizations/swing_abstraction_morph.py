"""
Interpolate between original and edited superquadric abstractions for swing animations.

Classification matches ``compare_abstractions`` / ``inpaint_data_preparation`` (curvature-only
changes → delete+add). Colors match ``glb_orientation_sweep.build_edited_abstraction_categorized_mesh``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from typing import Any, Dict, List, Tuple

import numpy as np

from inpaint_data_preparation import compare_abstractions
from abstraction import load_abstraction, mesh_from_abstraction
from scripts.prepare_compound_edit import ORIENTATION_TRANSFORMS

# Must match ``TEASER_MESH_ORIENTATIONS["original_abstraction" / "edited_abstraction_categorized"]``
# in visualizations_for_slides (baked into on-disk teaser GLBs before Blender view rotation).
TEASER_ABSTRACTION_BAKE_ORIENT_IDX = 1


# Same as visualizations/glb_orientation_sweep.py (edited categorized teaser)
COLOR_GRAY = [180, 180, 180]
COLOR_WHITE = [255, 255, 255]
COLOR_BLUE = [40, 90, 180]
COLOR_PURPLE = [150, 50, 170]


def resolve_teaser_result_folder(mesh_path: Path) -> Path:
    """
    JSONs live in the parent of ``teaser_renders`` (e.g. sheep_0/), when mesh is under
    ``.../teaser_renders/original_abstraction.glb``.
    """
    p = mesh_path.resolve().parent
    if p.name != "teaser_renders":
        raise ValueError(
            f"--transform expects mesh under a teaser_renders/ folder; got parent {p}"
        )
    return p.parent


def find_edited_abstraction_json(result_folder: Path) -> Path:
    for name in (
        "edited_abstraction.json",
        "edited_abstraction_final.json",
        "edited_abstraction_iter1.json",
    ):
        c = result_folder / name
        if c.is_file():
            return c
    raise FileNotFoundError(
        f"No edited_abstraction*.json found under {result_folder} "
        "(tried edited_abstraction.json, edited_abstraction_final.json, edited_abstraction_iter1.json)"
    )


def _lerp(a: float, b: float, t: float) -> float:
    return float(a) * (1.0 - t) + float(b) * t


def _lerp_vec(va: List[float], vb: List[float], t: float) -> List[float]:
    a = np.asarray(va, dtype=np.float64)
    b = np.asarray(vb, dtype=np.float64)
    return ((1.0 - t) * a + t * b).tolist()


def _lerp_rotation_matrix(R0: List[List[float]], R1: List[List[float]], t: float) -> List[List[float]]:
    """Linear blend + orthogonalize (QR) so the matrix stays a proper rotation."""
    m0 = np.asarray(R0, dtype=np.float64)
    m1 = np.asarray(R1, dtype=np.float64)
    m = (1.0 - t) * m0 + t * m1
    q, _ = np.linalg.qr(m)
    if np.linalg.det(q) < 0:
        q[:, 2] *= -1.0
    return q.tolist()


def _sq_copy(sq: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(sq)


def morph_abstraction_at_t(
    original: List[dict],
    edited: List[dict],
    categories: Dict[str, List[dict]],
    t: float,
) -> List[dict]:
    """
    Build one abstraction list for mesh_from_abstraction at interpolation parameter t in [0, 1].

    - Unchanged: gray, constant geometry (original params).
    - changed_original / changed_edited (exponents matched): lerp pose + exponents; color white→blue.
    - deleted (truly_deleted): original SQ with scale lerped toward ~0; gray.
    - added (truly_added): edited SQ with scale from 0; color white→purple.
    """
    t = max(0.0, min(1.0, float(t)))
    out: List[dict] = []

    unchanged_by_idx = {sq["index"]: sq for sq in categories["unchanged"]}

    changed_o = {sq["index"]: sq for sq in categories["changed_original"]}
    changed_e = {sq["index"]: sq for sq in categories["changed_edited"]}

    deleted_by_idx = {sq["index"]: sq for sq in categories["deleted"]}
    added_by_idx = {sq["index"]: sq for sq in categories["added"]}

    # Unchanged
    for idx, sq in sorted(unchanged_by_idx.items(), key=lambda x: x[0]):
        c = _sq_copy(sq)
        c["color"] = list(COLOR_GRAY)
        out.append(c)

    # Matched edits (non-curvature): lerp geometry, white → blue
    for idx in sorted(set(changed_o.keys()) & set(changed_e.keys())):
        o = changed_o[idx]
        e = changed_e[idx]
        c = _sq_copy(o)
        c["scale"] = _lerp_vec(o["scale"], e["scale"], t)
        c["translation"] = _lerp_vec(o["translation"], e["translation"], t)
        c["rotation"] = _lerp_rotation_matrix(o["rotation"], e["rotation"], t)
        c["exponents"] = _lerp_vec(o["exponents"], e["exponents"], t)
        c["color"] = [
            _lerp(COLOR_WHITE[i], COLOR_BLUE[i], t) for i in range(3)
        ]
        out.append(c)

    # Removed SQs: shrink scale toward zero
    for idx, sq in sorted(deleted_by_idx.items(), key=lambda x: x[0]):
        c = _sq_copy(sq)
        s = np.asarray(sq["scale"], dtype=np.float64) * (1.0 - t)
        min_s = 1e-4
        c["scale"] = np.maximum(s, min_s).tolist()
        c["color"] = list(COLOR_GRAY)
        out.append(c)

    # New SQs: grow from ~0, white → purple
    for idx, sq in sorted(added_by_idx.items(), key=lambda x: x[0]):
        c = _sq_copy(sq)
        target = np.asarray(sq["scale"], dtype=np.float64)
        c["scale"] = (target * t + 1e-4 * (1.0 - t)).tolist()
        c["translation"] = _sq_copy(sq)["translation"]
        c["rotation"] = _sq_copy(sq)["rotation"]
        c["exponents"] = _sq_copy(sq)["exponents"]
        c["color"] = [
            _lerp(COLOR_WHITE[i], COLOR_PURPLE[i], t) for i in range(3)
        ]
        out.append(c)

    # Stable order by index for reproducibility
    out.sort(key=lambda s: s.get("index", 0))
    return out


def apply_teaser_abstraction_bake_orientation(mesh):
    """Same as ``visualizations_for_slides._apply_teaser_orientation_mesh`` for abstraction GLBs."""
    if mesh is None:
        return mesh
    _, mat = ORIENTATION_TRANSFORMS[TEASER_ABSTRACTION_BAKE_ORIENT_IDX]
    out = mesh.copy()
    out.apply_transform(mat)
    return out


def export_morph_glb(
    original_json: Path,
    edited_json: Path,
    t: float,
    out_glb: Path,
    *,
    resolution: int = 30,
) -> None:
    """Write a single morphed abstraction mesh as GLB (same bake orientation as teaser GLBs)."""
    original = load_abstraction(str(original_json))
    edited = load_abstraction(str(edited_json))
    categories = compare_abstractions(original, edited)
    morphed = morph_abstraction_at_t(original, edited, categories, t)
    mesh = mesh_from_abstraction(morphed, resolution=resolution)
    if mesh is None:
        raise RuntimeError("mesh_from_abstraction returned None for morphed abstraction")
    mesh = apply_teaser_abstraction_bake_orientation(mesh)
    out_glb.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out_glb), file_type="glb")


def build_swing_mesh_paths(
    *,
    total_frames: int,
    start_transform: int,
    num_transform: int,
    original_glb: Path,
    edited_glb: Path,
    original_json: Path,
    edited_json: Path,
    temp_dir: Path,
    resolution: int,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Build per-frame GLB paths. Returns (mesh_paths, segments) where each segment is
    (start_idx, end_idx_exclusive) with the same mesh path for potential batching.

    Precomputes ``num_transform`` morphed GLBs in ``temp_dir`` named morph_000.glb, ...
    """
    mesh_paths: List[str] = [""] * total_frames
    end_t = min(total_frames, start_transform + num_transform)
    t_den = max(num_transform - 1, 1)

    for i in range(total_frames):
        if i < start_transform:
            mesh_paths[i] = str(original_glb.resolve())
        elif i < end_t:
            k = i - start_transform
            t = k / t_den if num_transform > 1 else 1.0
            morph_path = temp_dir / f"morph_{k:04d}.glb"
            if not morph_path.is_file():
                export_morph_glb(
                    original_json,
                    edited_json,
                    t,
                    morph_path,
                    resolution=resolution,
                )
            mesh_paths[i] = str(morph_path.resolve())
        else:
            mesh_paths[i] = str(edited_glb.resolve())

    # Segments of consecutive identical paths (for logging / future batching)
    segments: List[Tuple[int, int]] = []
    if total_frames == 0:
        return mesh_paths, segments
    seg_start = 0
    for i in range(1, total_frames):
        if mesh_paths[i] != mesh_paths[i - 1]:
            segments.append((seg_start, i))
            seg_start = i
    segments.append((seg_start, total_frames))

    return mesh_paths, segments
