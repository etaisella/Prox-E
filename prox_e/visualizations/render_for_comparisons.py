#!/usr/bin/env python3
"""
Render comparison images across baselines for the test set.

For each row in the test CSV, renders:
  - original_slat.glb  -> original.png   (our pipeline's input reconstruction;
    ``PIPELINE_ORIENTATION`` / ``ORIENTATION_TRANSFORMS`` before Blender view)
  - appearance_edited.glb -> ours.png     (our pipeline's output; same orientation)
  - <baseline>/<assignmentid>.{glb,obj} -> <baseline>.png  (each baseline)

With ``--export-glb``, also writes matching GLBs (same transform stack as the PNGs:
``baseline_orientation_chain`` then uniform scale and ``GLB_ROTATION`` baked in), plus
``original_mesh.glb`` from ``input_mesh/normalized.obj`` or the ShapeNet fallback, with
``ORIGINAL_MESH_ORIENTATION`` + baked ``GLB_ROTATION`` (picked from
``--orientation-sweep-original-mesh``; can differ from Trellis ``PIPELINE_ORIENTATION``).

Then composites them into a single comparison image with labels.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import trimesh
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils import render_obj_with_blender
from scripts.prepare_compound_edit import ORIENTATION_TRANSFORMS

def _env_path(name: str) -> str | None:
    return os.environ.get(name)


# Sentinel: ``gallery_baseline_orientation_chain(..., trellis_post_orient=...)`` not overridden.
_UNSET_TRELLIS_POST = object()

GLB_ROTATION = (-90, 0, 0)
RENDER_KWARGS = dict(rotation=GLB_ROTATION, invisible_ground=True, dist=2.0, light_energy=2.5)

# Pipeline meshes (original_slat.glb, appearance_edited.glb): index into ORIENTATION_TRANSFORMS
# applied before render / GLB export (same as sweep ``*_orient_{idx:02d}_*``).
PIPELINE_ORIENTATION = 2

# Dataset input mesh only (``original_mesh.glb``): index into ORIENTATION_TRANSFORMS before
# baked ``GLB_ROTATION`` (from ``--orientation-sweep-original-mesh``; orient_05).
ORIGINAL_MESH_ORIENTATION = 5

# Second orientation step for gallery baselines (before ``PIPELINE_ORIENTATION`` in the
# apply order; see ``baseline_orientation_chain``). Omitted when None or equal to
# pipeline index. ``TRELLIS`` is handled separately via ``TRELLIS_GALLERY_EXTRA_ORIENTATION``.
BASELINE_ORIENTATIONS = {
    "VoxHammer": 6,
    "EditP23": 10,
}

# Only the ``TRELLIS`` folder under ``--baselines-base``: extra
# ``ORIENTATION_TRANSFORMS`` index applied after ``PIPELINE_ORIENTATION`` (trimesh
# order: pipeline first, then this). Picked from ``--orientation-sweep-assignment``
# post-pipeline sweep; ``11`` = ``rot_180_z``. Use ``None`` to disable the extra step.
TRELLIS_GALLERY_EXTRA_ORIENTATION: int | None = 11

# Per-baseline uniform scale factors applied before rendering.
BASELINE_SCALES = {
    "Spice-E": 0.325,
    "EditP23": 0.4,
    "VoxHammer": 0.65,
}


def baseline_orientation_chain(
    baseline_idx: int | None,
    *,
    pipeline_only: bool = False,
) -> tuple[int, ...]:
    """
    Indices into ORIENTATION_TRANSFORMS for gallery baselines, applied in **this**
    order via ``apply_transform`` (so combined rotation is
    ``R_pipe @ R_bl`` on column vertices when both apply).

    Per-baseline indices were chosen on **raw** gallery exports; ``PIPELINE_ORIENTATION``
    matches Trellis ``original_slat`` / ``appearance_edited`` into the comparison
    camera. Apply baseline alignment **first**, then the shared pipeline frame.
    When the baseline index equals the pipeline index, only that matrix is applied once.
    The ``TRELLIS`` gallery baseline is handled in ``gallery_baseline_orientation_chain``.

    If ``pipeline_only`` is True (CLI ``--baselines-pipeline-orient-only``), every
    baseline uses only ``PIPELINE_ORIENTATION`` so you can ignore legacy
    ``BASELINE_ORIENTATIONS`` when debugging alignment.
    """
    if pipeline_only:
        return (PIPELINE_ORIENTATION,)
    if baseline_idx is None or baseline_idx == PIPELINE_ORIENTATION:
        return (PIPELINE_ORIENTATION,)
    return (baseline_idx, PIPELINE_ORIENTATION)


def gallery_baseline_orientation_chain(
    baseline_name: str,
    baseline_idx: int | None,
    *,
    pipeline_only: bool = False,
    trellis_post_orient: int | None | object = _UNSET_TRELLIS_POST,
) -> tuple[int, ...]:
    """
    Full orientation tuple for one gallery baseline (folder name under
    ``--baselines-base``), including the ``TRELLIS`` post-pipeline tweak.

    ``trellis_post_orient``: use module ``TRELLIS_GALLERY_EXTRA_ORIENTATION`` when unset
    (default sentinel); an ``int`` overrides; ``None`` means no extra step after pipeline.
    """
    if pipeline_only:
        return (PIPELINE_ORIENTATION,)
    if baseline_name == "TRELLIS":
        if trellis_post_orient is _UNSET_TRELLIS_POST:
            extra = TRELLIS_GALLERY_EXTRA_ORIENTATION
        else:
            extra = trellis_post_orient  # int or None (explicitly no extra)
        if extra is None or extra == PIPELINE_ORIENTATION:
            return (PIPELINE_ORIENTATION,)
        return (PIPELINE_ORIENTATION, extra)
    return baseline_orientation_chain(baseline_idx, pipeline_only=False)


def discover_baselines(baselines_base: Path) -> list:
    """Return sorted list of baseline folder names that exist under baselines_base."""
    if not baselines_base.is_dir():
        return []
    return sorted(
        d.name for d in baselines_base.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def find_baseline_mesh(baseline_dir: Path, assignment_id: str) -> Path | None:
    """Find a mesh file for the given assignment_id in the baseline dir (.glb or .obj)."""
    for ext in (".glb", ".obj"):
        candidate = baseline_dir / f"{assignment_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def find_original_input_mesh(result_folder: Path) -> Path | None:
    """
    Original ShapeNet / normalized input used by the pipeline (same resolution order
    as ``visualizations_for_slides.render_for_teaser``).
    """
    input_mesh_folder = result_folder / "input_mesh"
    if input_mesh_folder.is_dir():
        p = input_mesh_folder / "normalized.obj"
        if p.is_file():
            return p
    shapenet = result_folder / "from_shapenet" / "models" / "model_normalized.obj"
    if shapenet.is_file():
        return shapenet
    return None


def _blender_euler_xyz_deg_to_matrix_4(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Match ``utils.render_obj_with_blender`` vertex rotation: Rx @ Ry @ Rz (degrees),
    as a 4x4 homogeneous matrix (column vectors).
    """
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    r_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    r3 = r_x @ r_y @ r_z
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r3
    return out


def _apply_orientation_chain_to_mesh(mesh: trimesh.Trimesh, orientation_indices: Sequence[int]) -> None:
    for idx in orientation_indices:
        _, mat = ORIENTATION_TRANSFORMS[idx]
        mesh.apply_transform(mat)


def export_comparison_glb(
    mesh_path: Path,
    output_glb: Path,
    skip_existing: bool,
    orientation_indices: Sequence[int] | None = None,
    scale: float = None,
) -> bool:
    """
    Write a GLB whose vertices match the Blender render (orientation chain, scale, then
    ``GLB_ROTATION`` applied to vertices, same order as ``render_single``).
    """
    if skip_existing and output_glb.exists():
        return True
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    if orientation_indices:
        _apply_orientation_chain_to_mesh(mesh, orientation_indices)
    if scale is not None:
        mesh.apply_scale(scale)
    mesh.apply_transform(_blender_euler_xyz_deg_to_matrix_4(*GLB_ROTATION))
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_glb), file_type="glb")
    return output_glb.exists()


def _apply_pretransform(mesh_path: Path, tmp_dir: Path,
                        transform_matrix: np.ndarray = None,
                        scale: float = None) -> Path:
    """Apply a single 4x4 transform and/or uniform scale to a mesh. Returns a temp file path."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    if transform_matrix is not None:
        mesh.apply_transform(transform_matrix)
    if scale is not None:
        mesh.apply_scale(scale)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = mesh_path.suffix
    tmp_path = tmp_dir / f"_pretransformed{suffix}"
    mesh.export(str(tmp_path))
    return tmp_path


def _apply_pretransform_orientation_chain(
    mesh_path: Path, tmp_dir: Path,
    orientation_indices: Sequence[int],
    scale: float = None,
) -> Path:
    """Apply one or more ORIENTATION_TRANSFORMS indices in order, then scale. Returns temp path."""
    mesh = trimesh.load(str(mesh_path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    _apply_orientation_chain_to_mesh(mesh, orientation_indices)
    if scale is not None:
        mesh.apply_scale(scale)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    suffix = mesh_path.suffix
    tmp_path = tmp_dir / f"_pretransformed{suffix}"
    mesh.export(str(tmp_path))
    return tmp_path


def render_single(
    mesh_path: Path,
    output_png: Path,
    skip_existing: bool,
    orientation_indices: Sequence[int] | None = None,
    scale: float = None,
):
    """
    Render a mesh to PNG using the shared Blender renderer.

    Args:
        orientation_indices: ORIENTATION_TRANSFORMS indices applied in order before Blender.
        scale: If set, uniform scale factor applied after those transforms.
    """
    if skip_existing and output_png.exists():
        return True
    output_png.parent.mkdir(parents=True, exist_ok=True)

    needs_pretransform = bool(orientation_indices) or scale is not None
    actual_path = mesh_path

    if needs_pretransform:
        if orientation_indices:
            actual_path = _apply_pretransform_orientation_chain(
                mesh_path, output_png.parent / "_tmp",
                orientation_indices=orientation_indices, scale=scale,
            )
        else:
            actual_path = _apply_pretransform(
                mesh_path, output_png.parent / "_tmp",
                transform_matrix=None, scale=scale,
            )

    render_obj_with_blender(str(actual_path), str(output_png), **RENDER_KWARGS)

    if actual_path != mesh_path and actual_path.exists():
        actual_path.unlink()

    return output_png.exists()


def export_orientation_sweep_glbs(
    mesh_path: Path,
    out_dir: Path,
    label: str,
    uniform_scale: float | None = None,
    skip_existing: bool = False,
):
    """
    For each ``ORIENTATION_TRANSFORMS`` index, write a GLB with the same transform
    stack as comparison exports: optional orientation + optional uniform scale, then
    ``GLB_ROTATION`` baked into vertices (matches ``export_comparison_glb``).

    Also writes ``{label}_orientation_sweep_manifest.txt`` listing index, name, and file.

    ``label`` prefixes output filenames (e.g. baseline folder name or ``original_slat``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines = [
        f"# orientation sweep meshes: {label}",
        f"# source: {mesh_path}",
        "# idx\ttransform_name\tglb_filename",
    ]
    n_ok = 0

    for idx in sorted(ORIENTATION_TRANSFORMS.keys()):
        name, _mat = ORIENTATION_TRANSFORMS[idx]
        safe_name = name.replace("/", "_")
        glb_path = out_dir / f"{label}_orient_{idx:02d}_{safe_name}.glb"
        ok = export_comparison_glb(
            mesh_path,
            glb_path,
            skip_existing=skip_existing,
            orientation_indices=(idx,),
            scale=uniform_scale,
        )
        if ok:
            n_ok += 1
            manifest_lines.append(f"{idx}\t{name}\t{glb_path.name}")

    manifest_path = out_dir / f"{label}_orientation_sweep_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"  Wrote {n_ok} GLB(s) under {out_dir}")
    print(f"  Manifest: {manifest_path}")
    if n_ok == 0:
        print("  Warning: no GLB files were written.")


def export_orientation_sweep_glbs_post_pipeline(
    mesh_path: Path,
    out_dir: Path,
    label: str,
    uniform_scale: float | None = None,
    skip_existing: bool = False,
):
    """
    For each ``ORIENTATION_TRANSFORMS`` index ``idx``, write a GLB using the same
    composition as gallery baselines after ``PIPELINE_ORIENTATION``: vertex chain is
    ``(PIPELINE_ORIENTATION, idx)``, deduped to ``(PIPELINE_ORIENTATION,)`` when
    ``idx`` equals the pipeline index, then optional scale, then ``GLB_ROTATION``.

    Output names use the ``postpipe_orient`` prefix so they do not clash with raw
    ``export_orientation_sweep_glbs`` sweeps.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_lines = [
        f"# post-pipeline orientation sweep: {label}",
        f"# source: {mesh_path}",
        f"# pipeline index first: {PIPELINE_ORIENTATION}",
        "# idx\tsecond_transform_name\tglb_filename",
    ]
    n_ok = 0

    for idx in sorted(ORIENTATION_TRANSFORMS.keys()):
        name, _mat = ORIENTATION_TRANSFORMS[idx]
        safe_name = name.replace("/", "_")
        if idx == PIPELINE_ORIENTATION:
            chain = (PIPELINE_ORIENTATION,)
        else:
            chain = (PIPELINE_ORIENTATION, idx)
        glb_path = out_dir / f"{label}_postpipe_orient_{idx:02d}_{safe_name}.glb"
        ok = export_comparison_glb(
            mesh_path,
            glb_path,
            skip_existing=skip_existing,
            orientation_indices=chain,
            scale=uniform_scale,
        )
        if ok:
            n_ok += 1
            manifest_lines.append(f"{idx}\t{name}\t{glb_path.name}")

    manifest_path = out_dir / f"{label}_postpipe_orientation_sweep_manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"  Wrote {n_ok} GLB(s) under {out_dir}")
    print(f"  Manifest: {manifest_path}")
    if n_ok == 0:
        print("  Warning: no GLB files were written.")


def make_comparison_image(
    image_dict: dict,
    utterance: str,
    output_path: Path,
):
    """
    Create a side-by-side comparison figure.

    Args:
        image_dict: OrderedDict-like mapping label -> png path.
                    'original' is placed first, 'ours' last; others in between.
        utterance: Text shown as the figure title.
        output_path: Where to save the combined PNG.
    """
    ordered_labels = []
    if "original" in image_dict:
        ordered_labels.append("original")
    for label in image_dict:
        if label not in ("original", "ours"):
            ordered_labels.append(label)
    if "ours" in image_dict:
        ordered_labels.append("ours")

    n = len(ordered_labels)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4))
    if n == 1:
        axes = [axes]

    max_chars_per_line = 60
    title_lines = [utterance[i:i + max_chars_per_line] for i in range(0, len(utterance), max_chars_per_line)]
    fig.suptitle("\n".join(title_lines), fontsize=12, fontweight="bold", y=0.98)

    for ax, label in zip(axes, ordered_labels):
        img = mpimg.imread(image_dict[label])
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Render comparison images across baselines")
    parser.add_argument(
        "--csv",
        type=str,
        default=_env_path("PROXE_COMPARISON_CSV"),
        help="Path to test set CSV (default: PROXE_COMPARISON_CSV env)",
    )
    parser.add_argument(
        "--results-base",
        type=str,
        default=_env_path("PROXE_RESULTS_BASE"),
        help="Base path for pipeline results (default: PROXE_RESULTS_BASE env)",
    )
    parser.add_argument(
        "--baselines-base",
        type=str,
        default=_env_path("PROXE_BASELINES_BASE"),
        help="Base path for baseline subfolders (default: PROXE_BASELINES_BASE env)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=_env_path("PROXE_COMPARISON_OUTPUT"),
        help="Output directory for comparisons (default: PROXE_COMPARISON_OUTPUT env)",
    )
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip rendering if the output PNG already exists")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows (for debugging)")
    parser.add_argument("--orientation-sweep", type=str, default=None, metavar="BASELINE",
                        help="Export 16 oriented GLBs for the first CSV row that has a mesh "
                             "under BASELINE/ (e.g. VoxHammer) and exit. Combine with --limit 1.")
    parser.add_argument(
        "--orientation-sweep-assignment",
        type=str,
        default=None,
        metavar="ASSIGNMENT_ID",
        help="With --orientation-sweep BASELINE: sweep this assignment only. Writes 16 GLBs "
             "under --output/<ASSIGNMENT_ID>/orientation_sweep/ using post-pipeline composition "
             "(PIPELINE_ORIENTATION then each candidate index; see "
             "export_orientation_sweep_glbs_post_pipeline).",
    )
    parser.add_argument(
        "--orientation-sweep-original",
        type=str,
        default=None,
        metavar="ASSIGNMENT_ID",
        help="Export 16 oriented GLBs from pipeline original_slat.glb for this assignment "
             "(looks up category from --csv). Writes under --output/<id>/orientation_sweep/ "
             "and exits. Incompatible with --orientation-sweep.",
    )
    parser.add_argument(
        "--orientation-sweep-original-mesh",
        type=str,
        default=None,
        metavar="ASSIGNMENT_ID",
        help="Export 16 oriented GLBs from the dataset input mesh (input_mesh/normalized.obj "
             "or from_shapenet fallback) for this assignment. Writes "
             "original_mesh_orient_* under --output/<id>/orientation_sweep/ and exits. "
             "Same per-index convention as --orientation-sweep-original (single "
             "ORIENTATION_TRANSFORMS index + baked GLB_ROTATION per file).",
    )
    parser.add_argument("--baseline-orientation", type=str, nargs=2, action="append",
                        default=[], metavar=("BASELINE", "IDX"),
                        help="Set orientation for a baseline, e.g. --baseline-orientation VoxHammer 6. "
                             "Can be specified multiple times.")
    parser.add_argument("--export-glb", action="store_true",
                        help="Also write oriented GLBs beside each PNG (original.glb, ours.glb, "
                             "<Baseline>.glb), plus original_mesh.glb (input mesh; "
                             "ORIGINAL_MESH_ORIENTATION + GLB_ROTATION).")
    parser.add_argument("--assignment-ids", type=str, nargs="+", default=None, metavar="ID",
                        help="Process only these assignmentid values (must appear in the CSV).")
    parser.add_argument(
        "--baselines-pipeline-orient-only",
        action="store_true",
        help="For gallery baselines, apply only PIPELINE_ORIENTATION (ignore "
             "BASELINE_ORIENTATIONS). Useful if legacy per-baseline indices no longer "
             "compose correctly with the Trellis frame fix.",
    )
    parser.add_argument(
        "--trellis-gallery-post-orient",
        type=int,
        default=None,
        metavar="IDX",
        help="Override TRELLIS_GALLERY_EXTRA_ORIENTATION: ORIENTATION_TRANSFORMS index "
             "applied after PIPELINE_ORIENTATION for the TRELLIS gallery folder only. "
             "Use -1 to force no extra step (same as setting the constant to None).",
    )
    args = parser.parse_args()

    for name, value in (
        ("--csv / PROXE_COMPARISON_CSV", args.csv),
        ("--results-base / PROXE_RESULTS_BASE", args.results_base),
        ("--baselines-base / PROXE_BASELINES_BASE", args.baselines_base),
        ("--output / PROXE_COMPARISON_OUTPUT", args.output),
    ):
        if not value:
            parser.error(f"{name} is required.")

    for bl_name, idx_str in args.baseline_orientation:
        BASELINE_ORIENTATIONS[bl_name] = int(idx_str)

    if args.trellis_gallery_post_orient is None:
        trellis_post_kw: int | None | object = _UNSET_TRELLIS_POST
    elif args.trellis_gallery_post_orient == -1:
        trellis_post_kw = None  # no extra orientation after pipeline
    else:
        trellis_post_kw = args.trellis_gallery_post_orient

    if args.orientation_sweep and args.orientation_sweep_original:
        parser.error("Use only one of --orientation-sweep and --orientation-sweep-original.")
    if args.orientation_sweep_original and args.orientation_sweep_assignment:
        parser.error("Use only one of --orientation-sweep-original and --orientation-sweep-assignment.")
    if args.orientation_sweep_assignment and not args.orientation_sweep:
        parser.error("--orientation-sweep-assignment requires --orientation-sweep BASELINE.")
    _sweep_modes = sum(
        1
        for x in (
            args.orientation_sweep,
            args.orientation_sweep_original,
            args.orientation_sweep_original_mesh,
        )
        if x
    )
    if _sweep_modes > 1:
        parser.error(
            "Use only one sweep mode among --orientation-sweep, "
            "--orientation-sweep-original, and --orientation-sweep-original-mesh."
        )

    results_base = Path(args.results_base)
    baselines_base = Path(args.baselines_base)
    output_base = Path(args.output)

    df = pd.read_csv(args.csv)
    if args.assignment_ids is not None:
        want = set(args.assignment_ids)
        df = df[df["assignmentid"].isin(want)]
        missing = want - set(df["assignmentid"].unique())
        if missing:
            print(f"Warning: these IDs are not in the CSV and will be skipped: {sorted(missing)}")
        print(f"Filtered to {len(df)} row(s) matching --assignment-ids")
    baselines = discover_baselines(baselines_base)
    print(f"Found {len(baselines)} baselines: {baselines}")
    print(f"Processing {len(df)} rows from {args.csv}")
    if args.export_glb:
        print("GLB export enabled (oriented meshes match comparison PNG pipeline).")

    if args.limit:
        df = df.head(args.limit)
        print(f"  (limited to first {args.limit} rows)")

    # --- Orientation sweep: pipeline original_slat.glb ---
    if args.orientation_sweep_original:
        aid = args.orientation_sweep_original.strip()
        df_lookup = pd.read_csv(args.csv)
        matches = df_lookup[df_lookup["assignmentid"] == aid]
        if matches.empty:
            print(f"No CSV row for assignmentid={aid!r} in {args.csv}")
            return
        row0 = matches.iloc[0]
        category = row0["source_object_class"].strip().lower()
        original_glb = results_base / category / aid / "original_slat.glb"
        if not original_glb.is_file():
            print(f"Missing pipeline mesh: {original_glb}")
            return
        sweep_dir = output_base / aid / "orientation_sweep"
        print(f"\n=== Orientation sweep (original_slat.glb) ===")
        print(f"  assignment: {aid}  category: {category}")
        print(f"  mesh: {original_glb}")
        print(f"  output: {sweep_dir}")
        export_orientation_sweep_glbs(
            original_glb, sweep_dir, "original_slat", uniform_scale=None,
            skip_existing=args.skip_existing,
        )
        return

    # --- Orientation sweep: dataset input mesh (normalized.obj / ShapeNet) ---
    if args.orientation_sweep_original_mesh:
        aid = args.orientation_sweep_original_mesh.strip()
        df_lookup = pd.read_csv(args.csv)
        matches = df_lookup[df_lookup["assignmentid"] == aid]
        if matches.empty:
            print(f"No CSV row for assignmentid={aid!r} in {args.csv}")
            return
        row0 = matches.iloc[0]
        category = row0["source_object_class"].strip().lower()
        result_folder = results_base / category / aid
        input_mesh = find_original_input_mesh(result_folder)
        if input_mesh is None:
            print(f"Missing input mesh under {result_folder} (input_mesh/normalized.obj "
                  f"or from_shapenet/models/model_normalized.obj)")
            return
        sweep_dir = output_base / aid / "orientation_sweep"
        print(f"\n=== Orientation sweep (original input mesh) ===")
        print(f"  assignment: {aid}  category: {category}")
        print(f"  mesh: {input_mesh}")
        print(f"  output: {sweep_dir}")
        export_orientation_sweep_glbs(
            input_mesh, sweep_dir, "original_mesh", uniform_scale=None,
            skip_existing=args.skip_existing,
        )
        return

    # --- Orientation sweep mode (baseline gallery) ---
    if args.orientation_sweep:
        sweep_bl = args.orientation_sweep
        print(f"\n=== Orientation sweep for '{sweep_bl}' ===")
        if args.orientation_sweep_assignment:
            aid = args.orientation_sweep_assignment.strip()
            mesh = find_baseline_mesh(baselines_base / sweep_bl, aid)
            if mesh is None:
                print(f"No mesh for baseline {sweep_bl!r} assignment {aid!r} under {baselines_base / sweep_bl}")
                return
            sweep_dir = output_base / aid / "orientation_sweep"
            print(f"  assignment: {aid}")
            print(f"  mesh: {mesh}")
            print(f"  output: {sweep_dir}")
            print("  mode: post-pipeline (PIPELINE_ORIENTATION then each index; matches comparison GLBs)")
            bl_scale = BASELINE_SCALES.get(sweep_bl)
            export_orientation_sweep_glbs_post_pipeline(
                mesh, sweep_dir, sweep_bl, uniform_scale=bl_scale,
                skip_existing=args.skip_existing,
            )
            return

        found = False
        for _, row in df.iterrows():
            assignment_id = row["assignmentid"]
            mesh = find_baseline_mesh(baselines_base / sweep_bl, assignment_id)
            if mesh is None:
                continue
            found = True
            sweep_dir = output_base / assignment_id / "orientation_sweep"
            print(f"Using sample: {assignment_id}  mesh: {mesh}")
            bl_scale = BASELINE_SCALES.get(sweep_bl)
            export_orientation_sweep_glbs(
                mesh, sweep_dir, sweep_bl, uniform_scale=bl_scale,
                skip_existing=args.skip_existing,
            )
            break
        if not found:
            print(f"Could not find any mesh for baseline '{sweep_bl}'")
        return

    # --- Normal rendering ---
    stats = {"rendered": 0, "skipped_no_result": 0, "skipped_incomplete": 0}

    for _, row in df.iterrows():
        assignment_id = row["assignmentid"]
        category = row["source_object_class"].strip().lower()
        utterance = str(row["utterance"])

        result_folder = results_base / category / assignment_id
        if not result_folder.exists():
            stats["skipped_no_result"] += 1
            continue

        original_glb = result_folder / "original_slat.glb"
        edited_glb = result_folder / "appearance_edited.glb"
        if not original_glb.exists() or not edited_glb.exists():
            stats["skipped_incomplete"] += 1
            continue

        out_dir = output_base / assignment_id
        out_dir.mkdir(parents=True, exist_ok=True)

        image_dict = {}

        # --- Render our results ---
        original_png = out_dir / "original.png"
        ours_png = out_dir / "ours.png"

        _pipe_orient = (PIPELINE_ORIENTATION,)
        render_single(
            original_glb, original_png, args.skip_existing,
            orientation_indices=_pipe_orient, scale=None,
        )
        render_single(
            edited_glb, ours_png, args.skip_existing,
            orientation_indices=_pipe_orient, scale=None,
        )

        if args.export_glb:
            export_comparison_glb(
                original_glb, out_dir / "original.glb", args.skip_existing,
                orientation_indices=_pipe_orient, scale=None,
            )
            export_comparison_glb(
                edited_glb, out_dir / "ours.glb", args.skip_existing,
                orientation_indices=_pipe_orient, scale=None,
            )
            input_mesh_path = find_original_input_mesh(result_folder)
            if input_mesh_path is not None:
                export_comparison_glb(
                    input_mesh_path,
                    out_dir / "original_mesh.glb",
                    args.skip_existing,
                    orientation_indices=(ORIGINAL_MESH_ORIENTATION,),
                    scale=None,
                )

        if original_png.exists():
            image_dict["original"] = str(original_png)
        if ours_png.exists():
            image_dict["ours"] = str(ours_png)

        # --- Render baselines ---
        for baseline_name in baselines:
            baseline_mesh = find_baseline_mesh(baselines_base / baseline_name, assignment_id)
            if baseline_mesh is None:
                continue
            baseline_png = out_dir / f"{baseline_name}.png"
            bl_orient = BASELINE_ORIENTATIONS.get(baseline_name)
            bl_scale = BASELINE_SCALES.get(baseline_name)
            bl_chain = gallery_baseline_orientation_chain(
                baseline_name,
                bl_orient,
                pipeline_only=args.baselines_pipeline_orient_only,
                trellis_post_orient=trellis_post_kw,
            )
            render_single(
                baseline_mesh, baseline_png, args.skip_existing,
                orientation_indices=bl_chain, scale=bl_scale,
            )
            if args.export_glb:
                export_comparison_glb(
                    baseline_mesh, out_dir / f"{baseline_name}.glb", args.skip_existing,
                    orientation_indices=bl_chain, scale=bl_scale,
                )
            if baseline_png.exists():
                image_dict[baseline_name] = str(baseline_png)

        # --- Combined comparison ---
        comparison_png = out_dir / "comparison.png"
        if image_dict:
            make_comparison_image(image_dict, utterance, comparison_png)

        stats["rendered"] += 1
        print(f"[{stats['rendered']:4d}] {assignment_id}  ({category})  "
              f"images={len(image_dict)}")

    print(f"\nDone. rendered={stats['rendered']}  "
          f"skipped_no_result={stats['skipped_no_result']}  "
          f"skipped_incomplete={stats['skipped_incomplete']}")


if __name__ == "__main__":
    main()
