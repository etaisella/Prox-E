"""Unified evaluation for Prox-E.

Computes metrics organized into 3 groups against a method's mesh outputs:

  - identity_preservation: l-GD, LPIPS, DINO-I
  - 3d_quality:            PFD, FID
  - edit_fidelity:         CLIP-Sim, CLIP-Dir, VQA (placeholder)

End-to-end pipeline:
  1. ``prepare_inputs`` — discover sample ids and resolve inputs.
  2. Render input and pred meshes (Blender; input side skipped if no input meshes).
  3. Convert meshes to point clouds (trimesh + open3d; input side skipped likewise).
  4. Compute metric groups.
  5. Aggregate into results.json + per_sample.csv.

Inputs (only --pred_dir and --output_dir are required):
  --pred_dir            flat folder of predicted meshes ({sample_id}.{glb|obj|ply})
  --input_dir           optional flat folder of source meshes; required for PFD
                        and l-GD. Other input-dependent metrics (LPIPS, DINO-I, FID,
                        CLIP-Dir) can also be driven from cached renders via
                        --input_render_dir.
  --input_render_dir    optional folder of pre-existing input render PNGs
                        ({sample_id}.png); symlinked into <output_dir>/renders/gt/
                        before rendering runs, so image-based input metrics work
                        without needing input meshes.
  --instructions_json   optional {"sample_id": "..."} or
                        {"sample_id": {"instruction", "obj_class", "part_keyword"}}.
                        If omitted, a placeholder JSON ("edit the object" for
                        every sample) is written to <output_dir>/instructions.json
                        and CLIP-Dir / l-GD become meaningless.

Stage outputs are cached under --output_dir; re-runs skip preprocessing.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

METRIC_GROUPS = ("identity", "quality", "fidelity")
MESH_EXTS = (".glb", ".obj", ".ply")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Unified Prox-E evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pred_dir",
        type=Path,
        default=None,
        help="Flat folder of predicted meshes: {sample_id}.{glb|obj|ply}. "
        "Optional when --pred_render_dir is given (image-only fidelity eval).",
    )
    p.add_argument(
        "--pred_render_dir",
        type=Path,
        default=None,
        help="Optional folder of pre-existing pred render PNGs "
        "({sample_id}.png); symlinked into <output_dir>/renders/pred/ so "
        "image-only metrics (CLIP-Sim, CLIP-Dir, VQA) work without pred meshes.",
    )
    p.add_argument(
        "--input_dir",
        type=Path,
        default=None,
        help="Flat folder of source meshes:   {sample_id}.{glb|obj|ply}. "
        "Required for PFD and l-GD; LPIPS/DINO-I/FID/CLIP-Dir can use --input_render_dir instead.",
    )
    p.add_argument(
        "--input_render_dir",
        type=Path,
        default=None,
        help="Optional folder of pre-existing input render PNGs "
        "({sample_id}.png); symlinked into <output_dir>/renders/gt/ so "
        "input-image metrics work without input meshes.",
    )
    p.add_argument(
        "--input_pcd_dir",
        type=Path,
        default=None,
        help="Optional folder of pre-existing input point clouds "
        "({sample_id}.npz with key 'pointcloud'); symlinked into "
        "<output_dir>/pcd/gt/ so PFD and l-GD work without input meshes.",
    )
    p.add_argument(
        "--instructions_json",
        type=Path,
        default=None,
        help='JSON mapping {"sample_id": "..."} or '
        '{"sample_id": {"instruction","obj_class","part_keyword"}}. '
        "If omitted, a placeholder is generated and CLIP-Dir / l-GD become meaningless.",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where to write renders/, pcd/, results.json, per_sample.csv",
    )

    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(METRIC_GROUPS),
        choices=METRIC_GROUPS,
        help="Metric groups to compute",
    )
    p.add_argument(
        "--num_points", type=int, default=2048, help="Points per sampled cloud"
    )
    p.add_argument(
        "--num_views",
        type=int,
        default=1,
        help="Views per mesh (default 1 = single view)",
    )
    p.add_argument(
        "--image_size", type=int, default=512, help="Render resolution (square)"
    )
    p.add_argument(
        "--pred_rotation",
        type=float,
        nargs=3,
        default=[-90.0, 0.0, 0.0],
        metavar=("RX", "RY", "RZ"),
        help="Euler rotation (deg) applied to pred meshes before rendering. "
        "Default suits Prox-E .glb outputs (TRELLIS coords). "
        "Pass '0 0 0' to disable.",
    )
    p.add_argument(
        "--input_rotation",
        type=float,
        nargs=3,
        default=None,
        metavar=("RX", "RY", "RZ"),
        help="Euler rotation (deg) applied to input meshes before rendering "
        "(default: no rotation)",
    )
    p.add_argument(
        "--render_norm",
        action="store_true",
        help="Normalize mesh bbox before rendering",
    )

    p.add_argument(
        "--skip_render",
        action="store_true",
        help="Reuse cached renders under {output_dir}/renders/",
    )
    p.add_argument(
        "--skip_pcd",
        action="store_true",
        help="Reuse cached point clouds under {output_dir}/pcd/",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N samples (after sorting). Useful for smoke tests.",
    )
    p.add_argument(
        "--enable_vqa",
        action="store_true",
        help='Enable VQAScore (P("Yes") from a VLM judging pred vs GT renders '
        "against the instruction). Backend is --vqa_model.",
    )
    p.add_argument(
        "--vqa_model",
        default="qwen2.5-vl-7b",
        choices=[
            "sail-vl-8b",
            "sail-vl-8b-thinking",
            "qwen2.5-vl-7b",
            "sensenova-si-internvl3-8b",
        ],
        help="VLM backend for VQAScore (used only when --enable_vqa)",
    )
    p.add_argument(
        "--vqa_cache_dir",
        type=Path,
        default=None,
        help="HuggingFace cache_dir for VQAScore model weights. "
        "Default: HF's standard cache (~/.cache/huggingface/hub or $HF_HOME). "
        "Pass evals/checkpoints/vqascore/ for a project-local cache.",
    )
    p.add_argument(
        "--vqa_checkpoint",
        type=str,
        default=None,
        help="Override the model weights path for --vqa_model (local dir or HF id). "
        "Use this when the remote trust_remote_code files are broken against "
        "the installed transformers (e.g. SAIL-VL2's modeling_qwen3.py needs a "
        "patched LossKwargs import). Default: the hub id baked into the wrapper.",
    )
    p.add_argument(
        "--lgd_ckpt_dir",
        type=Path,
        default=Path("evals/checkpoints/lgd"),
        help="Directory holding per-class {class}_100.onnx segmenters used for l-GD "
        "(auto-downloaded from ailia-models on first use)",
    )
    p.add_argument(
        "--no_lgd_normalize_pcd",
        dest="lgd_normalize_pcd",
        action="store_false",
        help="Disable l-GD bbox unit-sphere normalization before Chamfer, "
        "matching ChangeIt3D refined --normalize_pcd false",
    )
    p.add_argument(
        "--lgd_pred_pcd_transform",
        default="proxe",
        choices=["default", "proxe", "blendedpc", "none"],
        help="Coordinate transform applied to predicted point clouds before l-GD",
    )
    p.set_defaults(lgd_normalize_pcd=True)
    p.add_argument(
        "--pfd_model_name",
        default="proxe",
        choices=["proxe", "trellis", "blendedpc", "none"],
        help="Coordinate transform convention for PFD point clouds; "
        "matches proxe/evals/scripts/evaluate_pfid.py",
    )
    p.add_argument(
        "--pfid_model_name",
        dest="pfd_model_name",
        choices=["proxe", "trellis", "blendedpc", "none"],
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--pfd_src_pc_from",
        default="gt",
        choices=["gt", "proxe", "none"],
        help="Which PFD side needs coordinate conversion. Default matches "
        "evaluate_pfid.py: transform predicted point clouds into GT coords.",
    )
    p.add_argument(
        "--pfid_src_pc_from",
        dest="pfd_src_pc_from",
        choices=["gt", "proxe", "none"],
        help=argparse.SUPPRESS,
    )
    return p.parse_args(argv)


def _index_meshes(d: Path, label: str) -> Dict[str, Path]:
    """Index .{glb,obj,ply} files under ``d`` by file stem."""
    if not d.is_dir():
        sys.exit(f"{label} is not a directory: {d}")
    out: Dict[str, Path] = {}
    for p in sorted(d.iterdir()):
        if p.suffix.lower() in MESH_EXTS:
            out[p.stem] = p
    return out


def _seed_render_cache(
    src_render_dir: Path,
    side: str,
    sample_ids: List[str],
    output_dir: Path,
) -> int:
    """Symlink ``{src_render_dir}/{sid}.png`` into ``{output_dir}/renders/{side}/``.

    ``side`` is ``"pred"`` or ``"gt"``. Files placed here are picked up by
    ``preprocessing.render_meshes`` as already-cached and skipped, so we never
    need meshes on that side to feed image-based metrics.
    """
    cache = output_dir / "renders" / side
    cache.mkdir(parents=True, exist_ok=True)
    count = 0
    for sid in sample_ids:
        src = src_render_dir / f"{sid}.png"
        if not src.exists():
            continue
        dst = cache / f"{sid}.png"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        count += 1
    return count


def _seed_pcd_cache(
    src_pcd_dir: Path,
    side: str,
    sample_ids: List[str],
    output_dir: Path,
) -> int:
    """Symlink ``{src_pcd_dir}/{sid}.npz`` into ``{output_dir}/pcd/{side}/``.

    Files placed here are picked up by ``preprocessing.mesh_to_pcd_pairs`` as
    already-cached and skipped, so we never need meshes on that side for
    point-cloud-based metrics (PFD, l-GD).
    """
    cache = output_dir / "pcd" / side
    cache.mkdir(parents=True, exist_ok=True)
    count = 0
    for sid in sample_ids:
        src = src_pcd_dir / f"{sid}.npz"
        if not src.exists():
            continue
        dst = cache / f"{sid}.npz"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src.resolve())
        count += 1
    return count


def prepare_inputs(args) -> Tuple[List[str], Dict[str, Path], Dict[str, Path], Dict]:
    """Resolve sample ids, mesh-path indices, and instructions.

    Side effects:
        - If ``--input_render_dir`` is set, symlinks its PNGs into
          ``<output_dir>/renders/gt/``.
        - If ``--instructions_json`` is missing, writes a placeholder to
          ``<output_dir>/instructions.json`` and warns loudly.

    Returns
    -------
    sample_ids   : sorted list of ids in the working set
    pred_paths   : {sid -> Path} for predicted meshes
    input_paths  : {sid -> Path} for input meshes (empty dict if --input_dir not set)
    instructions : the (possibly placeholder) instructions dict
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pred_dir is None and args.pred_render_dir is None:
        sys.exit(
            "must pass either --pred_dir (mesh folder) or --pred_render_dir (PNG folder)"
        )

    pred_paths: Dict[str, Path] = {}
    if args.pred_dir is not None:
        pred_paths = _index_meshes(args.pred_dir, "--pred_dir")
        if not pred_paths:
            sys.exit(f"--pred_dir contained no meshes: {args.pred_dir}")

    pred_render_ids = None
    if args.pred_render_dir is not None:
        if not args.pred_render_dir.is_dir():
            sys.exit(f"--pred_render_dir is not a directory: {args.pred_render_dir}")
        pred_render_ids = {p.stem for p in args.pred_render_dir.glob("*.png")}
        if not pred_render_ids:
            sys.exit(f"--pred_render_dir contained no PNGs: {args.pred_render_dir}")

    input_paths: Dict[str, Path] = {}
    if args.input_dir is not None:
        input_paths = _index_meshes(args.input_dir, "--input_dir")

    input_render_ids = None
    if args.input_render_dir is not None:
        if not args.input_render_dir.is_dir():
            sys.exit(f"--input_render_dir is not a directory: {args.input_render_dir}")
        input_render_ids = {p.stem for p in args.input_render_dir.glob("*.png")}

    input_pcd_ids = None
    if args.input_pcd_dir is not None:
        if not args.input_pcd_dir.is_dir():
            sys.exit(f"--input_pcd_dir is not a directory: {args.input_pcd_dir}")
        input_pcd_ids = {p.stem for p in args.input_pcd_dir.glob("*.npz")}

    instructions = None
    if args.instructions_json is not None and args.instructions_json.exists():
        instructions = json.loads(args.instructions_json.read_text())

    # Seed the working id set from whichever pred source was given.
    id_set = set(pred_paths) if pred_paths else set(pred_render_ids or ())
    if pred_paths and pred_render_ids is not None:
        id_set &= pred_render_ids
    if input_paths:
        id_set &= set(input_paths)
    if input_render_ids is not None:
        id_set &= input_render_ids
    if input_pcd_ids is not None:
        id_set &= input_pcd_ids
    if instructions is not None:
        id_set &= set(instructions)
    sample_ids = sorted(id_set)
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]
    if not sample_ids:
        sys.exit(
            "no samples to evaluate (intersection of pred/input/renders/instructions is empty)"
        )

    # Placeholder instructions: write to disk for reproducibility, warn loudly.
    if instructions is None:
        instructions = {sid: "edit the object" for sid in sample_ids}
        placeholder_path = args.output_dir / "instructions.json"
        placeholder_path.write_text(json.dumps(instructions, indent=2))
        print(
            f"[warn] no --instructions_json provided; wrote placeholder to {placeholder_path}\n"
            f"[warn] CLIP-Dir and l-GD will be meaningless without real instructions + obj_class + part_keyword",
            file=sys.stderr,
        )

    # Seed cached renders before any rendering runs.
    if args.pred_render_dir is not None:
        n = _seed_render_cache(
            args.pred_render_dir, "pred", sample_ids, args.output_dir
        )
        print(
            f"[pred_render_dir] seeded {n} cached pred renders into {args.output_dir / 'renders' / 'pred'}"
        )
    if args.input_render_dir is not None:
        n = _seed_render_cache(args.input_render_dir, "gt", sample_ids, args.output_dir)
        print(
            f"[input_render_dir] seeded {n} cached input renders into {args.output_dir / 'renders' / 'gt'}"
        )
    if args.input_pcd_dir is not None:
        n = _seed_pcd_cache(args.input_pcd_dir, "gt", sample_ids, args.output_dir)
        print(
            f"[input_pcd_dir] seeded {n} cached input point clouds into {args.output_dir / 'pcd' / 'gt'}"
        )

    return sample_ids, pred_paths, input_paths, instructions


def write_per_sample_csv(path: Path, per_sample: Dict[str, Dict[str, float]]) -> None:
    if not per_sample:
        return
    keys = sorted({k for vals in per_sample.values() for k in vals})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id"] + keys)
        for sid in sorted(per_sample):
            w.writerow([sid] + [per_sample[sid].get(k, "") for k in keys])


def scale_lgd_results(
    aggregate: Dict[str, object],
    per_sample: Dict[str, Dict[str, float]],
) -> None:
    if "l-GD" in aggregate:
        aggregate["l-GD"] = aggregate["l-GD"] / 1000.0
    if "l-GD_per_class" in aggregate:
        aggregate["l-GD_per_class"] = {
            cls: value / 1000.0 for cls, value in aggregate["l-GD_per_class"].items()
        }
    for vals in per_sample.values():
        if "l-GD" in vals:
            vals["l-GD"] /= 1000.0


def main(argv=None) -> int:
    args = parse_args(argv)

    sample_ids, pred_paths, input_paths, instructions = prepare_inputs(args)
    print(f"Evaluating {len(sample_ids)} samples")
    have_input_meshes = bool(input_paths)

    # Local imports so --help works without heavy deps loaded.
    from evals import preprocessing
    from evals import metrics_identity, metrics_quality, metrics_fidelity

    render_dir = args.output_dir / "renders"
    have_pred_meshes = bool(pred_paths)
    if not args.skip_render and (have_pred_meshes or have_input_meshes):
        preprocessing.render_meshes(
            sample_ids=sample_ids,
            pred_paths=pred_paths,
            gt_paths=input_paths,  # empty dict -> input render side is skipped (only cached renders used)
            output_dir=render_dir,
            num_views=args.num_views,
            image_size=args.image_size,
            pred_rotation=args.pred_rotation,
            gt_rotation=args.input_rotation,
            render_norm=args.render_norm,
        )

    pcd_dir = args.output_dir / "pcd"
    if not args.skip_pcd and have_pred_meshes:
        if not have_input_meshes and args.input_pcd_dir is None:
            print(
                "[pcd] --input_dir not set; sampling pred-only point clouds "
                "(PFD and l-GD will be skipped)"
            )
        preprocessing.mesh_to_pcd_pairs(
            sample_ids=sample_ids,
            pred_paths=pred_paths,
            gt_paths=input_paths,
            output_dir=pcd_dir,
            num_points=args.num_points,
        )

    results: Dict[str, object] = {"n_samples": len(sample_ids)}
    per_sample: Dict[str, Dict[str, float]] = {sid: {} for sid in sample_ids}

    if "identity" in args.metrics:
        agg, per = metrics_identity.compute(
            sample_ids=sample_ids,
            render_dir=render_dir,
            pcd_dir=pcd_dir,
            device=args.device,
            instructions=instructions,
            lgd_ckpt_dir=args.lgd_ckpt_dir,
            lgd_normalize_pcd=args.lgd_normalize_pcd,
            lgd_pred_pcd_transform=args.lgd_pred_pcd_transform,
        )
        scale_lgd_results(agg, per)
        results["identity_preservation"] = agg
        for sid, vals in per.items():
            per_sample[sid].update(vals)

    if "quality" in args.metrics:
        agg, per = metrics_quality.compute(
            sample_ids=sample_ids,
            render_dir=render_dir,
            pcd_dir=pcd_dir,
            device=args.device,
            pfd_model_name=args.pfd_model_name,
            pfd_src_pc_from=args.pfd_src_pc_from,
        )
        results["3d_quality"] = agg
        for sid, vals in per.items():
            per_sample[sid].update(vals)

    if "fidelity" in args.metrics:
        agg, per = metrics_fidelity.compute(
            sample_ids=sample_ids,
            render_dir=render_dir,
            instructions=instructions,
            device=args.device,
            enable_vqa=args.enable_vqa,
            vqa_model=args.vqa_model,
            vqa_cache_dir=args.vqa_cache_dir,
            vqa_checkpoint=args.vqa_checkpoint,
        )
        results["edit_fidelity"] = agg
        for sid, vals in per.items():
            per_sample[sid].update(vals)

    results_path = args.output_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    write_per_sample_csv(args.output_dir / "per_sample.csv", per_sample)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
