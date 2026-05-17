#!/usr/bin/env python3
"""
Render a swing animation: camera azimuth offset follows 0° → -swing → +swing° → 0° over one loop,
with the left half (0 → -swing → 0) and right half (0 → +swing → 0) using the **same** number of
frames, and tunable ``--swing-convexity`` blending a triangle wave (constant speed per quarter)
with a smooth sine (default), then encodes frames to an MP4.

Default camera / render settings match ``visualizations_for_slides`` teaser pipeline GLB renders
(dist=2, azim=70, elev=20, fov=45, light_energy=2.5, invisible ground, transparent film).
Default mesh orientation matches teaser ``original_slat.glb`` (inv(M) @ R_base view matrix).

Use ``--video-only`` with ``--output`` to build ``swing.mp4`` from existing ``frames/swing_*.png``
without re-rendering. Encoding defaults to **OpenCV** ``VideoWriter`` (``pip install opencv-python-headless``);
use ``--encoder ffmpeg`` to force ffmpeg if preferred.

Use ``--no-smooth-shading`` to skip Blender smooth shading on imported meshes (faceted look).
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Repo root for Blender render path only (lazy import)
REPO_ROOT = Path(__file__).resolve().parent.parent


def output_directory(path_str: str) -> Path:
    """Output root as given for absolute paths (no symlink resolution)."""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        return (Path.cwd() / p).resolve()
    return p


# Match visualizations_for_slides teaser renders for pipeline GLBs
DEFAULT_DIST = 2.0
DEFAULT_AZIM_BASE = 80.0
DEFAULT_ELEV = 20.0
DEFAULT_FOV = 45.0
DEFAULT_LIGHT_ENERGY = 2.5
DEFAULT_RES = 1024
TEASER_PIPELINE_ORIENT_IDX = 5
GLB_BASE_EULER = (-90.0, 0.0, 0.0)

DEFAULT_SWING_FRAMES = 100
DEFAULT_SWING_CONVEXITY = 2.0

# Teaser abstraction GLBs encode gray / blue / purple as vertex colors; Blender must use the
# vertex-color shader (see ``force_glb_vertex_color`` in utils.render_obj_with_blender), not
# default glTF PBR, or colors look wrong.
GLB_FILENAMES_FORCE_VERTEX_COLOR = frozenset(
    {
        "original_abstraction.glb",
        "edited_abstraction_categorized.glb",
    }
)


def mesh_wants_glb_vertex_color(mesh_path: Path, *, transform: bool) -> bool:
    """True when swing should render with Blender's vertex-color path for this mesh."""
    if transform:
        return True
    return mesh_path.name.lower() in GLB_FILENAMES_FORCE_VERTEX_COLOR


def _triangle_swing_offset(u: float, s_amplitude: float) -> float:
    """Piecewise-linear cycle 0 → -S → 0 → +S → 0; each quarter uses 25% of u ∈ [0,1]."""
    x = u * 4.0
    if x <= 1.0:
        return -s_amplitude * x
    if x <= 2.0:
        return -s_amplitude * (2.0 - x)
    if x <= 3.0:
        return s_amplitude * (x - 2.0)
    return s_amplitude * (4.0 - x)


def _sine_swing_offset(u: float, s_amplitude: float) -> float:
    """Smooth cycle: u ∈ [0, ½] is 0 → -S → 0; u ∈ [½, 1] is 0 → +S → 0 (equal time per lobe)."""
    return -s_amplitude * math.sin(2.0 * math.pi * u)


def azimuth_offsets_deg_list(
    frame_count: int,
    swing_deg: float,
    *,
    convexity: float = DEFAULT_SWING_CONVEXITY,
) -> list[float]:
    """
    Azimuth offsets in degrees: one full cycle 0 → -swing → +swing → 0.

    Normalized time ``u`` runs 0→1 across frames. The **left lobe** (``u`` from 0 to ½) and
    **right lobe** (``u`` from ½ to 1) each use half of the frames. Offsets are a blend of a
    triangle wave (``convexity → 1``) and a sine (``convexity ≥ 2``), so motion stays smooth
    at offset 0 (no piecewise-power derivative kinks).
    """
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    if frame_count == 1:
        return [0.0]
    s_amplitude = float(swing_deg)
    # 1 → pure triangle; 2 → pure sine; between → blend. >2 → sine (same as 2).
    w = min(1.0, max(0.0, float(convexity) - 1.0))

    def offset_at_global_u(u: float) -> float:
        u = max(0.0, min(1.0, u))
        tri = _triangle_swing_offset(u, s_amplitude)
        sine = _sine_swing_offset(u, s_amplitude)
        return (1.0 - w) * tri + w * sine

    return [offset_at_global_u(i / (frame_count - 1)) for i in range(frame_count)]


def discover_swing_frames(frames_dir: Path, stem: str = "swing") -> int:
    """
    Return N if ``stem_0000.png`` … ``stem_{N-1}.png`` exist under ``frames_dir`` (contiguous, from 0).
    """
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")
    rx = re.compile(rf"^{re.escape(stem)}_(\d+)\.png$", re.IGNORECASE)
    indices = set()
    for p in frames_dir.iterdir():
        if not p.is_file():
            continue
        m = rx.match(p.name)
        if m:
            indices.add(int(m.group(1)))
    if not indices:
        raise FileNotFoundError(f"No {stem}_*.png files in {frames_dir}")
    ordered = sorted(indices)
    if ordered[0] != 0:
        raise ValueError(f"Expected {stem}_0000.png; smallest index is {ordered[0]}")
    for j, idx in enumerate(ordered):
        if idx != j:
            raise ValueError(f"Missing {stem}_{j:04d}.png (gap before index {idx})")
    return len(ordered)


def _bgra_to_bgr_white(img) -> "object":
    """Composite BGRA (OpenCV) onto white; return BGR uint8."""
    import cv2
    import numpy as np

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        return img
    b, g, r, a = cv2.split(img)
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32)
    g = g.astype(np.float32)
    r = r.astype(np.float32)
    b = b * a + 255.0 * (1.0 - a)
    g = g * a + 255.0 * (1.0 - a)
    r = r * a + 255.0 * (1.0 - a)
    return cv2.merge([b, g, r]).astype(np.uint8)


def _verify_video_file(path: Path, *, min_bytes: int = 256) -> None:
    if not path.is_file():
        raise RuntimeError(f"Expected output file missing after encode: {path}")
    if path.stat().st_size < min_bytes:
        raise RuntimeError(
            f"Output file is too small ({path.stat().st_size} B); encode likely failed. "
            f"Try: --encoder ffmpeg"
        )


def _install_encoded_file(src: Path, dst: Path) -> None:
    """
    Move ``src`` to ``dst``. Same-device: ``os.replace``. Cross-device (e.g. ``/tmp`` → NFS):
    ``shutil.copyfile`` only — some FUSE backends reject ``utime``/metadata (``copy2``/``move``).
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
        return
    except OSError:
        pass
    try:
        shutil.copyfile(src, dst)
    finally:
        if src.exists():
            try:
                src.unlink()
            except OSError:
                pass


def run_opencv_video_writer(
    frames_dir: Path,
    out_video: Path,
    n_frames: int,
    fps: float,
    stem: str = "swing",
) -> None:
    """
    Encode numbered PNGs to MP4 using OpenCV (RGBA composited on white).

    Writes to a temp file under the OS temp dir (e.g. ``/tmp``), then moves to ``out_video``.
    Some FUSE / object-store mounts accept ``VideoWriter`` but never materialize the file on
    the mount; encoding on a local filesystem avoids that.
    """
    import cv2

    out_video = Path(out_video)
    fd, tmp_str = tempfile.mkstemp(
        suffix=".mp4", prefix=f"swing_cv2_{os.getpid()}_{time.time_ns()}_", dir=tempfile.gettempdir()
    )
    os.close(fd)
    tmp_path = Path(tmp_str)

    path0 = frames_dir / f"{stem}_0000.png"
    first = cv2.imread(str(path0), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise FileNotFoundError(f"Could not read {path0}")
    h, w = first.shape[:2]
    bgr0 = _bgra_to_bgr_white(first)

    fourcc_candidates = ("mp4v", "avc1", "XVID")
    writer = None
    out_path = str(tmp_path)
    for tag in fourcc_candidates:
        fourcc = cv2.VideoWriter_fourcc(*tag)
        wrt = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
        if wrt.isOpened():
            writer = wrt
            break
        wrt.release()
    if writer is None or not writer.isOpened():
        raise RuntimeError(
            "cv2.VideoWriter could not open MP4 (try reinstalling opencv or use --encoder ffmpeg)."
        )

    try:
        writer.write(bgr0)
        for i in range(1, n_frames):
            p = frames_dir / f"{stem}_{i:04d}.png"
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Missing or unreadable frame {p}")
            writer.write(_bgra_to_bgr_white(img))
    finally:
        writer.release()

    try:
        _verify_video_file(tmp_path)
        _install_encoded_file(tmp_path, out_video)
        _verify_video_file(out_video)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise

    print(f"Wrote video (OpenCV): {out_video} ({out_video.stat().st_size} bytes)")


def run_ffmpeg(
    frames_dir: Path,
    pattern: str,
    out_video: Path,
    fps: float,
    *,
    require_ffmpeg: bool = False,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        msg = "ffmpeg not found in PATH; install ffmpeg or add it to PATH."
        if require_ffmpeg:
            print(msg, file=sys.stderr)
            sys.exit(1)
        print(msg, file=sys.stderr)
        return
    fd, tmp_str = tempfile.mkstemp(
        suffix=".mp4", prefix=f"swing_ff_{os.getpid()}_{time.time_ns()}_", dir=tempfile.gettempdir()
    )
    os.close(fd)
    tmp_out = Path(tmp_str)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(tmp_out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except OSError:
                pass
        print(proc.stderr or proc.stdout, file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}")
    out_p = Path(out_video)
    try:
        _verify_video_file(tmp_out)
        _install_encoded_file(tmp_out, out_p)
        _verify_video_file(out_p)
    except Exception:
        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except OSError:
                pass
        raise
    print(f"Wrote video (ffmpeg): {out_p} ({out_p.stat().st_size} bytes)")


def encode_swing_video(
    frames_dir: Path,
    out_video: Path,
    n_frames: int,
    fps: float,
    stem: str,
    *,
    encoder: str,
    require_success: bool,
) -> None:
    """
    encoder: 'auto' | 'cv2' | 'ffmpeg'
    """
    if encoder == "ffmpeg":
        run_ffmpeg(frames_dir, f"{stem}_%04d.png", out_video, fps, require_ffmpeg=require_success)
        return

    if encoder == "cv2":
        try:
            run_opencv_video_writer(frames_dir, out_video, n_frames, fps, stem=stem)
        except Exception as e:
            print(f"OpenCV encoding failed: {e}", file=sys.stderr)
            if require_success:
                sys.exit(1)
            raise
        return

    # auto: try OpenCV first, then ffmpeg
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("OpenCV not installed; falling back to ffmpeg. (pip install opencv-python-headless)", file=sys.stderr)
        run_ffmpeg(frames_dir, f"{stem}_%04d.png", out_video, fps, require_ffmpeg=require_success)
        return

    try:
        run_opencv_video_writer(frames_dir, out_video, n_frames, fps, stem=stem)
    except Exception as e:
        print(f"OpenCV encoding failed ({e}); trying ffmpeg...", file=sys.stderr)
        run_ffmpeg(frames_dir, f"{stem}_%04d.png", out_video, fps, require_ffmpeg=require_success)


def _run_blender_swing(args: argparse.Namespace) -> None:
    """Import numpy/utils only when actually rendering in Blender."""
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT))
    from scripts.prepare_compound_edit import ORIENTATION_TRANSFORMS
    from utils import render_obj_with_blender_sequence

    def _blender_euler_xyz_deg_to_matrix_3x3(rx: float, ry: float, rz: float) -> np.ndarray:
        rx, ry, rz = [math.radians(x) for x in (rx, ry, rz)]
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        r_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        return r_x @ r_y @ r_z

    def teaser_pipeline_glb_rotation_matrix() -> np.ndarray:
        r_base = _blender_euler_xyz_deg_to_matrix_3x3(*GLB_BASE_EULER)
        _, m_full = ORIENTATION_TRANSFORMS[TEASER_PIPELINE_ORIENT_IDX]
        m3 = np.asarray(m_full[:3, :3], dtype=np.float64)
        return np.linalg.inv(m3) @ r_base

    mesh_path = Path(args.mesh).resolve()
    out_dir = output_directory(args.output)
    frames_dir = out_dir / "frames"
    stem = "swing"
    video_path = out_dir / "swing.mp4"

    frames_dir.mkdir(parents=True, exist_ok=True)

    if getattr(args, "transform", False):
        _viz_dir = Path(__file__).resolve().parent
        if str(_viz_dir) not in sys.path:
            sys.path.insert(0, str(_viz_dir))
        from swing_abstraction_morph import (
            build_swing_mesh_paths,
            find_edited_abstraction_json,
            resolve_teaser_result_folder,
        )

        if mesh_path.name != "original_abstraction.glb":
            print(
                "With --transform, --mesh must point to original_abstraction.glb "
                f"(got {mesh_path.name})",
                file=sys.stderr,
            )
            sys.exit(1)
        edited_glb = mesh_path.parent / "edited_abstraction_categorized.glb"
        if not edited_glb.is_file():
            print(f"Missing edited mesh: {edited_glb}", file=sys.stderr)
            sys.exit(1)
        result_folder = resolve_teaser_result_folder(mesh_path)
        original_json = result_folder / "abstraction.json"
        if not original_json.is_file():
            print(f"Missing {original_json}", file=sys.stderr)
            sys.exit(1)
        try:
            edited_json = find_edited_abstraction_json(result_folder)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
        morph_dir = out_dir / "_swing_morph_glbs"
        morph_dir.mkdir(parents=True, exist_ok=True)

    if args.mesh_rotation == "teaser-pipeline":
        rot_mat = teaser_pipeline_glb_rotation_matrix()
        rot_kw = dict(rotation_matrix=rot_mat, rotation=None)
    elif args.mesh_rotation == "none":
        rot_kw = dict(rotation=None, rotation_matrix=None)
    else:
        rot_kw = dict(rotation=(args.rx, args.ry, args.rz), rotation_matrix=None)

    offsets = azimuth_offsets_deg_list(args.frames, args.swing, convexity=args.swing_convexity)
    azims = [args.azim + offsets[i] for i in range(args.frames)]
    out_pngs = [str(frames_dir / f"{stem}_{i:04d}.png") for i in range(args.frames)]

    render_kw = dict(
        res_x=args.res,
        res_y=args.res,
        dist=args.dist,
        elev=args.elev,
        fov=args.fov,
        light_energy=args.light_energy,
        transparent=True,
        # Teaser abstraction GLBs rely on vertex colors; Blender must use the vertex-color
        # shader path (same as teaser PNG renders), not default glTF PBR.
        force_glb_vertex_color=mesh_wants_glb_vertex_color(
            mesh_path, transform=bool(getattr(args, "transform", False))
        ),
        invisible_ground=not args.no_ground,
        cycles_gpu=not args.no_gpu,
        shade_smooth=not args.no_smooth_shading,
        **rot_kw,
    )

    if getattr(args, "transform", False):
        mesh_paths, _segments = build_swing_mesh_paths(
            total_frames=args.frames,
            start_transform=args.start_transform_frame,
            num_transform=args.num_transformation_frames,
            original_glb=mesh_path,
            edited_glb=edited_glb,
            original_json=original_json,
            edited_json=edited_json,
            temp_dir=morph_dir,
            resolution=args.abstraction_resolution,
        )
        print(
            f"Swing + abstraction morph: {args.frames} frames, "
            f"morph frames [{args.start_transform_frame}, "
            f"{args.start_transform_frame + args.num_transformation_frames - 1}] "
            f"({args.num_transformation_frames} steps), azim {azims[0]:.4f}° … {azims[-1]:.4f}°"
        )
        i = 0
        seg_n = 0
        while i < args.frames:
            j = i
            while j + 1 < args.frames and mesh_paths[j + 1] == mesh_paths[i]:
                j += 1
            seg_n += 1
            print(
                f"Blender session {seg_n}: frames {i + 1}–{j + 1} "
                f"({j - i + 1} frames) → {mesh_paths[i]}"
            )
            r = render_obj_with_blender_sequence(
                mesh_paths[i],
                azims[i : j + 1],
                out_pngs[i : j + 1],
                **render_kw,
            )
            if r.returncode != 0:
                print(r.stderr or r.stdout, file=sys.stderr)
                sys.exit(r.returncode or 1)
            i = j + 1
    else:
        print(
            f"Rendering {args.frames} frames in one Blender session (azim "
            f"{azims[0]:.4f}° … {azims[-1]:.4f}°)…"
        )
        r = render_obj_with_blender_sequence(str(mesh_path), azims, out_pngs, **render_kw)
        if r.returncode != 0:
            print(r.stderr or r.stdout, file=sys.stderr)
            sys.exit(r.returncode or 1)

    if not args.skip_video:
        encode_swing_video(
            frames_dir,
            video_path,
            args.frames,
            args.fps,
            stem,
            encoder=args.encoder,
            require_success=False,
        )
    else:
        print(f"Skipping video (--skip-video). Frames: {frames_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Swing camera animation around a mesh (Blender + OpenCV/ffmpeg).")
    p.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Path to mesh (.glb, .obj, .ply, …). Not used with --video-only.",
    )
    p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory (frames in frames/; video swing.mp4 at top level)",
    )
    p.add_argument("--frames", type=int, default=DEFAULT_SWING_FRAMES, help=f"Number of frames (default: {DEFAULT_SWING_FRAMES})")
    p.add_argument("--swing", type=float, default=15.0, help="Half-range of azimuth swing in degrees (default: 15)")
    p.add_argument(
        "--swing-convexity",
        type=float,
        default=DEFAULT_SWING_CONVEXITY,
        metavar="C",
        help=(
            "Blend shape: 1=triangle (constant speed in each quarter); 2=smooth sine (default); "
            "values between 1 and 2 interpolate. Left and right half-cycles (0→-swing→0 vs "
            "0→+swing→0) always use the same number of frames."
        ),
    )
    p.add_argument("--fps", type=float, default=30.0, help="Video frame rate (default: 30)")
    p.add_argument("--dist", type=float, default=DEFAULT_DIST, help=f"Camera distance (default: {DEFAULT_DIST})")
    p.add_argument("--azim", type=float, default=DEFAULT_AZIM_BASE, help=f"Base azimuth ° (default: {DEFAULT_AZIM_BASE})")
    p.add_argument("--elev", type=float, default=DEFAULT_ELEV, help=f"Elevation ° (default: {DEFAULT_ELEV})")
    p.add_argument("--fov", type=float, default=DEFAULT_FOV, help=f"Vertical FOV ° (default: {DEFAULT_FOV})")
    p.add_argument(
        "--light-energy",
        type=float,
        default=DEFAULT_LIGHT_ENERGY,
        help=f"Area light scale (default: {DEFAULT_LIGHT_ENERGY})",
    )
    p.add_argument("--res", type=int, default=DEFAULT_RES, help=f"Square resolution (default: {DEFAULT_RES})")
    p.add_argument(
        "--mesh-rotation",
        choices=("teaser-pipeline", "none", "euler"),
        default="teaser-pipeline",
        help="Vertex rotation before camera: teaser pipeline GLB matrix, none, or euler (see --rx/--ry/--rz)",
    )
    p.add_argument("--rx", type=float, default=-90.0, help="Euler X° for --mesh-rotation euler (default: -90)")
    p.add_argument("--ry", type=float, default=0.0, help="Euler Y° for --mesh-rotation euler")
    p.add_argument("--rz", type=float, default=0.0, help="Euler Z° for --mesh-rotation euler")
    p.add_argument("--no-ground", action="store_true", help="Disable invisible shadow-catcher ground")
    p.add_argument(
        "--no-gpu",
        action="store_true",
        help="Use CPU for Cycles (default: enable CUDA/OPTIX/HIP/METAL when available).",
    )
    p.add_argument(
        "--no-smooth-shading",
        action="store_true",
        help="Skip Blender shade_smooth on meshes (faceted / non-smoothed look).",
    )
    p.add_argument("--skip-video", action="store_true", help="Only render PNG frames, do not call ffmpeg")
    p.add_argument(
        "--video-only",
        action="store_true",
        help="Skip Blender; encode existing output/frames/swing_*.png into swing.mp4.",
    )
    p.add_argument(
        "--encoder",
        type=str,
        choices=("auto", "cv2", "ffmpeg"),
        default="auto",
        help="Video backend: OpenCV (default auto), ffmpeg, or auto=OpenCV then ffmpeg fallback.",
    )
    p.add_argument(
        "--transform",
        action="store_true",
        help=(
            "Morph abstraction.json → edited JSON between meshes: requires --mesh "
            "…/original_abstraction.glb, abstraction.json + edited_abstraction*.json in "
            "the parent of teaser_renders/, and edited_abstraction_categorized.glb next to the mesh."
        ),
    )
    p.add_argument(
        "--start-transform-frame",
        type=int,
        default=0,
        metavar="F",
        help="First frame index (0-based) where JSON morph starts (default: 0).",
    )
    p.add_argument(
        "--num-transformation-frames",
        type=int,
        default=15,
        metavar="N",
        help="Number of frames over which the abstraction morph runs (default: 10).",
    )
    p.add_argument(
        "--abstraction-resolution",
        type=int,
        default=30,
        help="Superquadric mesh resolution for morph GLBs (default: 30, same as teaser).",
    )
    args = p.parse_args()

    if args.video_only and args.skip_video:
        p.error("Cannot combine --video-only with --skip-video")

    t0 = time.perf_counter()
    try:
        out_dir = output_directory(args.output)
        frames_dir = out_dir / "frames"
        stem = "swing"
        video_path = out_dir / "swing.mp4"

        if args.video_only:
            try:
                n = discover_swing_frames(frames_dir, stem=stem)
            except (FileNotFoundError, ValueError) as e:
                print(e, file=sys.stderr)
                sys.exit(1)
            print(f"Found {n} frames under {frames_dir}; encoding at {args.fps} fps -> {video_path}")
            encode_swing_video(
                frames_dir,
                video_path,
                n,
                args.fps,
                stem,
                encoder=args.encoder,
                require_success=True,
            )
            canon = out_dir.resolve()
            if canon != out_dir:
                print(f"Also visible at (same folder): {canon / 'swing.mp4'}")
            print(f"Done. Video file: {video_path}")
            return

        if not args.mesh:
            p.error("--mesh is required unless --video-only")

        if args.transform and args.start_transform_frame < 0:
            p.error("--start-transform-frame must be >= 0")
        if args.transform and args.num_transformation_frames < 1:
            p.error("--num-transformation-frames must be >= 1")
        if (
            args.transform
            and args.start_transform_frame + args.num_transformation_frames > args.frames
        ):
            p.error(
                "start-transform-frame + num-transformation-frames cannot exceed --frames"
            )

        _run_blender_swing(args)
    finally:
        print(f"Total time: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
