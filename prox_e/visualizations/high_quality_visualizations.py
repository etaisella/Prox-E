#!/usr/bin/env python3
"""
Visualizations for slides and papers.

This script creates various visualizations from tredit pipeline outputs:
1. Decoded voxels at every timestep of structure inversion
2. Voxelized edited abstraction with superquadric colors
3. ``--vlm-slide``: multiview renders of all abstraction JSONs (JSON colors; +90° X vertex frame)
"""

import sys
import shutil
import argparse
from typing import Optional, Set, Tuple
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import trimesh
import open3d as o3d

# Add parent directory (tredit repo root) to path for imports
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Add submodules to path
SUPERGEN_PATH = REPO_ROOT / "submodules" / "supergen"
sys.path.insert(0, str(SUPERGEN_PATH))

# Lazy import of trellis - only needed for timestep visualization
# from trellis.pipelines import TrellisTextTo3DPipeline
from inpaint_data_preparation import voxels_to_cube_mesh
from utils import render_obj_with_blender
from scripts.prepare_compound_edit import ORIENTATION_TRANSFORMS

# Noisy pointcloud mask palette (RGB 0–1). Unchanged voxels in inpainted mask modes = PCOLOR_ORANGE.
# Categorized / teaser *abstraction meshes* use DEFAULT_CATEGORIZED_UNCHANGED_RGB (same orange as PCOLOR_ORANGE).
PCOLOR_ORANGE = np.array([190.0, 120.0, 40.0], dtype=np.float64) / 255.0
PCOLOR_BLUE = np.array([40.0, 90.0, 180.0], dtype=np.float64) / 255.0
PCOLOR_PURPLE = np.array([150.0, 50.0, 170.0], dtype=np.float64) / 255.0
PCOLOR_GRAY = np.array([180.0, 180.0, 180.0], dtype=np.float64) / 255.0

DEFAULT_CATEGORIZED_UNCHANGED_RGB = [190, 120, 40]

# ORIENTATION_TRANSFORMS index for voxelizing warped mesh + edited abstraction in --noisy-pointcloud
# (post-normalize, before discretization). Picked from orientation sweep.
NOISY_POINTCLOUD_MESH_VOXEL_ORIENT_IDX = 6  # swap_y_z_flip_z

# Canonical single superquadric (abstraction.json schema) for --include-unit-sq / --categorized-and-unit-sq.
UNIT_SQ_EXPONENTS = (0.48, 0.52)  # mild curvature; typical of real extractions
UNIT_SQ_RGB_255 = (40, 90, 180)  # default: same as categorized ``changed`` blue


def unit_sq_abstraction_list(color_rgb_255: Optional[tuple] = None) -> list:
    """
    One superquadric: isotropic scale 0.1, identity rotation, zero translation.
    ``color_rgb_255`` defaults to ``UNIT_SQ_RGB_255`` (categorized blue).
    """
    c = list(color_rgb_255) if color_rgb_255 is not None else list(UNIT_SQ_RGB_255)
    return [
        {
            "index": 0,
            "scale": [0.1, 0.1, 0.1],
            "translation": [0.0, 0.0, 0.0],
            "rotation": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "exponents": [float(UNIT_SQ_EXPONENTS[0]), float(UNIT_SQ_EXPONENTS[1])],
            "color": c,
        }
    ]


# Teaser asset orientation indices into ORIENTATION_TRANSFORMS (edit3d-bench; see glb_orientation_sweep.py).
TEASER_MESH_ORIENTATIONS = {
    "original_slat": 5,  # swap_y_z
    "appearance_edited": 5,  # swap_y_z
    "original_abstraction": 1,  # flip_z
    "edited_abstraction_categorized": 1,  # flip_z
}

# Abstraction teaser Blender view: standalone ORIENTATION_TRANSFORMS[idx] on the baked mesh
# (sweep label a02_flip_y => index 2).
TEASER_ABSTRACTION_VIEW_ORIENT_IDX = 2


def teaser_mesh_orientation(mesh_key: str):
    """Return ORIENTATION_TRANSFORMS index for this teaser mesh, or None if unset."""
    return TEASER_MESH_ORIENTATIONS.get(mesh_key)


def _apply_teaser_orientation_mesh(mesh, orientation_idx):
    """Copy of mesh with ORIENTATION_TRANSFORMS[idx] applied; unchanged if idx is None."""
    if mesh is None or orientation_idx is None:
        return mesh
    _, mat = ORIENTATION_TRANSFORMS[orientation_idx]
    out = mesh.copy()
    out.apply_transform(mat)
    return out


def _blender_euler_xyz_deg_to_matrix_3x3(rx: float, ry: float, rz: float) -> np.ndarray:
    """Match utils.render_obj_with_blender: Rx @ Ry @ Rz on column vectors (degrees)."""
    rx, ry, rz = np.radians([rx, ry, rz])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    r_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    r_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    r_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return r_x @ r_y @ r_z


# ``--vlm-slide`` base views: vertex rotation in Blender (matches old default_pose_sweep ``01_R_pos90_x``).
VLM_SLIDE_BASE_VERTEX_ROTATION = _blender_euler_xyz_deg_to_matrix_3x3(90, 0, 0)


def _teaser_vertex_view_rotation_matrix(
    base_euler_deg_xyz, orientation_idx, compose: str = "invM_Rb"
):
    """
    Teaser GLBs encode vertices as M @ v. Blender applies R_v to imported vertices so the
    effective transform is R_v @ M @ v. To match the legacy view R_base @ v on raw mesh,
    use R_v @ M = R_base.

    Sweep showed for **pipeline** Trellis GLBs the correct compose is inv(M) @ R_base (option 02),
    i.e. R_v = inv(M) @ R_base (same as M.T @ R_base for orthogonal M).

    Args:
        compose: ``invM_Rb`` (default) or ``Rb_invM`` for R_base @ inv(M) (older attempt).
    """
    r_base = _blender_euler_xyz_deg_to_matrix_3x3(*base_euler_deg_xyz)
    if orientation_idx is None:
        return r_base
    _, m_full = ORIENTATION_TRANSFORMS[orientation_idx]
    m3 = np.asarray(m_full[:3, :3], dtype=np.float64)
    inv = np.linalg.inv(m3)
    if compose == "Rb_invM":
        return r_base @ inv
    return inv @ r_base


def _teaser_abstraction_blender_view_matrix() -> np.ndarray:
    """Vertex rotation matrix for abstraction teaser PNGs (see TEASER_ABSTRACTION_VIEW_ORIENT_IDX)."""
    _, mat = ORIENTATION_TRANSFORMS[TEASER_ABSTRACTION_VIEW_ORIENT_IDX]
    return np.asarray(mat[:3, :3], dtype=np.float64)


def _teaser_export_glb_from_source(src: Path, dst_glb: Path) -> None:
    """Write GLB beside teaser PNGs: copy if src is already .glb, else load mesh and export."""
    src = Path(src)
    dst_glb = Path(dst_glb)
    dst_glb.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".glb":
        # copyfile avoids copystat/utime; copy2 often fails on S3/FUSE/NFS destinations.
        shutil.copyfile(src, dst_glb)
        return
    mesh = trimesh.load(str(src), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    mesh.export(str(dst_glb), file_type="glb")


def _teaser_write_pipeline_glb(src: Path, dst_glb: Path, orientation_idx=None) -> None:
    """Write pipeline GLB to teaser_renders: copy or apply ORIENTATION_TRANSFORMS[index]."""
    src = Path(src)
    dst_glb = Path(dst_glb)
    if orientation_idx is None:
        _teaser_export_glb_from_source(src, dst_glb)
        return
    dst_glb.parent.mkdir(parents=True, exist_ok=True)
    _, mat = ORIENTATION_TRANSFORMS[orientation_idx]
    mesh = trimesh.load(str(src), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )
    mesh.apply_transform(mat)
    mesh.export(str(dst_glb), file_type="glb")


def load_pipeline():
    """Load the Trellis text-to-3D pipeline."""
    # Lazy import - only load when actually needed
    from trellis.pipelines import TrellisTextTo3DPipeline

    print("Loading TrellisTextTo3DPipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-large")
    pipeline.cuda()
    return pipeline


def decode_latent_to_voxels(pipeline, latent_tensor):
    """
    Decode a structure latent tensor to voxels.

    Args:
        pipeline: Trellis pipeline with sparse_structure_decoder
        latent_tensor: Latent tensor of shape [1, 8, 16, 16, 16]

    Returns:
        Voxel tensor of shape [1, 1, 64, 64, 64]
    """
    decoder = pipeline.models["sparse_structure_decoder"]
    latent_tensor = latent_tensor.cuda()
    with torch.no_grad():
        voxels = decoder(latent_tensor)
    return voxels


def save_voxels_as_ply(voxels, output_path, threshold: float):
    """
    Save voxel grid as PLY mesh with cube visualization.

    Args:
        voxels: Voxel tensor of shape [1, 1, 64, 64, 64] or [64, 64, 64]
        output_path: Path to save the PLY file
        threshold: Threshold for voxel occupancy
    """
    if voxels.dim() == 5:
        voxels = voxels[0, 0]  # Remove batch and channel dims

    voxels_np = (voxels.detach().cpu().numpy() > threshold).astype(np.float32)
    cube_mesh = voxels_to_cube_mesh(voxels_np, use_position_colors=True)
    o3d.io.write_triangle_mesh(str(output_path), cube_mesh, write_vertex_colors=True)


def voxelize_mesh_with_colors(
    mesh_path: str,
    resolution: int = 64,
    center: np.ndarray = None,
    scale: float = None,
) -> tuple:
    """
    Voxelize a mesh while preserving vertex colors.

    Uses Open3D's create_from_triangle_mesh_within_bounds for proper voxelization
    that captures the full mesh surface (including edges), not just vertices.

    Args:
        mesh_path: Path to mesh file with vertex colors (OBJ)
        resolution: Voxel grid resolution (default 64)
        center: Optional normalization center. If None, computed from mesh.
        scale: Optional normalization scale. If None, computed from mesh.

    Returns:
        Tuple of (voxel_grid, voxel_colors) where:
        - voxel_grid: np.ndarray of shape (resolution, resolution, resolution) with binary occupancy
        - voxel_colors: np.ndarray of shape (resolution, resolution, resolution, 3) with RGB colors [0-1]
    """
    from scipy.spatial import cKDTree

    # Load mesh with trimesh to get vertex colors
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(
            [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        )

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    # Get vertex colors (trimesh stores as RGBA)
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
        vertex_colors = (
            mesh.visual.vertex_colors[:, :3] / 255.0
        )  # RGB, normalize to [0-1]
    else:
        vertex_colors = np.ones((len(vertices), 3)) * 0.5  # Default gray

    # Normalize vertices to [-0.5, 0.5]^3 using provided or computed normalization
    if center is None:
        center = (vertices.max(0) + vertices.min(0)) / 2
    if scale is None:
        scale = (vertices.max(0) - vertices.min(0)).max()
    vertices_normalized = (vertices - center) / scale

    # Create Open3D mesh for proper voxelization
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices_normalized)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()

    # Voxelize using Open3D (captures edges and faces, not just vertices)
    voxel_size = 1.0 / resolution
    voxel_grid_o3d = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        o3d_mesh,
        voxel_size=voxel_size,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )

    # Convert to dense numpy array
    voxel_grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    voxel_centers = []  # Store centers for color lookup
    voxel_positions = []  # Store grid positions

    for voxel in voxel_grid_o3d.get_voxels():
        idx = voxel.grid_index
        if (
            0 <= idx[0] < resolution
            and 0 <= idx[1] < resolution
            and 0 <= idx[2] < resolution
        ):
            voxel_grid[idx[0], idx[1], idx[2]] = 1.0
            # Compute voxel center in normalized space
            voxel_center = (np.array(idx) + 0.5) * voxel_size - 0.5
            voxel_centers.append(voxel_center)
            voxel_positions.append(idx)

    # Build KD-tree of original vertices for color lookup
    tree = cKDTree(vertices_normalized)

    # Assign colors based on nearest vertex
    color_grid = np.zeros((resolution, resolution, resolution, 3), dtype=np.float32)

    if voxel_centers:
        voxel_centers = np.array(voxel_centers)
        # Find nearest vertex for each voxel
        distances, nearest_indices = tree.query(voxel_centers, k=1)

        for i, (pos, nearest_idx) in enumerate(zip(voxel_positions, nearest_indices)):
            color_grid[pos[0], pos[1], pos[2]] = vertex_colors[nearest_idx]

    return voxel_grid, color_grid


def voxels_to_colored_cube_mesh(
    voxels: np.ndarray, colors: np.ndarray
) -> o3d.geometry.TriangleMesh:
    """
    Convert a colored voxel grid to a mesh of cubes.

    Args:
        voxels: Binary voxel grid of shape (res, res, res)
        colors: RGB color grid of shape (res, res, res, 3) with values [0-1]

    Returns:
        Open3D TriangleMesh with colored cubes
    """
    resolution = voxels.shape[0]
    cube_size = 1.0 / resolution

    # Get occupied voxel positions
    occupied = np.argwhere(voxels > 0.5)

    if len(occupied) == 0:
        return o3d.geometry.TriangleMesh()

    all_vertices = []
    all_triangles = []
    all_colors = []

    # Unit cube vertices and triangles
    unit_verts = (
        np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.float32,
        )
        * cube_size
    )

    unit_tris = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # bottom
            [4, 5, 6],
            [4, 6, 7],  # top
            [0, 1, 5],
            [0, 5, 4],  # front
            [2, 3, 7],
            [2, 7, 6],  # back
            [0, 4, 7],
            [0, 7, 3],  # left
            [1, 2, 6],
            [1, 6, 5],  # right
        ],
        dtype=np.int32,
    )

    for pos in occupied:
        x, y, z = pos
        # Position offset (center at origin)
        offset = np.array([x, y, z], dtype=np.float32) * cube_size - 0.5

        # Add vertices
        verts = unit_verts + offset
        vert_offset = len(all_vertices)
        all_vertices.extend(verts)

        # Add triangles with offset
        tris = unit_tris + vert_offset
        all_triangles.extend(tris)

        # Add colors (same color for all 8 vertices of the cube)
        color = colors[x, y, z]
        all_colors.extend([color] * 8)

    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(all_vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(all_triangles))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(all_colors))
    mesh.compute_vertex_normals()

    return mesh


def visualize_inversion_timesteps(
    result_folder: str,
    output_dir: str,
    threshold: float,
    pipeline=None,
    skip_existing: bool = True,
):
    """
    Visualize decoded voxels at every timestep of structure inversion.

    Args:
        result_folder: Path to the tredit result folder
        output_dir: Directory to save output images
        threshold: Threshold for voxel occupancy
        pipeline: Optional pre-loaded pipeline (will load if None)
        skip_existing: Skip rendering if output already exists

    Returns:
        List of output image paths
    """
    result_folder = Path(result_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find latents file
    latents_path = result_folder / "inversion" / "original_shape_ss_latents.pt"
    if not latents_path.exists():
        print(f"Error: Latents file not found: {latents_path}")
        return []

    # Load latents
    print(f"Loading latents from: {latents_path}")
    data = torch.load(latents_path, weights_only=False)
    ss_latent = data["ss_latent"]

    # Sort timesteps (they're stored as string keys)
    timesteps = sorted(ss_latent.keys(), key=float)
    print(f"Found {len(timesteps)} timesteps (threshold={threshold})")

    # Load pipeline if not provided
    if pipeline is None:
        pipeline = load_pipeline()

    # Rotation to align Trellis-oriented voxels with standard view
    trellis_rotation = (-90, 0, 0)

    output_paths = []

    for i, t_key in enumerate(tqdm(timesteps, desc="Decoding and rendering timesteps")):
        t_value = float(t_key)

        # Output paths for this timestep
        ply_path = output_dir / f"timestep_{i:02d}_t{t_value:.3f}.ply"
        png_path = output_dir / f"timestep_{i:02d}_t{t_value:.3f}.png"

        if skip_existing and png_path.exists():
            output_paths.append(str(png_path))
            continue

        # Get latent for this timestep
        latent = ss_latent[t_key]
        if hasattr(latent, "cuda"):
            latent = latent.cuda()

        # Decode to voxels
        voxels = decode_latent_to_voxels(pipeline, latent)

        # Check if any voxels are occupied
        num_voxels = (voxels > threshold).sum().item()
        if num_voxels == 0:
            print(f"  Timestep {i} (t={t_value:.3f}): No voxels occupied, skipping")
            continue

        # Save as PLY
        save_voxels_as_ply(voxels, ply_path, threshold=threshold)

        # Render with invisible ground
        render_obj_with_blender(
            str(ply_path),
            str(png_path),
            rotation=trellis_rotation,
            invisible_ground=True,
        )

        output_paths.append(str(png_path))
        print(
            f"  Timestep {i} (t={t_value:.3f}): {num_voxels} voxels -> {png_path.name}"
        )

    print(f"\nSaved {len(output_paths)} renders to: {output_dir}")
    return output_paths, pipeline


def _pointcloud_noise_for_voxel_tensor(
    voxels_tensor: torch.Tensor,
    threshold: float,
    seed: int,
    noise_std: float,
) -> Optional[np.ndarray]:
    """
    Sample one Gaussian noise cloud (same shape as active voxel count) for reuse across
    multiple renders (e.g. original gray vs orange with identical motion).
    """
    vt = voxels_tensor
    if vt.is_cuda:
        vt = vt.cpu()
    if vt.dim() == 5:
        vt = vt[0, 0]
    elif vt.dim() == 4:
        vt = vt[0]
    active_mask = (vt > threshold).numpy()
    active_indices = np.argwhere(active_mask)
    n_active = len(active_indices)
    if n_active == 0:
        return None
    np.random.seed(seed)
    return np.random.normal(0, noise_std, (n_active, 3)).astype(np.float32)


def _voxelize_mesh_path_cpu(
    mesh_path: str,
    resolution: int = 64,
    center: np.ndarray = None,
    scale: float = None,
    post_normalize_transform_4x4: Optional[np.ndarray] = None,
    global_center: Optional[np.ndarray] = None,
    global_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Same normalization and occupancy as ``inversion.voxelize_mesh``, but returns a
    float32 CPU tensor of shape (1, 1, R, R, R) (no CUDA requirement).

    Normalization modes:
    - **Default**: per-mesh axis-aligned bbox center + max extent (mesh fills the voxel cube).
    - **Merge-aligned** (``global_center`` + ``global_scale``): same as
      ``inpaint_data_preparation.prepare_edit_data`` / ``voxelize_mesh`` — vertices become
      ``(v - global_center) / global_scale``, then clipped into ``[-0.5, 0.5]`` like mask
      construction. Use this when voxel occupancy must align with ``merging_data.pt`` masks.

    If ``post_normalize_transform_4x4`` is set, it is applied to **normalized** vertices
    (after centering and isotropic scale into the voxelization box) as ``x' = R @ x + t``
    using the upper 3x4 of the matrix (see ``ORIENTATION_TRANSFORMS`` in
    ``prepare_compound_edit``). Mask-aligned runs typically leave this ``None`` so Trellis
    orientation matches ``transform_voxels_to_trellis`` on masks.
    """
    mesh_path = Path(mesh_path)
    if mesh_path.suffix.lower() == ".ply":
        mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    else:
        tm = trimesh.load(str(mesh_path), force="mesh")
        if isinstance(tm, trimesh.Scene):
            parts = [g for g in tm.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if not parts:
                raise ValueError(f"No valid meshes found in {mesh_path}")
            tm = trimesh.util.concatenate(parts)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    use_merge_norm = global_center is not None and global_scale is not None
    if use_merge_norm:
        gc = np.asarray(global_center, dtype=np.float64).reshape(-1)[:3]
        gs = float(global_scale)
        if not np.isfinite(gs) or gs == 0:
            gs = 1.0
        vertices = (vertices - gc) / gs
        vertices = np.clip(vertices, -0.5 + 1e-6, 0.5 - 1e-6)
    else:
        if center is None:
            center = (vertices.max(0) + vertices.min(0)) / 2
        if scale is None:
            scale = (vertices.max(0) - vertices.min(0)).max()
        if scale == 0:
            scale = 1.0
        vertices = (vertices - center) / scale
    if post_normalize_transform_4x4 is not None:
        M = np.asarray(post_normalize_transform_4x4, dtype=np.float64)
        R = M[:3, :3]
        t = M[:3, 3]
        vertices = (R @ np.asarray(vertices, dtype=np.float64).T).T + t
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_size = 1.0 / resolution
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=voxel_size,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    voxels = torch.zeros(1, 1, resolution, resolution, resolution, dtype=torch.float32)
    for voxel in voxel_grid.get_voxels():
        idx = voxel.grid_index
        if (
            0 <= idx[0] < resolution
            and 0 <= idx[1] < resolution
            and 0 <= idx[2] < resolution
        ):
            voxels[0, 0, idx[0], idx[1], idx[2]] = 1.0
    return voxels


def _voxel_tensor_to_trellis_grid_layout(voxels_tensor: torch.Tensor) -> torch.Tensor:
    """
    Reindex occupancy with ``transform_voxels_to_trellis`` so (i,j,k) matches ``merging_data`` masks
    and Trellis-stored ``inpainted_voxels.pt`` (the same transform ``prepare_edit_data`` applies to masks).
    """
    from utils import transform_voxels_to_trellis

    vt = voxels_tensor.detach().cpu()
    if vt.dim() == 5:
        arr = vt[0, 0].numpy()
        out = torch.from_numpy(
            np.ascontiguousarray(transform_voxels_to_trellis(arr))
        ).float()
        return out.unsqueeze(0).unsqueeze(0)
    if vt.dim() == 4:
        arr = vt[0].numpy()
        out = torch.from_numpy(
            np.ascontiguousarray(transform_voxels_to_trellis(arr))
        ).float()
        return out.unsqueeze(0)
    arr = vt.numpy()
    return torch.from_numpy(
        np.ascontiguousarray(transform_voxels_to_trellis(arr))
    ).float()


def load_category_masks_from_merging_data(result_folder: Path) -> Optional[dict]:
    """
    Load voxel-resolution category masks from ``merging_data.pt`` (same indexing as inpainted voxels).

    Returns dict with keys ``unchanged``, ``changed``, ``added_deleted`` (values may be ``None``),
    or ``None`` if ``merging_data.pt`` is missing.
    """
    merging_data_path = result_folder / "merging_data.pt"
    if not merging_data_path.exists():
        return None
    merging_data = torch.load(merging_data_path, weights_only=False)
    unchanged_mask = merging_data.get("unchanged_mask_voxel")
    added_deleted_mask = merging_data.get("added_deleted_mask_voxel")
    changed_mask = None
    changed_edited_masks = merging_data.get("changed_edited_masks", [])
    if changed_edited_masks:
        first_mask = changed_edited_masks[0].get("mask_voxel")
        if first_mask is not None:
            changed_mask = np.zeros_like(first_mask, dtype=np.float32)
            for mask_data in changed_edited_masks:
                mask_voxel = mask_data.get("mask_voxel")
                if mask_voxel is not None:
                    changed_mask = np.logical_or(changed_mask, mask_voxel > 0.5).astype(
                        np.float32
                    )
    return {
        "unchanged": unchanged_mask,
        "changed": changed_mask,
        "added_deleted": added_deleted_mask,
    }


def load_merging_global_normalization(
    result_folder: Path,
) -> Optional[Tuple[np.ndarray, float]]:
    """
    Return ``(global_center, global_scale)`` from ``merging_data.pt`` — the same fields used
    when building masks in ``prepare_edit_data`` (unified scene bbox, not per-mesh).
    """
    merging_data_path = result_folder / "merging_data.pt"
    if not merging_data_path.exists():
        return None
    merging_data = torch.load(merging_data_path, weights_only=False)
    if "global_center" not in merging_data or "global_scale" not in merging_data:
        return None
    gc = np.asarray(merging_data["global_center"], dtype=np.float64).reshape(-1)[:3]
    gs = float(merging_data["global_scale"])
    return (gc, gs)


def dilate_category_masks_for_pointcloud(
    category_masks: dict,
    added_deleted_extra_dilation_iterations: int = 0,
    changed_extra_dilation_iterations: int = 0,
) -> dict:
    """
    Apply the same 3D binary dilations as ``_render_noisy_voxels_pointcloud_single`` uses for mask coloring.
    """
    from scipy.ndimage import binary_dilation, generate_binary_structure

    struct = generate_binary_structure(3, 1)
    out: dict = {}

    unchanged_mask = category_masks.get("unchanged")
    if unchanged_mask is not None:
        unchanged_mask = np.asarray(unchanged_mask)
        original_count = (unchanged_mask > 0.5).sum()
        unchanged_mask = binary_dilation(
            unchanged_mask > 0.5, structure=struct, iterations=2
        ).astype(np.float32)
        print(
            f"    Unchanged mask: {original_count} -> {(unchanged_mask > 0.5).sum()} voxels (after 2x dilation)"
        )
        out["unchanged"] = unchanged_mask
    else:
        out["unchanged"] = None

    changed_mask = category_masks.get("changed")
    if changed_mask is not None:
        changed_mask = np.asarray(changed_mask)
        original_count = (changed_mask > 0.5).sum()
        changed_mask = binary_dilation(
            changed_mask > 0.5, structure=struct, iterations=2
        ).astype(np.float32)
        print(
            f"    Changed mask: {original_count} -> {(changed_mask > 0.5).sum()} voxels (after 2x dilation)"
        )
        extra_ch = (
            int(changed_extra_dilation_iterations)
            if changed_extra_dilation_iterations
            else 0
        )
        if extra_ch > 0:
            n_before = (changed_mask > 0.5).sum()
            changed_mask = binary_dilation(
                changed_mask > 0.5, structure=struct, iterations=extra_ch
            ).astype(np.float32)
            print(
                f"    Changed mask: {n_before} -> {(changed_mask > 0.5).sum()} voxels "
                f"(+{extra_ch} extra dilation iter)"
            )
        out["changed"] = changed_mask
    else:
        out["changed"] = None

    added_deleted_mask = category_masks.get("added_deleted")
    if added_deleted_mask is not None:
        added_deleted_mask = np.asarray(added_deleted_mask)
        original_count = (added_deleted_mask > 0.5).sum()
        added_deleted_mask = binary_dilation(
            added_deleted_mask > 0.5, structure=struct, iterations=2
        ).astype(np.float32)
        print(
            f"    Added/deleted mask: {original_count} -> {(added_deleted_mask > 0.5).sum()} voxels (after 2x dilation)"
        )
        extra_ade = (
            int(added_deleted_extra_dilation_iterations)
            if added_deleted_extra_dilation_iterations
            else 0
        )
        if extra_ade > 0:
            n_before = (added_deleted_mask > 0.5).sum()
            added_deleted_mask = binary_dilation(
                added_deleted_mask > 0.5, structure=struct, iterations=extra_ade
            ).astype(np.float32)
            print(
                f"    Added/deleted mask: {n_before} -> {(added_deleted_mask > 0.5).sum()} voxels "
                f"(+{extra_ade} extra dilation iter)"
            )
        out["added_deleted"] = added_deleted_mask
    else:
        out["added_deleted"] = None

    return out


def render_binary_mask_volume_png(
    mask_bool_3d: np.ndarray,
    output_png: Path,
    uniform_color_rgb01: np.ndarray,
    skip_existing: bool,
) -> bool:
    """
    Render a single PNG of a boolean voxel mask (full grid), same Trellis export as noisy point-cloud voxels.
    Returns True if a PNG was written or already existed.
    """
    from utils import transform_voxels_to_trellis

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and output_png.is_file():
        return True
    mask_bool_3d = np.asarray(mask_bool_3d, dtype=bool)
    if mask_bool_3d.ndim != 3 or not (
        mask_bool_3d.shape[0] == mask_bool_3d.shape[1] == mask_bool_3d.shape[2]
    ):
        raise ValueError(f"mask_bool_3d must be a cube; got shape {mask_bool_3d.shape}")
    if not np.any(mask_bool_3d):
        print(f"  Warning: empty mask, skipping render for {output_png.name}")
        return False

    resolution = mask_bool_3d.shape[0]
    voxel_grid_binary = mask_bool_3d.astype(np.float32)
    color_grid = np.zeros((resolution, resolution, resolution, 3), dtype=np.float32)
    color_grid[mask_bool_3d] = uniform_color_rgb01

    voxel_grid_trellis = transform_voxels_to_trellis(voxel_grid_binary)
    color_grid_trellis = np.swapaxes(color_grid, 1, 2)
    color_grid_trellis = np.flip(color_grid_trellis, axis=1)

    ply_path = output_png.with_suffix(".ply")
    cube_mesh = voxels_to_colored_cube_mesh(voxel_grid_trellis, color_grid_trellis)
    o3d.io.write_triangle_mesh(str(ply_path), cube_mesh, write_vertex_colors=True)

    render_obj_with_blender(
        str(ply_path),
        str(output_png),
        rotation=(-90, 0, 0),
        invisible_ground=True,
        shade_smooth=False,
    )
    print(f"  Saved mask volume render: {output_png}")
    return True


def _apply_rigid_4x4_to_open3d_mesh(
    mesh: o3d.geometry.TriangleMesh, transform_4x4: np.ndarray
) -> None:
    """Apply ``transform_4x4`` (row-homogeneous convention: ``p' = (M @ p_hom.T).T``) to vertex positions in-place."""
    M = np.asarray(transform_4x4, dtype=np.float64)
    V = np.asarray(mesh.vertices)
    ones = np.ones((len(V), 1), dtype=np.float64)
    X = np.hstack([V, ones])
    Y = (M @ X.T).T[:, :3]
    mesh.vertices = o3d.utility.Vector3dVector(Y)
    mesh.compute_vertex_normals()


def _render_noisy_voxels_pointcloud_single(
    voxels_tensor: torch.Tensor,
    output_dir: Path,
    threshold: float,
    num_steps: int,
    skip_existing: bool,
    seed: int,
    noise_std: float,
    prefix: str = "noisy_pc",
    desc: str = "Rendering noisy point cloud voxels",
    category_masks: dict = None,
    color_override_changed: np.ndarray = None,
    color_override_added_deleted: np.ndarray = None,
    unmasked_uniform_color: np.ndarray = None,
    fixed_noise_positions: np.ndarray = None,
    added_deleted_extra_dilation_iterations: int = 0,
    changed_extra_dilation_iterations: int = 0,
    input_voxels_already_trellis: bool = False,
    display_mesh_transform_4x4: Optional[np.ndarray] = None,
) -> list:
    """
    Helper function to render noisy versions of a single voxel tensor.

    Args:
        voxels_tensor: Voxel tensor (will be moved to CPU and reshaped as needed)
        output_dir: Directory to save output images
        threshold: Threshold for voxel occupancy
        num_steps: Number of noise levels to render
        skip_existing: Skip rendering if output already exists
        seed: Random seed for reproducibility
        noise_std: Standard deviation of Gaussian noise (in voxel units)
        prefix: Filename prefix for output files
        desc: Description for progress bar
        category_masks: Optional dict with masks for coloring:
            - 'unchanged': mask for unchanged regions (orange)
            - 'changed': mask for changed/edited regions (blue)
            - 'added_deleted': mask for added/deleted regions (purple)
            All masks should be at voxel resolution (64^3) and already in Trellis orientation.
        input_voxels_already_trellis: If ``True``, ``voxels_tensor`` is already in Trellis grid layout
            (same as ``merging_data`` masks). Skip ``transform_voxels_to_trellis`` / swap-flip at export so the
            Blender mesh is not rotated twice (use after mesh voxelization + ``transform_voxels_to_trellis``).
        display_mesh_transform_4x4: Optional 4x4 rigid transform applied **only** to the colored voxel **mesh**
            after cube construction (does not change voxel indices or mask sampling). Use e.g.
            ``ORIENTATION_TRANSFORMS[NOISY_POINTCLOUD_MESH_VOXEL_ORIENT_IDX]`` to match legacy noisy
            point-cloud orientation while keeping mask-aligned voxelization.
        color_override_changed: Optional color to use for changed mask instead of blue (RGB 0-1)
        color_override_added_deleted: Optional color to use for added_deleted mask instead of purple (RGB 0-1)
        unmasked_uniform_color: If ``category_masks`` is None, use this RGB 0-1 for every voxel; default gray
        fixed_noise_positions: If set, per-active-voxel noise offsets (N,3); must match active voxel count; uses same
            noising path as a fresh run with ``seed`` / ``noise_std`` (for paired renders with identical motion)
        added_deleted_extra_dilation_iterations: After the default 2 iterations, run this many extra
            3D binary dilations on ``added_deleted`` (same structure as other mask dilations)
        changed_extra_dilation_iterations: After the default 2 iterations, run this many extra dilations
            on the ``changed`` (edited) mask.

    Returns:
        List of output image paths
    """
    from utils import transform_voxels_to_trellis

    output_dir.mkdir(parents=True, exist_ok=True)

    # Move to CPU for processing
    if voxels_tensor.is_cuda:
        voxels_tensor = voxels_tensor.cpu()

    # Handle different tensor shapes
    if voxels_tensor.dim() == 5:
        voxels_tensor = voxels_tensor[0, 0]  # Remove batch and channel dims
    elif voxels_tensor.dim() == 4:
        voxels_tensor = voxels_tensor[0]  # Remove batch dim

    resolution = voxels_tensor.shape[0]
    print(f"  Voxel resolution: {resolution}^3")

    # Get active voxel positions (indices where value > threshold)
    active_mask = (voxels_tensor > threshold).numpy()
    active_indices = np.argwhere(active_mask)  # Shape: (N, 3)
    n_active = len(active_indices)

    print(f"  Found {n_active} active voxels")

    if n_active == 0:
        print("  No active voxels found!")
        return []

    # Get masks if provided (already in Trellis orientation from merging_data)
    unchanged_mask = None
    changed_mask = None
    added_deleted_mask = None

    if category_masks is not None:
        category_masks = dilate_category_masks_for_pointcloud(
            category_masks,
            added_deleted_extra_dilation_iterations=added_deleted_extra_dilation_iterations,
            changed_extra_dilation_iterations=changed_extra_dilation_iterations,
        )
        unchanged_mask = category_masks.get("unchanged")
        changed_mask = category_masks.get("changed")
        added_deleted_mask = category_masks.get("added_deleted")

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Convert indices to continuous coordinates centered at origin
    # Center of grid: (resolution-1)/2 = 31.5 for 64^3 grid
    center = (resolution - 1) / 2.0
    original_positions = active_indices.astype(np.float32) - center  # Shape: (N, 3)

    print(
        f"  Original position range: [{original_positions.min():.2f}, {original_positions.max():.2f}]"
    )

    if fixed_noise_positions is not None:
        if fixed_noise_positions.shape != (n_active, 3):
            raise ValueError(
                f"fixed_noise_positions shape {fixed_noise_positions.shape} != ({n_active}, 3) for this voxel set"
            )
        noise_positions = np.asarray(fixed_noise_positions, dtype=np.float32)
    else:
        noise_positions = np.random.normal(0, noise_std, (n_active, 3)).astype(
            np.float32
        )
    print(
        f"  Noise std: {noise_std}, noise range: [{noise_positions.min():.2f}, {noise_positions.max():.2f}]"
    )

    output_paths = []

    # Generate timesteps from 0 to 1
    timesteps = np.linspace(0, 1, num_steps) * 0.8

    for i, t in enumerate(tqdm(timesteps, desc=desc)):
        # Output paths for this timestep
        ply_path = output_dir / f"{prefix}_{i:02d}_t{t:.3f}.ply"
        png_path = output_dir / f"{prefix}_{i:02d}_t{t:.3f}.png"

        if skip_existing and png_path.exists():
            output_paths.append(str(png_path))
            continue

        # Interpolate between original and noise positions
        # t=0: original positions, t=1: noise positions
        interpolated_positions = (1 - t) * original_positions + t * noise_positions

        # Convert back to voxel indices (quantize)
        voxel_indices = np.round(interpolated_positions + center).astype(np.int32)

        # Clip to valid range [0, resolution-1]
        voxel_indices = np.clip(voxel_indices, 0, resolution - 1)

        # Create new voxel grid and color grid
        noisy_voxels = np.zeros((resolution, resolution, resolution), dtype=np.float32)
        color_grid = np.zeros((resolution, resolution, resolution, 3), dtype=np.float32)

        # Set active voxels at their noisy positions
        for voxel_i, (idx, orig_idx) in enumerate(zip(voxel_indices, active_indices)):
            x, y, z = idx
            ox, oy, oz = orig_idx

            # Get value from original position
            if 0 <= ox < resolution and 0 <= oy < resolution and 0 <= oz < resolution:
                orig_value = (
                    voxels_tensor[ox, oy, oz].item()
                    if voxels_tensor[ox, oy, oz] > threshold
                    else 1.0
                )
            else:
                orig_value = 1.0

            # Only update if this voxel has higher value (handle collisions)
            if orig_value > noisy_voxels[x, y, z]:
                noisy_voxels[x, y, z] = orig_value

        # Now assign colors based on the NEW (noisy) positions in the masks
        # This happens AFTER all voxels are placed at their noisy positions
        occupied_indices = np.argwhere(noisy_voxels > threshold)

        # Determine effective colors (use overrides if provided)
        effective_color_changed = (
            color_override_changed
            if color_override_changed is not None
            else PCOLOR_BLUE
        )
        effective_color_added_deleted = (
            color_override_added_deleted
            if color_override_added_deleted is not None
            else PCOLOR_PURPLE
        )

        uniform_unmasked = (
            unmasked_uniform_color
            if unmasked_uniform_color is not None
            else PCOLOR_GRAY
        )
        for idx in occupied_indices:
            x, y, z = idx

            if category_masks is None:
                color_grid[x, y, z] = uniform_unmasked
            else:
                # Determine color based on NEW position in masks
                # Priority: added_deleted > changed > unchanged > default orange (not in any mask)
                if added_deleted_mask is not None and added_deleted_mask[x, y, z] > 0.5:
                    color_grid[x, y, z] = effective_color_added_deleted
                elif changed_mask is not None and changed_mask[x, y, z] > 0.5:
                    color_grid[x, y, z] = effective_color_changed
                elif unchanged_mask is not None and unchanged_mask[x, y, z] > 0.5:
                    color_grid[x, y, z] = PCOLOR_ORANGE
                else:
                    color_grid[x, y, z] = (
                        PCOLOR_ORANGE  # Not in any mask: same as unchanged
                    )

        num_voxels = (noisy_voxels > threshold).sum()

        # Create binary mask for occupied voxels
        voxel_grid_binary = (noisy_voxels > threshold).astype(np.float32)

        # Trellis orientation for Blender (matches inpainted / merging_data mask indexing when True)
        if input_voxels_already_trellis:
            voxel_grid_trellis = voxel_grid_binary
            color_grid_trellis = color_grid
        else:
            voxel_grid_trellis = transform_voxels_to_trellis(voxel_grid_binary)
            color_grid_trellis = np.swapaxes(color_grid, 1, 2)  # swap y and z
            color_grid_trellis = np.flip(color_grid_trellis, axis=1)  # flip new y axis

        # Create colored cube mesh
        cube_mesh = voxels_to_colored_cube_mesh(voxel_grid_trellis, color_grid_trellis)
        if display_mesh_transform_4x4 is not None:
            _apply_rigid_4x4_to_open3d_mesh(cube_mesh, display_mesh_transform_4x4)
        o3d.io.write_triangle_mesh(str(ply_path), cube_mesh, write_vertex_colors=True)

        # Render with invisible ground (no rotation needed - voxels are already in Trellis orientation)
        trellis_rotation = (-90, 0, 0)
        render_obj_with_blender(
            str(ply_path),
            str(png_path),
            rotation=trellis_rotation,
            invisible_ground=True,
            shade_smooth=False,  # faceted voxel cubes, not smooth blobs
        )

        output_paths.append(str(png_path))
        print(f"  t={t:.3f}: {num_voxels} voxels -> {png_path.name}")

    print(f"\nSaved {len(output_paths)} renders to: {output_dir}")
    return output_paths


def visualize_noisy_voxels_pointcloud(
    result_folder: str,
    output_dir: str,
    threshold: float,
    num_steps: int = 10,
    skip_existing: bool = True,
    seed: int = 42,
    noise_std: float = 15.0,
    only_modes: Optional[Set[str]] = None,
    added_deleted_extra_dilation: int = 0,
    changed_extra_dilation: int = 0,
):
    """
    Visualize noisy versions of both original and inpainted voxels using point cloud interpolation.

    Treats active voxels as a point cloud and interpolates between original positions
    and Gaussian noise sampled around the center.

    At t=0: Original voxel positions (no noise)
    At t=1: Random Gaussian positions centered at origin (full noise)

    Args:
        result_folder: Path to the tredit result folder
        output_dir: Directory to save output images
        threshold: Threshold for voxel occupancy
        num_steps: Number of noise levels to render
        skip_existing: Skip rendering if output already exists
        seed: Random seed for reproducibility
        noise_std: Standard deviation of Gaussian noise (in voxel units)
        only_modes: If not ``None``, only run named modes (see ``--noisy-pointcloud-modes``), e.g.
            ``{'inpainted_no_blue'}`` or ``{'mask_changed', 'mask_added_deleted'}``.
        added_deleted_extra_dilation: Extra dilation iterations for ``added_deleted`` after the default
            2, on renders that use category masks
        changed_extra_dilation: Same for the ``changed`` (edited) mask (after the default 2)

    Returns:
        Dict of output image path lists (only keys for modes that ran)
    """
    result_folder = Path(result_folder)
    output_dir = Path(output_dir)

    all_output_paths: dict = {}
    _np_vox_name, _np_vox_T = ORIENTATION_TRANSFORMS[
        NOISY_POINTCLOUD_MESH_VOXEL_ORIENT_IDX
    ]

    def _want(mode: str) -> bool:
        if only_modes is None:
            return True
        return mode in only_modes

    def _d_extra_kwargs():
        d = {}
        if added_deleted_extra_dilation and int(added_deleted_extra_dilation) > 0:
            d["added_deleted_extra_dilation_iterations"] = int(
                added_deleted_extra_dilation
            )
        if changed_extra_dilation and int(changed_extra_dilation) > 0:
            d["changed_extra_dilation_iterations"] = int(changed_extra_dilation)
        return d

    _mask_kws = _d_extra_kwargs()

    _modes_using_merge_masks = frozenset(
        {
            "inpainted",
            "inpainted_no_purple",
            "inpainted_no_blue",
            "inpainted_no_blue_purple",
            "edited_abstraction_categorized",
            "mask_changed",
            "mask_added_deleted",
        }
    )
    _should_load_merge_masks = only_modes is None or bool(
        only_modes & _modes_using_merge_masks
    )
    merging_category_masks_raw = (
        load_category_masks_from_merging_data(result_folder)
        if _should_load_merge_masks
        else None
    )
    if merging_category_masks_raw is not None:
        print(f"\nLoaded category masks from: {result_folder / 'merging_data.pt'}")
        _cm = merging_category_masks_raw
        if _cm.get("unchanged") is not None:
            print(
                f"  Unchanged mask: {(np.asarray(_cm['unchanged']) > 0.5).sum()} voxels"
            )
        if _cm.get("changed") is not None:
            print(f"  Changed mask: {(np.asarray(_cm['changed']) > 0.5).sum()} voxels")
        if _cm.get("added_deleted") is not None:
            print(
                f"  Added/deleted mask: {(np.asarray(_cm['added_deleted']) > 0.5).sum()} voxels"
            )

    # === Process ORIGINAL voxels (gray + orange, shared noise trajectory) ===
    original_voxels_path = result_folder / "inversion" / "original_shape_voxels.pt"
    if original_voxels_path.exists() and (
        _want("original") or _want("original_orange")
    ):
        print("\n=== Processing ORIGINAL voxels (gray + orange) ===")
        print(f"Loading from: {original_voxels_path}")
        original_voxels = torch.load(original_voxels_path, weights_only=False)
        pair_both = _want("original") and _want("original_orange")
        shared_noise = (
            _pointcloud_noise_for_voxel_tensor(
                original_voxels, threshold, seed, noise_std
            )
            if pair_both
            else None
        )
        if _want("original"):
            original_output_dir = output_dir / "original"
            all_output_paths["original"] = _render_noisy_voxels_pointcloud_single(
                voxels_tensor=original_voxels,
                output_dir=original_output_dir,
                threshold=threshold,
                num_steps=num_steps,
                skip_existing=skip_existing,
                seed=seed,
                noise_std=noise_std,
                prefix="noisy_pc",
                desc="Rendering noisy original voxels (gray)",
                fixed_noise_positions=shared_noise,
            )
        if _want("original_orange"):
            original_orange_dir = output_dir / "original_orange"
            all_output_paths["original_orange"] = (
                _render_noisy_voxels_pointcloud_single(
                    voxels_tensor=original_voxels,
                    output_dir=original_orange_dir,
                    threshold=threshold,
                    num_steps=num_steps,
                    skip_existing=skip_existing,
                    seed=seed,
                    noise_std=noise_std,
                    prefix="noisy_pc",
                    desc="Rendering noisy original voxels (orange)",
                    unmasked_uniform_color=PCOLOR_ORANGE,
                    fixed_noise_positions=shared_noise,
                )
            )
    elif not _want("original") and not _want("original_orange"):
        pass
    else:
        print(f"Warning: Original voxels not found: {original_voxels_path}")

    # === Process INPAINTED voxels ===
    _inpaint_mode_keys = frozenset(
        {
            "inpainted",
            "inpainted_gray",
            "inpainted_no_purple",
            "inpainted_no_blue",
            "inpainted_no_blue_purple",
        }
    )
    _need_inpaint = only_modes is None or bool(only_modes & _inpaint_mode_keys)
    inpainted_voxels_path = result_folder / "inpainted_voxels.pt"
    if inpainted_voxels_path.exists() and _need_inpaint:
        print("\n=== Processing INPAINTED voxels ===")
        print(f"Loading from: {inpainted_voxels_path}")
        inpainted_voxels = torch.load(inpainted_voxels_path, weights_only=False)

        category_masks = merging_category_masks_raw
        if category_masks is None:
            print("  Warning: merging_data.pt not found, using uniform gray color")

        if _want("inpainted"):
            # Render inpainted voxels with category mask colors
            inpainted_output_dir = output_dir / "inpainted"
            all_output_paths["inpainted"] = _render_noisy_voxels_pointcloud_single(
                voxels_tensor=inpainted_voxels,
                output_dir=inpainted_output_dir,
                threshold=threshold,
                num_steps=num_steps,
                skip_existing=skip_existing,
                seed=seed,
                noise_std=noise_std,
                prefix="noisy_pc",
                desc="Rendering noisy inpainted voxels (colored)",
                category_masks=category_masks,
                **_mask_kws,
            )

        if _want("inpainted_gray"):
            # Render inpainted voxels again in uniform gray (no category masks)
            inpainted_gray_output_dir = output_dir / "inpainted_gray"
            all_output_paths["inpainted_gray"] = _render_noisy_voxels_pointcloud_single(
                voxels_tensor=inpainted_voxels,
                output_dir=inpainted_gray_output_dir,
                threshold=threshold,
                num_steps=num_steps,
                skip_existing=skip_existing,
                seed=seed,
                noise_std=noise_std,
                prefix="noisy_pc",
                desc="Rendering noisy inpainted voxels (gray)",
                category_masks=None,  # No masks = uniform gray
            )

        if _want("inpainted_no_purple"):
            inpainted_no_purple_output_dir = output_dir / "inpainted_no_purple"
            all_output_paths["inpainted_no_purple"] = (
                _render_noisy_voxels_pointcloud_single(
                    voxels_tensor=inpainted_voxels,
                    output_dir=inpainted_no_purple_output_dir,
                    threshold=threshold,
                    num_steps=num_steps,
                    skip_existing=skip_existing,
                    seed=seed,
                    noise_std=noise_std,
                    prefix="noisy_pc",
                    desc="Rendering noisy inpainted voxels (added_deleted as gray)",
                    category_masks=category_masks,
                    color_override_added_deleted=PCOLOR_GRAY,  # Purple -> gray
                    **_mask_kws,
                )
            )

        if _want("inpainted_no_blue"):
            inpainted_no_blue_output_dir = output_dir / "inpainted_no_blue"
            all_output_paths["inpainted_no_blue"] = (
                _render_noisy_voxels_pointcloud_single(
                    voxels_tensor=inpainted_voxels,
                    output_dir=inpainted_no_blue_output_dir,
                    threshold=threshold,
                    num_steps=num_steps,
                    skip_existing=skip_existing,
                    seed=seed,
                    noise_std=noise_std,
                    prefix="noisy_pc",
                    desc="Rendering noisy inpainted voxels (changed as gray, no blue)",
                    category_masks=category_masks,
                    color_override_changed=PCOLOR_GRAY,  # Blue -> gray
                    **_mask_kws,
                )
            )

        if _want("inpainted_no_blue_purple"):
            inpainted_no_blue_purple_output_dir = (
                output_dir / "inpainted_no_blue_purple"
            )
            all_output_paths["inpainted_no_blue_purple"] = (
                _render_noisy_voxels_pointcloud_single(
                    voxels_tensor=inpainted_voxels,
                    output_dir=inpainted_no_blue_purple_output_dir,
                    threshold=threshold,
                    num_steps=num_steps,
                    skip_existing=skip_existing,
                    seed=seed,
                    noise_std=noise_std,
                    prefix="noisy_pc",
                    desc="Rendering noisy inpainted voxels (changed & added_deleted as gray)",
                    category_masks=category_masks,
                    color_override_changed=PCOLOR_GRAY,  # Blue -> gray
                    color_override_added_deleted=PCOLOR_GRAY,  # Purple -> gray
                    **_mask_kws,
                )
            )
    elif not _need_inpaint:
        pass
    else:
        print(f"Warning: Inpainted voxels not found: {inpainted_voxels_path}")

    # === Voxelized warped mesh (blue) — same schedule; mesh from transformed_meshes ===
    transformed_dir = result_folder / "transformed_meshes"
    mesh_path = None
    if transformed_dir.is_dir():
        preferred = transformed_dir / "transformed_mesh_sq2.obj"
        if preferred.is_file():
            mesh_path = preferred
        else:
            for p in sorted(transformed_dir.glob("*.obj")) + sorted(
                transformed_dir.glob("*.ply")
            ):
                mesh_path = p
                break
    if mesh_path is not None and mesh_path.is_file() and _want("warped_mesh"):
        print("\n=== Noisy point cloud: warped mesh (voxelized) ===")
        print(f"  Mesh: {mesh_path.name}")
        print(
            f"  Voxel pre-transform: ORIENTATION_TRANSFORMS[{NOISY_POINTCLOUD_MESH_VOXEL_ORIENT_IDX}] "
            f"({_np_vox_name})"
        )
        try:
            vox_mesh = _voxelize_mesh_path_cpu(
                str(mesh_path), post_normalize_transform_4x4=_np_vox_T
            )
            if (vox_mesh > threshold).any():
                all_output_paths["warped_mesh"] = (
                    _render_noisy_voxels_pointcloud_single(
                        voxels_tensor=vox_mesh,
                        output_dir=output_dir / "warped_mesh",
                        threshold=threshold,
                        num_steps=num_steps,
                        skip_existing=skip_existing,
                        seed=seed,
                        noise_std=noise_std,
                        prefix="noisy_pc",
                        desc="Rendering noisy warped mesh voxels (blue)",
                        unmasked_uniform_color=PCOLOR_BLUE,
                    )
                )
            else:
                print(
                    "  Warning: Voxelized warped mesh is empty, skipping warped_mesh point cloud"
                )
        except Exception as e:
            print(f"  Warning: Could not voxelize/render warped mesh: {e}")
    elif _want("warped_mesh"):
        print(
            "  (No transformed_meshes/*.obj|ply; skipping warped mesh noisy point cloud)"
        )

    # === Voxelized edited abstraction (purple) — same as mesh_from_abstraction in categorized view ===
    edited_json = None
    for name in (
        "edited_abstraction_final.json",
        "edited_abstraction.json",
        "edited_abstraction_iter1.json",
    ):
        p = result_folder / name
        if p.is_file():
            edited_json = p
            break
    if edited_json is not None and (
        _want("edited_abstraction") or _want("edited_abstraction_categorized")
    ):
        print("\n=== Noisy point cloud: edited abstraction (voxelized) ===")
        print(f"  {edited_json.name}")
        if _want("edited_abstraction"):
            print(
                f"  edited_abstraction: voxel pre-transform ORIENTATION_TRANSFORMS[{NOISY_POINTCLOUD_MESH_VOXEL_ORIENT_IDX}] "
                f"({_np_vox_name}), per-mesh bbox normalize"
            )
        if _want("edited_abstraction_categorized"):
            print(
                "  edited_abstraction_categorized: merge global_center/global_scale (mask-aligned voxelization); "
                "Trellis grid remap; ORIENTATION_TRANSFORMS applied only to the final voxel mesh for Blender "
                f"(same index as noisy PC: {_np_vox_name})"
            )
        try:
            from abstraction import load_abstraction, mesh_from_abstraction

            edited = load_abstraction(str(edited_json))
            abs_mesh = mesh_from_abstraction(edited, resolution=30)
            if abs_mesh is None or abs_mesh.is_empty:
                print("  Warning: mesh_from_abstraction returned empty mesh, skipping")
            else:
                tmp_abs = output_dir / "_tmp_edited_abstraction_for_noisy_pc.obj"
                abs_mesh.export(str(tmp_abs), file_type="obj")
                try:
                    merge_norm = (
                        load_merging_global_normalization(result_folder)
                        if _want("edited_abstraction_categorized")
                        else None
                    )
                    merge_voxel_resolution = 64
                    if (
                        _want("edited_abstraction_categorized")
                        and merge_norm is not None
                    ):
                        gc, gs = merge_norm
                        md_full = torch.load(
                            result_folder / "merging_data.pt", weights_only=False
                        )
                        merge_voxel_resolution = int(
                            md_full.get("voxel_resolution", 64)
                        )
                        print(
                            f"    merging_data normalization: center={gc}, scale={gs}, "
                            f"voxel_resolution={merge_voxel_resolution}"
                        )
                    elif _want("edited_abstraction_categorized"):
                        print(
                            "  Warning: merging_data.pt missing global_center/global_scale; "
                            "edited_abstraction_categorized would misalign masks — skipping"
                        )

                    vox_abs_legacy = None
                    vox_abs_merge = None
                    if _want("edited_abstraction"):
                        vox_abs_legacy = _voxelize_mesh_path_cpu(
                            str(tmp_abs), post_normalize_transform_4x4=_np_vox_T
                        )
                    if (
                        _want("edited_abstraction_categorized")
                        and merge_norm is not None
                    ):
                        gc, gs = merge_norm
                        # Must NOT apply ORIENTATION_TRANSFORMS here: merging_data masks are built from the same
                        # global normalization without a vertex rotation (inpaint_data_preparation.prepare_edit_data).
                        vox_abs_merge = _voxelize_mesh_path_cpu(
                            str(tmp_abs),
                            resolution=merge_voxel_resolution,
                            post_normalize_transform_4x4=None,
                            global_center=gc,
                            global_scale=gs,
                        )

                    if (
                        _want("edited_abstraction")
                        and vox_abs_legacy is not None
                        and (vox_abs_legacy > threshold).any()
                    ):
                        all_output_paths["edited_abstraction"] = (
                            _render_noisy_voxels_pointcloud_single(
                                voxels_tensor=vox_abs_legacy,
                                output_dir=output_dir / "edited_abstraction",
                                threshold=threshold,
                                num_steps=num_steps,
                                skip_existing=skip_existing,
                                seed=seed,
                                noise_std=noise_std,
                                prefix="noisy_pc",
                                desc="Rendering noisy edited abstraction voxels (purple)",
                                unmasked_uniform_color=PCOLOR_PURPLE,
                            )
                        )
                    elif _want("edited_abstraction") and vox_abs_legacy is not None:
                        print(
                            "  Warning: Voxelized edited abstraction (legacy) is empty, skipping"
                        )

                    if _want("edited_abstraction_categorized"):
                        if merging_category_masks_raw is None:
                            print(
                                "  Warning: merging_data.pt not found; skipping edited_abstraction_categorized"
                            )
                        elif merge_norm is None:
                            pass
                        elif vox_abs_merge is None:
                            pass
                        elif not (vox_abs_merge > threshold).any():
                            print(
                                "  Warning: Voxelized edited abstraction (merge-normalized) is empty, skipping"
                            )
                        else:
                            vox_cat_trellis = _voxel_tensor_to_trellis_grid_layout(
                                vox_abs_merge
                            )
                            all_output_paths["edited_abstraction_categorized"] = (
                                _render_noisy_voxels_pointcloud_single(
                                    voxels_tensor=vox_cat_trellis,
                                    output_dir=output_dir
                                    / "edited_abstraction_categorized",
                                    threshold=threshold,
                                    num_steps=num_steps,
                                    skip_existing=skip_existing,
                                    seed=seed,
                                    noise_std=noise_std,
                                    prefix="noisy_pc",
                                    desc="Rendering noisy edited abstraction voxels (inpainted mask colors)",
                                    category_masks=merging_category_masks_raw,
                                    input_voxels_already_trellis=True,
                                    display_mesh_transform_4x4=_np_vox_T,
                                    **_mask_kws,
                                )
                            )
                finally:
                    if tmp_abs.exists():
                        tmp_abs.unlink()
        except Exception as e:
            print(f"  Warning: Could not build/voxelize edited abstraction: {e}")
    elif _want("edited_abstraction") or _want("edited_abstraction_categorized"):
        print(
            "  (No edited_abstraction_*.json; skipping edited abstraction noisy point cloud)"
        )

    # === Full merging masks as separate volumes (same dilation as inpainted coloring) ===
    if _want("mask_changed") or _want("mask_added_deleted"):
        print(
            "\n=== Mask volume renders (merging_data; dilated like inpainted coloring) ==="
        )
        if merging_category_masks_raw is None:
            print(
                "  Warning: merging_data.pt not found; skipping mask_changed / mask_added_deleted"
            )
        else:
            dilated_masks = dilate_category_masks_for_pointcloud(
                merging_category_masks_raw, **_mask_kws
            )
            if _want("mask_changed"):
                ch = dilated_masks.get("changed")
                if ch is None:
                    print(
                        "  Warning: no changed mask in merging_data; skipping mask_changed"
                    )
                else:
                    outp = output_dir / "mask_changed" / "mask_volume.png"
                    if render_binary_mask_volume_png(
                        ch > 0.5, outp, PCOLOR_BLUE, skip_existing
                    ):
                        all_output_paths["mask_changed"] = [str(outp)]
            if _want("mask_added_deleted"):
                ad = dilated_masks.get("added_deleted")
                if ad is None:
                    print(
                        "  Warning: no added_deleted mask in merging_data; skipping mask_added_deleted"
                    )
                else:
                    outp = output_dir / "mask_added_deleted" / "mask_volume.png"
                    if render_binary_mask_volume_png(
                        ad > 0.5, outp, PCOLOR_PURPLE, skip_existing
                    ):
                        all_output_paths["mask_added_deleted"] = [str(outp)]

    return all_output_paths


def visualize_noisy_voxels(
    result_folder: str,
    output_dir: str,
    threshold: float,
    num_steps: int = 10,
    skip_existing: bool = True,
    seed: int = 42,
):
    """
    Visualize noisy versions of the original voxels at different noise levels.

    At each timestep t (from 0 to 1), interpolates between original voxels and random noise:
    noisy_voxels = (1 - t) * original_voxels + t * noise

    Args:
        result_folder: Path to the tredit result folder
        output_dir: Directory to save output images
        threshold: Threshold for voxel occupancy
        num_steps: Number of noise levels to render
        skip_existing: Skip rendering if output already exists
        seed: Random seed for reproducibility

    Returns:
        List of output image paths
    """
    result_folder = Path(result_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original voxels
    voxels_path = result_folder / "inversion" / "original_shape_voxels.pt"
    if not voxels_path.exists():
        print(f"Error: Voxels file not found: {voxels_path}")
        return []

    print(f"Loading original voxels from: {voxels_path}")
    original_voxels = torch.load(voxels_path, weights_only=False)

    # Move to CPU for processing
    if original_voxels.is_cuda:
        original_voxels = original_voxels.cpu()

    # Handle different tensor shapes
    if original_voxels.dim() == 5:
        original_voxels = original_voxels[0, 0]  # Remove batch and channel dims
    elif original_voxels.dim() == 4:
        original_voxels = original_voxels[0]  # Remove batch dim

    resolution = original_voxels.shape[0]
    print(f"  Voxel resolution: {resolution}^3")
    print(
        f"  Original voxel range: [{original_voxels.min():.3f}, {original_voxels.max():.3f}]"
    )

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate base noise grid (Gaussian, scaled to [-1, 1])
    base_noise = torch.randn(resolution, resolution, resolution)
    base_noise = base_noise / 3  # Map roughly [-3, 3] to [-1, 1]
    base_noise = base_noise.clamp(-1, 1)

    # Get binary mask of original voxels
    original_mask = (original_voxels > 0.5).float()

    # Rotation to align with standard view
    trellis_rotation = (-90, 0, 0)

    output_paths = []

    # Generate timesteps from 0 to 1
    timesteps = np.linspace(0, 1, num_steps)

    # Max dilation radius at t=1
    max_dilation = 8

    for i, t in enumerate(tqdm(timesteps, desc="Rendering noisy voxels")):
        # Output paths for this timestep
        ply_path = output_dir / f"noisy_{i:02d}_t{t:.3f}.ply"
        png_path = output_dir / f"noisy_{i:02d}_t{t:.3f}.png"

        if skip_existing and png_path.exists():
            output_paths.append(str(png_path))
            continue

        # Compute dilation radius based on timestep
        dilation_radius = int(t * max_dilation)

        # Dilate the original mask
        if dilation_radius > 0:
            from scipy.ndimage import binary_dilation, generate_binary_structure

            # Create a spherical structuring element
            struct = generate_binary_structure(3, 1)  # 3D cross
            dilated_mask = binary_dilation(
                original_mask.numpy() > 0.5,
                structure=struct,
                iterations=dilation_radius,
            )
            dilated_mask = torch.from_numpy(dilated_mask.astype(np.float32))
        else:
            dilated_mask = original_mask.clone()

        # Create noisy voxels:
        # 1. Start with zeros
        # 2. Within dilated mask, add noise
        # 3. Inject original voxels where they are > 0.5
        noisy_voxels = torch.zeros_like(original_voxels)
        noisy_voxels[dilated_mask > 0.5] = base_noise[dilated_mask > 0.5]
        noisy_voxels[original_mask > 0.5] = original_voxels[original_mask > 0.5]

        # Check if any voxels are occupied
        num_voxels = (noisy_voxels > threshold).sum().item()
        if num_voxels == 0:
            print(f"  t={t:.3f}: No voxels occupied, skipping")
            continue

        # Save as PLY (need to add batch/channel dims for save_voxels_as_ply)
        noisy_voxels_5d = noisy_voxels.unsqueeze(0).unsqueeze(0)
        save_voxels_as_ply(noisy_voxels_5d, ply_path, threshold=threshold)

        # Render with invisible ground
        render_obj_with_blender(
            str(ply_path),
            str(png_path),
            rotation=trellis_rotation,
            invisible_ground=True,
        )

        output_paths.append(str(png_path))
        print(f"  t={t:.3f}: {num_voxels} voxels -> {png_path.name}")

    print(f"\nSaved {len(output_paths)} renders to: {output_dir}")
    return output_paths


def visualize_abstraction_voxels(
    result_folder: str,
    output_dir: str,
    skip_existing: bool = True,
):
    """
    Voxelize and render the edited abstraction with superquadric colors.

    Args:
        result_folder: Path to the tredit result folder
        output_dir: Directory to save output images
        skip_existing: Skip rendering if output already exists

    Returns:
        Path to output image
    """
    from utils import transform_voxels_to_trellis

    result_folder = Path(result_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find edited_final.obj
    mesh_path = result_folder / "edited_final.obj"
    if not mesh_path.exists():
        print(f"Error: Mesh file not found: {mesh_path}")
        return None

    # Load normalization from inversion
    normalization_path = result_folder / "inversion" / "normalization.pt"
    if normalization_path.exists():
        normalization = torch.load(normalization_path, weights_only=False)
        center = normalization["center"]
        scale = normalization["scale"]
        print(f"  Loaded normalization: center={center}, scale={scale}")
    else:
        print("  Warning: normalization.pt not found, computing from mesh")
        center = None
        scale = None

    ply_path = output_dir / "abstraction_voxels.ply"
    png_path = output_dir / "abstraction_voxels.png"

    if skip_existing and png_path.exists():
        print(f"Skipping existing: {png_path}")
        return str(png_path)

    print(f"Voxelizing mesh: {mesh_path}")

    # Voxelize with colors using the same normalization as inversion
    voxel_grid, color_grid = voxelize_mesh_with_colors(
        str(mesh_path),
        resolution=64,
        center=center,
        scale=scale,
    )

    num_voxels = (voxel_grid > 0.5).sum()
    print(f"  Voxelized to {num_voxels} voxels")

    # Transform voxels to Trellis orientation (same as inversion.py)
    # Need to transform both voxel grid and color grid in the same way
    voxel_grid_trellis = transform_voxels_to_trellis(voxel_grid)
    # Apply same transformation to color grid: swap axes 1,2 then flip axis 1
    color_grid_trellis = np.swapaxes(color_grid, 0, 1)  # swap x and y
    color_grid_trellis = np.swapaxes(
        color_grid_trellis, 1, 2
    )  # swap y and z (now x and z swapped)
    # Actually transform_voxels_to_trellis does: swap axis 1,2 then flip axis 1
    # For color grid (res, res, res, 3), we need to do the same on first 3 dims
    color_grid_trellis = np.swapaxes(color_grid, 1, 2)  # swap y and z
    color_grid_trellis = np.flip(color_grid_trellis, axis=1)  # flip new y axis

    # Create colored cube mesh
    cube_mesh = voxels_to_colored_cube_mesh(voxel_grid_trellis, color_grid_trellis)

    # Save PLY
    o3d.io.write_triangle_mesh(str(ply_path), cube_mesh, write_vertex_colors=True)
    print(f"  Saved PLY: {ply_path}")

    # Render with invisible ground
    render_obj_with_blender(str(ply_path), str(png_path), invisible_ground=True)
    print(f"  Saved render: {png_path}")

    return str(png_path)


def _vlm_slide_default_pose_rotation_candidates() -> list:
    """
    3x3 rotation matrices applied to mesh vertices in Blender for the **default camera**
    (``azim=70°``, ``elev=20°``): same spirit as ``_teaser_input_shape_view_candidates`` and
    dataset abstraction orientations, so you can pick a correct frame when raw SQ meshes look tilted.
    """

    def _lab(x: str) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(x))

    out: list = []
    out.append(("00_identity", np.eye(3)))
    rb_glb = _blender_euler_xyz_deg_to_matrix_3x3(-90, 0, 0)
    # +90° X is the default for all VLM base views (``VLM_SLIDE_BASE_VERTEX_ROTATION``), not listed here
    out.append(("01_R_neg90_x", rb_glb))
    out.append(("02_R_0_neg90_y", _blender_euler_xyz_deg_to_matrix_3x3(0, -90, 0)))
    out.append(("03_R_0_0_neg90_z", _blender_euler_xyz_deg_to_matrix_3x3(0, 0, -90)))
    _on, mat_t = ORIENTATION_TRANSFORMS[TEASER_ABSTRACTION_VIEW_ORIENT_IDX]
    out.append(
        (
            f"04_teaser_abstraction_view_{TEASER_ABSTRACTION_VIEW_ORIENT_IDX}_"
            + _lab(_on),
            np.asarray(mat_t[:3, :3], dtype=np.float64),
        )
    )
    oa = teaser_mesh_orientation("original_abstraction")
    if oa is not None:
        oname, mfull = ORIENTATION_TRANSFORMS[oa]
        out.append(
            (
                f"05_dataset_abstraction_{oa}_" + _lab(oname),
                np.asarray(mfull[:3, :3], dtype=np.float64),
            )
        )
    return out


def render_vlm_slide(
    result_folder: str,
    output_dir: str,
    skip_existing: bool = True,
    mesh_resolution: int = 30,
    default_pose_sweep: bool = False,
) -> dict:
    """
    Render every abstraction JSON in the result folder with **RGB colors taken from the JSON**
    (no category recoloring).

    Files included:

    - ``abstraction.json`` (original extraction), if present, listed first
    - every ``edited_abstraction*.json`` (e.g. ``edited_abstraction_final.json``,
      ``edited_abstraction_iter1.json``, …), sorted by name

    For each file, writes under ``output_dir / <json_stem> /``:

    - **All five base views** use a **+90° X** vertex rotation in Blender (``VLM_SLIDE_BASE_VERTEX_ROTATION``,
      the former sweep option ``01_R_pos90_x``), then the usual camera:
    - ``default.png`` — ``azim=70°``, ``elev=20°``
    - ``left.png``, ``front.png``, ``right.png``, ``back.png`` — same azimuth / elevation
      as :func:`abstraction.render_multiview` (``elev=10°``; **left** / **front** / **right** / **back**
      at ``azim`` 0° / 90° / 180° / 270°)
    - If ``default_pose_sweep`` is True: ``default_pose_sweep/default__<label>.png`` — optional
      extra **vertex rotation** candidates (default camera on each) for comparison; the production
      base views no longer include +90° X in this sweep because it is always applied above.

    Args:
        result_folder: Tredit result directory containing abstraction JSON files
        output_dir: Directory for a ``vlm_slide/``-style tree (caller may pass ``.../vlm_slide``)
        skip_existing: If True, skip any view whose PNG already exists
        mesh_resolution: Superquadric mesh resolution passed to :func:`abstraction.mesh_from_abstraction`
        default_pose_sweep: When True, render the extra ``default_pose_sweep/`` exploration images

    Returns:
        Map from JSON filename stem to ``{"base_views": [...5 paths...], "default_pose_sweep": [...]}``
    """
    from abstraction import load_abstraction, mesh_from_abstraction

    result_folder = Path(result_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_paths: list = []
    p_abs = result_folder / "abstraction.json"
    if p_abs.is_file():
        json_paths.append(p_abs)
    _seen = {p.resolve() for p in json_paths}
    for p in sorted(result_folder.glob("edited_abstraction*.json")):
        r = p.resolve()
        if r not in _seen:
            json_paths.append(p)
            _seen.add(r)

    if not json_paths:
        print(
            f"render_vlm_slide: no JSON found under {result_folder} "
            f"(expected abstraction.json and/or edited_abstraction*.json)"
        )
        return {}

    # Match inference single render (utils defaults) + abstraction.render_multiview cardinals
    _common = dict(
        dist=2.0,
        light_energy=2.5,
        invisible_ground=True,
        shade_smooth=False,
    )
    poses: list = [
        ("default", {"azim": 70.0, "elev": 20.0}),
        ("left", {"azim": 0.0, "elev": 10.0}),
        ("front", {"azim": 90.0, "elev": 10.0}),
        ("right", {"azim": 180.0, "elev": 10.0}),
        ("back", {"azim": 270.0, "elev": 10.0}),
    ]

    all_out: dict = {}
    sweep_cand = (
        _vlm_slide_default_pose_rotation_candidates() if default_pose_sweep else []
    )
    for json_path in json_paths:
        stem = json_path.stem
        sub = output_dir / stem
        sub.mkdir(parents=True, exist_ok=True)
        sweep_dir = sub / "default_pose_sweep"
        base_expected = [sub / f"{name}.png" for name, _ in poses]
        sweep_expected = (
            [sweep_dir / f"default__{lab}.png" for lab, _ in sweep_cand]
            if sweep_cand
            else []
        )
        base_ok = all(f.is_file() for f in base_expected)
        sweep_ok = (
            (not default_pose_sweep)
            or (not sweep_expected)
            or all(f.is_file() for f in sweep_expected)
        )
        if skip_existing and base_ok and sweep_ok:
            print(f"render_vlm_slide: skipping (all views + sweep exist): {stem}")
            all_out[stem] = {
                "base_views": [str(f) for f in base_expected],
                "default_pose_sweep": [str(f) for f in sweep_expected]
                if sweep_expected
                else [],
            }
            continue

        print(f"render_vlm_slide: {json_path.name} -> {sub}")
        try:
            abstraction = load_abstraction(str(json_path))
        except Exception as e:
            print(f"  Error loading JSON: {e}")
            continue

        mesh = mesh_from_abstraction(abstraction, resolution=mesh_resolution)
        if mesh is None or (hasattr(mesh, "is_empty") and mesh.is_empty):
            print("  Warning: empty mesh, skipping")
            continue

        ply_path = sub / "_vlm_slide_temp_mesh.ply"
        mesh.export(str(ply_path), file_type="ply")
        out_list: list = []
        sweep_paths: list = []
        try:
            print(
                "  base views: vertex rotation +90° X (VLM_SLIDE_BASE_VERTEX_ROTATION)"
            )
            for name, extra in poses:
                out_png = sub / f"{name}.png"
                if skip_existing and out_png.is_file():
                    print(f"  skip existing {name}.png")
                    out_list.append(str(out_png))
                    continue
                render_obj_with_blender(
                    str(ply_path),
                    str(out_png),
                    rotation_matrix=VLM_SLIDE_BASE_VERTEX_ROTATION,
                    **{**_common, **extra},
                )
                print(f"  wrote {name}.png")
                out_list.append(str(out_png))
            if default_pose_sweep and sweep_cand:
                sweep_dir.mkdir(parents=True, exist_ok=True)
                default_cam = {"azim": 70.0, "elev": 20.0}
                print(
                    f"  default_pose_sweep: {len(sweep_cand)} candidate orientations -> {sweep_dir.name}/"
                )
                for lab, rmat in sweep_cand:
                    out_png = sweep_dir / f"default__{lab}.png"
                    if skip_existing and out_png.is_file():
                        print(f"  skip existing {out_png.name}")
                        sweep_paths.append(str(out_png))
                        continue
                    render_obj_with_blender(
                        str(ply_path),
                        str(out_png),
                        rotation_matrix=rmat,
                        **{**_common, **default_cam},
                    )
                    print(f"  wrote {out_png.name}")
                    sweep_paths.append(str(out_png))
        finally:
            if ply_path.is_file():
                try:
                    ply_path.unlink()
                except OSError:
                    pass

        all_out[stem] = {
            "base_views": out_list,
            "default_pose_sweep": sweep_paths,
        }

    return all_out


def visualize_categorized_abstraction(
    result_folder: str,
    output_dir: str,
    skip_existing: bool = True,
    color_unchanged: list = None,
    color_changed: list = None,
    color_added_deleted: list = None,
    include_unit_sq: bool = False,
    categorized_mesh_render_resolution: int = 1024,
):
    """
    Visualize the edited abstraction with colors based on edit categories:
    - Unchanged: orange (default; same RGB as noisy-pointcloud unchanged / ``PCOLOR_ORANGE``)
    - Changed/edited: blue (default)
    - Added/deleted: purple (default)

    Renders both the mesh and voxelized versions. When ``original_slat.glb`` exists, also writes
    ``original_slat_gray.png``: the slat mesh in the same light gray ``[180,180,180]`` as
    ``original_abstraction_gray.png``.

    Args:
        result_folder: Path to the tredit result folder
        output_dir: Directory to save output images
        skip_existing: Skip rendering if output already exists
        color_unchanged: RGB color [0-255] for unchanged parts (default: 190,120,40 orange)
        color_changed: RGB color [0-255] for changed/edited parts (default: [40, 90, 180])
        color_added_deleted: RGB color [0-255] for added/deleted parts (default: [150, 50, 170])
        include_unit_sq: If True, also render ``unit_sq.ply`` / ``unit_sq.png``: a single canonical
            superquadric (isotropic scale 0.1, identity rotation, no translation) in the **changed** blue, with
            the same Blender settings as ``categorized_abstraction`` (mesh, not voxels).
        categorized_mesh_render_resolution: Width and height in pixels for ``categorized_abstraction.png``
            (recolored superquadric mesh). Other categorized outputs still use the default 512 unless changed later.

    Returns:
        Dict with paths to output images
    """
    from abstraction import mesh_from_abstraction, load_abstraction
    from inpaint_data_preparation import compare_abstractions
    from utils import transform_voxels_to_trellis

    result_folder = Path(result_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find abstraction files
    original_json = result_folder / "abstraction.json"

    # Try different possible names for edited abstraction
    edited_json = None
    for name in [
        "edited_abstraction_final.json",
        "edited_abstraction.json",
        "edited_abstraction_iter1.json",
    ]:
        candidate = result_folder / name
        if candidate.exists():
            edited_json = candidate
            break

    if not original_json.exists():
        print(f"Error: Original abstraction not found: {original_json}")
        return None
    if edited_json is None:
        print(f"Error: No edited abstraction found in {result_folder}")
        return None

    print("Loading abstractions...")
    original = load_abstraction(str(original_json))
    edited = load_abstraction(str(edited_json))
    print(f"  Original: {len(original)} superquadrics")
    print(f"  Edited: {len(edited)} superquadrics")

    # Compare abstractions to get categories
    print("Comparing abstractions...")
    categories = compare_abstractions(original, edited)

    unchanged_indices = {sq["index"] for sq in categories.get("unchanged", [])}
    changed_indices = {sq["index"] for sq in categories.get("changed_edited", [])}
    added_deleted_indices = {sq["index"] for sq in categories.get("added", [])}
    added_deleted_indices.update({sq["index"] for sq in categories.get("deleted", [])})

    print(f"  Unchanged: {len(unchanged_indices)} superquadrics")
    print(f"  Changed: {len(changed_indices)} superquadrics")
    print(f"  Added/Deleted: {len(added_deleted_indices)} superquadrics")

    # Define colors (RGB 0-255)
    COLOR_UNCHANGED = (
        color_unchanged
        if color_unchanged is not None
        else DEFAULT_CATEGORIZED_UNCHANGED_RGB
    )
    COLOR_BLUE = color_changed if color_changed is not None else [40, 90, 180]
    COLOR_PURPLE = (
        color_added_deleted if color_added_deleted is not None else [150, 50, 170]
    )
    COLOR_GRAY = [
        180,
        180,
        180,
    ]  # original abstraction mesh + original_slat_gray (light gray)

    print(
        f"  Using colors: unchanged={COLOR_UNCHANGED}, changed={COLOR_BLUE}, added/deleted={COLOR_PURPLE}"
    )

    # Recolor edited abstraction based on categories
    recolored_edited = []
    for sq in edited:
        sq_copy = sq.copy()
        idx = sq.get("index", -1)

        if idx in unchanged_indices:
            sq_copy["color"] = COLOR_UNCHANGED
        elif idx in changed_indices:
            sq_copy["color"] = COLOR_BLUE
        elif idx in added_deleted_indices:
            sq_copy["color"] = COLOR_PURPLE
        else:
            # Fallback - check if this SQ exists in original
            sq_copy["color"] = COLOR_PURPLE  # New SQs are added

        recolored_edited.append(sq_copy)

    # Load normalization
    normalization_path = result_folder / "inversion" / "normalization.pt"
    if normalization_path.exists():
        normalization = torch.load(normalization_path, weights_only=False)
        center = normalization["center"]
        scale = normalization["scale"]
        print(f"  Loaded normalization: center={center}, scale={scale}")
    else:
        print("  Warning: normalization.pt not found, computing from mesh")
        center = None
        scale = None

    output_paths = {}

    # Rotation to align with Trellis coordinate system (90 instead of -90 to flip z)
    trellis_rotation = (90, 0, 0)

    # === 1. Render categorized edited abstraction mesh ===
    mesh_ply_path = output_dir / "categorized_abstraction.ply"
    mesh_png_path = output_dir / "categorized_abstraction.png"

    if not (skip_existing and mesh_png_path.exists()):
        print("Creating recolored mesh...")
        print(
            f"  Render resolution (categorized_abstraction.png): {int(categorized_mesh_render_resolution)}x{int(categorized_mesh_render_resolution)}"
        )
        recolored_mesh = mesh_from_abstraction(recolored_edited, resolution=30)

        if recolored_mesh is not None:
            # Save as PLY to preserve vertex colors
            recolored_mesh.export(str(mesh_ply_path), file_type="ply")
            print(f"  Saved PLY: {mesh_ply_path}")

            # Render with trellis rotation and invisible ground to match other visualizations
            # Much lower light energy for richer, less washed out colors (gets multiplied by 2.5 internally)
            render_obj_with_blender(
                str(mesh_ply_path),
                str(mesh_png_path),
                rotation=trellis_rotation,
                invisible_ground=True,
                dist=2.0,
                light_energy=2.5,
                shade_smooth=False,
                res_x=int(categorized_mesh_render_resolution),
                res_y=int(categorized_mesh_render_resolution),
            )
            print(f"  Saved render: {mesh_png_path}")
            output_paths["mesh"] = str(mesh_png_path)
    else:
        print(f"Skipping existing: {mesh_png_path}")
        output_paths["mesh"] = str(mesh_png_path)

    # === 1b. Render original abstraction in gray ===
    original_mesh_ply_path = output_dir / "original_abstraction_gray.ply"
    original_mesh_png_path = output_dir / "original_abstraction_gray.png"

    if not (skip_existing and original_mesh_png_path.exists()):
        print("Creating gray original abstraction mesh...")

        # Recolor original abstraction to gray
        gray_original = []
        for sq in original:
            sq_copy = sq.copy()
            sq_copy["color"] = COLOR_GRAY
            gray_original.append(sq_copy)

        gray_mesh = mesh_from_abstraction(gray_original, resolution=30)

        if gray_mesh is not None:
            # Save as PLY to preserve vertex colors
            gray_mesh.export(str(original_mesh_ply_path), file_type="ply")
            print(f"  Saved PLY: {original_mesh_ply_path}")

            # Render with same settings as categorized abstraction
            render_obj_with_blender(
                str(original_mesh_ply_path),
                str(original_mesh_png_path),
                rotation=trellis_rotation,
                invisible_ground=True,
                dist=2.0,
                light_energy=2.5,
                shade_smooth=False,
            )
            print(f"  Saved render: {original_mesh_png_path}")
            output_paths["original_abstraction_gray"] = str(original_mesh_png_path)
    else:
        print(f"Skipping existing: {original_mesh_png_path}")
        output_paths["original_abstraction_gray"] = str(original_mesh_png_path)

    # === 1c. Unit superquadric (mesh; same view / Blender settings as step 1) ===
    if include_unit_sq:
        unit_sq_ply_path = output_dir / "unit_sq.ply"
        unit_sq_png_path = output_dir / "unit_sq.png"
        if not (skip_existing and unit_sq_png_path.exists()):
            print(
                "Creating unit superquadric mesh (isotropic scale 0.1, I, no translation; changed blue)..."
            )
            u_list = unit_sq_abstraction_list(color_rgb_255=tuple(COLOR_BLUE))
            unit_mesh = mesh_from_abstraction(u_list, resolution=30)
            if unit_mesh is not None and not unit_mesh.is_empty:
                unit_mesh.export(str(unit_sq_ply_path), file_type="ply")
                print(f"  Saved PLY: {unit_sq_ply_path}")
                render_obj_with_blender(
                    str(unit_sq_ply_path),
                    str(unit_sq_png_path),
                    rotation=trellis_rotation,
                    invisible_ground=True,
                    dist=2.0,
                    light_energy=2.5,
                    shade_smooth=False,
                )
                print(f"  Saved render: {unit_sq_png_path}")
                output_paths["unit_sq"] = str(unit_sq_png_path)
            else:
                print("  Warning: unit superquadric mesh is empty, skipping")
        else:
            print(f"Skipping existing: {unit_sq_png_path}")
            output_paths["unit_sq"] = str(unit_sq_png_path)

    # === 2. Render voxelized version ===
    voxel_ply_path = output_dir / "categorized_abstraction_voxels.ply"
    voxel_png_path = output_dir / "categorized_abstraction_voxels.png"

    if not (skip_existing and voxel_png_path.exists()):
        print("Creating voxelized recolored abstraction...")

        # Create mesh and voxelize
        recolored_mesh = mesh_from_abstraction(recolored_edited, resolution=30)

        if recolored_mesh is not None:
            # Save mesh temporarily for voxelization
            temp_mesh_path = output_dir / "temp_categorized.ply"
            recolored_mesh.export(str(temp_mesh_path), file_type="ply")

            # Voxelize with colors
            voxel_grid, color_grid = voxelize_mesh_with_colors(
                str(temp_mesh_path),
                resolution=64,
                center=center,
                scale=scale,
            )

            num_voxels = (voxel_grid > 0.5).sum()
            print(f"  Voxelized to {num_voxels} voxels")

            # Transform to Trellis orientation
            voxel_grid_trellis = transform_voxels_to_trellis(voxel_grid)
            color_grid_trellis = np.swapaxes(color_grid, 1, 2)
            color_grid_trellis = np.flip(color_grid_trellis, axis=1)

            # Create colored cube mesh
            cube_mesh = voxels_to_colored_cube_mesh(
                voxel_grid_trellis, color_grid_trellis
            )

            # Save PLY
            o3d.io.write_triangle_mesh(
                str(voxel_ply_path), cube_mesh, write_vertex_colors=True
            )
            print(f"  Saved PLY: {voxel_ply_path}")

            # Render with invisible ground - increase distance and much lower light for richer colors
            render_obj_with_blender(
                str(voxel_ply_path),
                str(voxel_png_path),
                invisible_ground=True,
                shade_smooth=False,
            )
            print(f"  Saved render: {voxel_png_path}")
            output_paths["voxels"] = str(voxel_png_path)

            # Clean up temp file
            if temp_mesh_path.exists():
                temp_mesh_path.unlink()
    else:
        print(f"Skipping existing: {voxel_png_path}")
        output_paths["voxels"] = str(voxel_png_path)

    # === 3. Render transformed meshes ===
    # Render with same settings as categorized_abstraction, colored blue (edited color)
    transformed_meshes_folder = result_folder / "transformed_meshes"
    combined_mesh_path = result_folder / "transformed_original_mesh.ply"

    # Render individual transformed meshes
    if transformed_meshes_folder.exists():
        print("Rendering individual transformed meshes (colored blue)...")
        output_paths["transformed"] = []

        for mesh_file in sorted(transformed_meshes_folder.glob("*.obj")) + sorted(
            transformed_meshes_folder.glob("*.ply")
        ):
            png_path = output_dir / f"{mesh_file.stem}.png"

            if skip_existing and png_path.exists():
                print(f"  Skipping existing: {png_path.name}")
                output_paths["transformed"].append(str(png_path))
                continue

            print(f"  Rendering: {mesh_file.name}")

            # Load mesh and recolor it blue (edited/changed color)
            mesh = trimesh.load(str(mesh_file), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(
                    [
                        g
                        for g in mesh.geometry.values()
                        if isinstance(g, trimesh.Trimesh)
                    ]
                )

            # Apply blue color (same as COLOR_BLUE for changed/edited)
            blue_color = np.array([70, 130, 220, 255], dtype=np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh, vertex_colors=np.tile(blue_color, (len(mesh.vertices), 1))
            )

            # Save colored mesh temporarily
            colored_ply_path = output_dir / f"{mesh_file.stem}_colored.ply"
            mesh.export(str(colored_ply_path), file_type="ply")

            # Render with same settings as categorized_abstraction
            render_obj_with_blender(
                str(colored_ply_path),
                str(png_path),
                rotation=trellis_rotation,
                invisible_ground=True,
                dist=2.0,
                light_energy=2.5,
            )
            output_paths["transformed"].append(str(png_path))

            # Clean up temp colored mesh
            if colored_ply_path.exists():
                colored_ply_path.unlink()

    # Render combined transformed mesh
    if combined_mesh_path.exists():
        combined_png_path = output_dir / "transformed_original_mesh.png"

        if not (skip_existing and combined_png_path.exists()):
            print("Rendering combined transformed mesh (colored blue)...")

            # Load and recolor blue
            mesh = trimesh.load(str(combined_mesh_path), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(
                    [
                        g
                        for g in mesh.geometry.values()
                        if isinstance(g, trimesh.Trimesh)
                    ]
                )

            blue_color = np.array([70, 130, 220, 255], dtype=np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh, vertex_colors=np.tile(blue_color, (len(mesh.vertices), 1))
            )

            colored_ply_path = output_dir / "transformed_original_mesh_colored.ply"
            mesh.export(str(colored_ply_path), file_type="ply")

            # Render with same settings as categorized_abstraction
            render_obj_with_blender(
                str(colored_ply_path),
                str(combined_png_path),
                rotation=trellis_rotation,
                invisible_ground=True,
                dist=2.0,
                light_energy=2.5,
            )
            print(f"  Saved render: {combined_png_path}")
            output_paths["combined_transformed"] = str(combined_png_path)

            # Clean up temp colored mesh
            if colored_ply_path.exists():
                colored_ply_path.unlink()
        else:
            print(f"Skipping existing: {combined_png_path}")
            output_paths["combined_transformed"] = str(combined_png_path)

    # === 4. Render original_slat.glb and appearance_edited.glb (with original colors) ===
    # GLB files from Trellis need -90 X rotation for correct orientation
    glb_rotation = (-90, 0, 0)
    # PLY exported from trimesh-loaded GLB: camera rotation identity (see original_slat_colored)
    colored_ply_rotation = (0, 0, 0)

    original_slat_glb = result_folder / "original_slat.glb"
    if original_slat_glb.exists():
        original_slat_png = output_dir / "original_slat.png"

        if not (skip_existing and original_slat_png.exists()):
            print("Rendering original_slat.glb (original colors)...")
            render_obj_with_blender(
                str(original_slat_glb),
                str(original_slat_png),
                rotation=glb_rotation,
                invisible_ground=True,
                dist=2.0,
                light_energy=2.5,
            )
            print(f"  Saved render: {original_slat_png}")
            output_paths["original_slat"] = str(original_slat_png)
        else:
            print(f"Skipping existing: {original_slat_png}")
            output_paths["original_slat"] = str(original_slat_png)

    appearance_edited_glb = result_folder / "appearance_edited.glb"
    if appearance_edited_glb.exists():
        appearance_edited_png = output_dir / "appearance_edited.png"

        if not (skip_existing and appearance_edited_png.exists()):
            print("Rendering appearance_edited.glb (original colors)...")
            render_obj_with_blender(
                str(appearance_edited_glb),
                str(appearance_edited_png),
                rotation=glb_rotation,
                invisible_ground=True,
                dist=2.0,
                light_energy=2.5,
            )
            print(f"  Saved render: {appearance_edited_png}")
            output_paths["appearance_edited"] = str(appearance_edited_png)
        else:
            print(f"Skipping existing: {appearance_edited_png}")
            output_paths["appearance_edited"] = str(appearance_edited_png)

    # === 4b. Render original_slat mesh in the same gray as original_abstraction_gray ===
    if original_slat_glb.exists():
        original_slat_gray_png = output_dir / "original_slat_gray.png"
        if not (skip_existing and original_slat_gray_png.exists()):
            print(
                f"Rendering original_slat.glb (gray {COLOR_GRAY}, same RGB as original_abstraction_gray)..."
            )
            mesh = trimesh.load(str(original_slat_glb), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(
                    [
                        g
                        for g in mesh.geometry.values()
                        if isinstance(g, trimesh.Trimesh)
                    ]
                )
            gray_slat_rgba = np.array(list(COLOR_GRAY) + [255], dtype=np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.tile(gray_slat_rgba, (len(mesh.vertices), 1)),
            )
            gray_slat_ply = output_dir / "original_slat_gray.ply"
            mesh.export(str(gray_slat_ply), file_type="ply")
            render_obj_with_blender(
                str(gray_slat_ply),
                str(original_slat_gray_png),
                rotation=colored_ply_rotation,
                invisible_ground=True,
                dist=2.0,
                light_energy=2.5,
            )
            print(f"  Saved render: {original_slat_gray_png}")
            output_paths["original_slat_gray"] = str(original_slat_gray_png)
            if gray_slat_ply.exists():
                gray_slat_ply.unlink()
        else:
            print(f"Skipping existing: {original_slat_gray_png}")
            output_paths["original_slat_gray"] = str(original_slat_gray_png)

    # === 5. Render original_slat.glb in unchanged color (orange by default) ===
    if original_slat_glb.exists():
        original_slat_colored_png = output_dir / "original_slat_colored.png"

        if not (skip_existing and original_slat_colored_png.exists()):
            print(
                f"Rendering original_slat.glb (colored unchanged {COLOR_UNCHANGED})..."
            )

            # Load mesh and recolor with unchanged color
            mesh = trimesh.load(str(original_slat_glb), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(
                    [
                        g
                        for g in mesh.geometry.values()
                        if isinstance(g, trimesh.Trimesh)
                    ]
                )

            unchanged_slat_rgba = np.array(
                list(COLOR_UNCHANGED) + [255], dtype=np.uint8
            )
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh,
                vertex_colors=np.tile(unchanged_slat_rgba, (len(mesh.vertices), 1)),
            )

            colored_ply_path = output_dir / "original_slat_colored.ply"
            mesh.export(str(colored_ply_path), file_type="ply")

            render_obj_with_blender(
                str(colored_ply_path),
                str(original_slat_colored_png),
                rotation=colored_ply_rotation,
                invisible_ground=True,
                dist=2.0,
                light_energy=2.5,
            )
            print(f"  Saved render: {original_slat_colored_png}")
            output_paths["original_slat_colored"] = str(original_slat_colored_png)

            if colored_ply_path.exists():
                colored_ply_path.unlink()
        else:
            print(f"Skipping existing: {original_slat_colored_png}")
            output_paths["original_slat_colored"] = str(original_slat_colored_png)

    # === 6. Render edited_final.obj in added/deleted color (purple) ===
    edited_final_obj = result_folder / "edited_final.obj"
    if edited_final_obj.exists():
        edited_abstraction_colored_png = output_dir / "edited_abstraction_colored.png"

        if not (skip_existing and edited_abstraction_colored_png.exists()):
            print(f"Rendering edited_final.obj (colored {COLOR_PURPLE})...")

            # Load mesh and recolor with added/deleted color (purple)
            mesh = trimesh.load(str(edited_final_obj), force="mesh")
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(
                    [
                        g
                        for g in mesh.geometry.values()
                        if isinstance(g, trimesh.Trimesh)
                    ]
                )

            purple_color = np.array(list(COLOR_PURPLE) + [255], dtype=np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh=mesh, vertex_colors=np.tile(purple_color, (len(mesh.vertices), 1))
            )

            colored_ply_path = output_dir / "edited_abstraction_colored.ply"
            mesh.export(str(colored_ply_path), file_type="ply")

            render_obj_with_blender(
                str(colored_ply_path),
                str(edited_abstraction_colored_png),
                rotation=trellis_rotation,
                invisible_ground=True,
                dist=2.0,
                light_energy=2.5,
            )
            print(f"  Saved render: {edited_abstraction_colored_png}")
            output_paths["edited_abstraction_colored"] = str(
                edited_abstraction_colored_png
            )

            if colored_ply_path.exists():
                colored_ply_path.unlink()
        else:
            print(f"Skipping existing: {edited_abstraction_colored_png}")
            output_paths["edited_abstraction_colored"] = str(
                edited_abstraction_colored_png
            )

    return output_paths


def render_for_teaser(
    teaser_folder: str,
    output_dir: str = None,
    skip_existing: bool = True,
    color_unchanged: list = None,
    color_changed: list = None,
    color_added_deleted: list = None,
    teaser_only: str = None,
):
    """
    Render visualizations for teaser figure from a folder containing multiple result subfolders.

    For each result folder found, renders PNGs and writes a matching GLB per asset under
    teaser_renders/ (including abstraction meshes built from JSON).

    1. Original input shape (from input_mesh/ or from_shapenet) — original_shape.png + original_shape.glb
    2. original_slat.glb from the run — original_slat.png + copy as teaser_renders/original_slat.glb
    3. appearance_edited.glb — appearance_edited_mesh.png + appearance_edited_mesh.glb
    3b. transformed_meshes (structural / warped mesh) — warped_mesh.png + warped_mesh.glb, vertex color = changed blue
    4. Original abstraction (gray) — original_abstraction.png + original_abstraction.glb
    5. Edited abstraction (categorized colors) — edited_abstraction_categorized.png + .glb

    Args:
        teaser_folder: Path to folder containing result subfolders
        output_dir: Directory to save output images (default: each subfolder's own directory)
        skip_existing: Skip rendering if output already exists
        color_unchanged: RGB color [0-255] for unchanged parts in edited abstraction (default: 190,120,40 orange)
        color_changed: RGB color [0-255] for changed/edited parts (default: [40, 90, 180])
        color_added_deleted: RGB color [0-255] for added/deleted parts (default: [150, 50, 170])
        teaser_only: If ``"abstractions"``, only steps 4–5 (abstraction GLBs/PNGs); skips original
            shape and pipeline GLB steps.

    Returns:
        Dict mapping subfolder paths to their output paths (PNG and *_glb keys)
    """
    from abstraction import mesh_from_abstraction, load_abstraction
    from inpaint_data_preparation import compare_abstractions

    teaser_folder = Path(teaser_folder)

    if not teaser_folder.exists():
        print(f"Error: Teaser folder not found: {teaser_folder}")
        return {}

    # Define colors (RGB 0-255)
    COLOR_UNCHANGED = (
        color_unchanged
        if color_unchanged is not None
        else DEFAULT_CATEGORIZED_UNCHANGED_RGB
    )
    COLOR_BLUE = color_changed if color_changed is not None else [40, 90, 180]
    COLOR_PURPLE = (
        color_added_deleted if color_added_deleted is not None else [150, 50, 170]
    )
    COLOR_GRAY = [180, 180, 180]  # original (pre-edit) abstraction mesh only

    # Rotations for rendering (pipeline GLBs; abstractions use TEASER_ABSTRACTION_VIEW_ORIENT_IDX)
    glb_rotation = (-90, 0, 0)  # For GLB files from Trellis

    all_outputs = {}

    # Find all result folders (those containing both abstraction.json and edited_abstraction_final.json)
    def find_result_folders(folder: Path) -> list:
        """Recursively find all result folders."""
        result_folders = []

        # Check if this folder is a result folder
        has_original_abstraction = (folder / "abstraction.json").exists()
        has_edited_abstraction = any(
            (folder / name).exists()
            for name in [
                "edited_abstraction_final.json",
                "edited_abstraction.json",
                "edited_abstraction_iter1.json",
            ]
        )
        # has_original_mesh = (folder / "original_slat.glb").exists()
        # has_edited_mesh = (folder / "appearance_edited.glb").exists()

        if has_original_abstraction and has_edited_abstraction:
            result_folders.append(folder)

        # Recurse into subdirectories
        for subdir in folder.iterdir():
            if subdir.is_dir():
                result_folders.extend(find_result_folders(subdir))

        return result_folders

    result_folders = find_result_folders(teaser_folder)
    print(f"Found {len(result_folders)} result folders in {teaser_folder}")

    for result_folder in result_folders:
        print(f"\n{'=' * 60}")
        print(f"Processing: {result_folder}")
        print(f"{'=' * 60}")

        # Determine output directory
        if output_dir:
            # Create subfolder structure in output_dir
            rel_path = result_folder.relative_to(teaser_folder)
            folder_output_dir = Path(output_dir) / rel_path / "teaser_renders"
        else:
            folder_output_dir = result_folder / "teaser_renders"

        folder_output_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {}

        if teaser_only != "abstractions":
            # === 1. Original input shape (OBJ) + GLB ==========================================
            input_mesh_folder = result_folder / "input_mesh"
            original_input_mesh = None
            obj_rotation = (0, 0, 0)  # Correct rotation for normalized.obj

            if input_mesh_folder.exists():
                if (input_mesh_folder / "normalized.obj").exists():
                    original_input_mesh = input_mesh_folder / "normalized.obj"

            if original_input_mesh is None:
                from_shapenet_model = (
                    result_folder / "from_shapenet" / "models" / "model_normalized.obj"
                )
                if from_shapenet_model.exists():
                    original_input_mesh = from_shapenet_model
                    print("  Using fallback: from_shapenet/models/model_normalized.obj")

            if original_input_mesh is not None:
                original_shape_png = folder_output_dir / "original_shape.png"
                original_shape_glb = folder_output_dir / "original_shape.glb"
                skip_both = (
                    skip_existing
                    and original_shape_png.exists()
                    and original_shape_glb.exists()
                )
                if not skip_both:
                    if not (skip_existing and original_shape_glb.exists()):
                        print(
                            f"  Writing original input GLB ({original_shape_glb.name})..."
                        )
                        _teaser_export_glb_from_source(
                            original_input_mesh, original_shape_glb
                        )
                        print(f"    Saved: {original_shape_glb}")
                    if not (skip_existing and original_shape_png.exists()):
                        print(
                            f"  Rendering original input shape: {original_input_mesh.name}..."
                        )
                        render_obj_with_blender(
                            str(original_input_mesh),
                            str(original_shape_png),
                            rotation=obj_rotation,
                            invisible_ground=True,
                            dist=2.0,
                            light_energy=2.5,
                        )
                        print(f"    Saved: {original_shape_png}")
                    else:
                        print(f"  Skipping existing: {original_shape_png.name}")
                else:
                    print(f"  Skipping existing: {original_shape_png.name} (+ GLB)")
                output_paths["original_shape"] = str(original_shape_png)
                output_paths["original_shape_glb"] = str(original_shape_glb)
            else:
                print(
                    "  Warning: No original input mesh found in input_mesh/ or from_shapenet/models/"
                )

            # === 2. original_slat.glb (Trellis reconstruction) + GLB ==========================
            original_slat_glb = result_folder / "original_slat.glb"
            if original_slat_glb.exists():
                original_slat_png = folder_output_dir / "original_slat.png"
                teaser_original_slat_glb = folder_output_dir / "original_slat.glb"
                orig_slat_orient = teaser_mesh_orientation("original_slat")
                skip_both = (
                    skip_existing
                    and original_slat_png.exists()
                    and teaser_original_slat_glb.exists()
                )
                if not skip_both:
                    if not (skip_existing and teaser_original_slat_glb.exists()):
                        msg = "  Writing pipeline original_slat.glb into teaser_renders"
                        if orig_slat_orient is not None:
                            _oname = ORIENTATION_TRANSFORMS[orig_slat_orient][0]
                            msg += f" (orientation {orig_slat_orient}: {_oname})..."
                        else:
                            msg += "..."
                        print(msg)
                        _teaser_write_pipeline_glb(
                            original_slat_glb,
                            teaser_original_slat_glb,
                            orig_slat_orient,
                        )
                        print(f"    Saved: {teaser_original_slat_glb}")
                    if not (skip_existing and original_slat_png.exists()):
                        print("  Rendering original_slat (teaser GLB)...")
                        render_obj_with_blender(
                            str(teaser_original_slat_glb),
                            str(original_slat_png),
                            rotation_matrix=_teaser_vertex_view_rotation_matrix(
                                glb_rotation, orig_slat_orient, compose="invM_Rb"
                            ),
                            invisible_ground=True,
                            dist=2.0,
                            light_energy=2.5,
                        )
                        print(f"    Saved: {original_slat_png}")
                    else:
                        print(f"  Skipping existing: {original_slat_png.name}")
                else:
                    print(f"  Skipping existing: {original_slat_png.name} (+ GLB)")
                output_paths["original_slat"] = str(original_slat_png)
                output_paths["original_slat_glb"] = str(teaser_original_slat_glb)
            else:
                print("  Warning: original_slat.glb not found")

            # === 3. appearance_edited.glb + GLB ===============================================
            appearance_edited_glb = result_folder / "appearance_edited.glb"
            if appearance_edited_glb.exists():
                appearance_edited_png = folder_output_dir / "appearance_edited_mesh.png"
                appearance_edited_out_glb = (
                    folder_output_dir / "appearance_edited_mesh.glb"
                )
                appearance_edited_orient = teaser_mesh_orientation("appearance_edited")
                skip_both = (
                    skip_existing
                    and appearance_edited_png.exists()
                    and appearance_edited_out_glb.exists()
                )
                if not skip_both:
                    if not (skip_existing and appearance_edited_out_glb.exists()):
                        msg = "  Writing pipeline appearance_edited.glb into teaser_renders"
                        if appearance_edited_orient is not None:
                            msg += f" (orientation {appearance_edited_orient}: {ORIENTATION_TRANSFORMS[appearance_edited_orient][0]})"
                        msg += "..."
                        print(msg)
                        _teaser_write_pipeline_glb(
                            appearance_edited_glb,
                            appearance_edited_out_glb,
                            appearance_edited_orient,
                        )
                        print(f"    Saved: {appearance_edited_out_glb}")
                    if not (skip_existing and appearance_edited_png.exists()):
                        print("  Rendering appearance_edited (teaser GLB)...")
                        render_obj_with_blender(
                            str(appearance_edited_out_glb),
                            str(appearance_edited_png),
                            rotation_matrix=_teaser_vertex_view_rotation_matrix(
                                glb_rotation,
                                appearance_edited_orient,
                                compose="invM_Rb",
                            ),
                            invisible_ground=True,
                            dist=2.0,
                            light_energy=2.5,
                        )
                        print(f"    Saved: {appearance_edited_png}")
                    else:
                        print(f"  Skipping existing: {appearance_edited_png.name}")
                else:
                    print(f"  Skipping existing: {appearance_edited_png.name} (+ GLB)")
                output_paths["appearance_edited_mesh"] = str(appearance_edited_png)
                output_paths["appearance_edited_mesh_glb"] = str(
                    appearance_edited_out_glb
                )
            else:
                print("  Warning: appearance_edited.glb not found")

            # === 3b. Warped / transformed mesh (structural edit) in changed blue =================
            transformed_dir = result_folder / "transformed_meshes"
            warped_src = None
            if transformed_dir.is_dir():
                preferred = transformed_dir / "transformed_mesh_sq2.obj"
                if preferred.is_file():
                    warped_src = preferred
                else:
                    for p in sorted(transformed_dir.glob("*.obj")) + sorted(
                        transformed_dir.glob("*.ply")
                    ):
                        warped_src = p
                        break
            if warped_src is not None:
                warped_png = folder_output_dir / "warped_mesh.png"
                warped_glb = folder_output_dir / "warped_mesh.glb"
                warped_mesh_trellis_rotation = (
                    90,
                    0,
                    0,
                )  # same as categorized ``transformed_meshes`` renders
                blue_rgba = np.array(COLOR_BLUE + [255], dtype=np.uint8)
                skip_both = (
                    skip_existing and warped_png.exists() and warped_glb.exists()
                )
                if not skip_both:
                    if not (skip_existing and warped_glb.exists()):
                        print(
                            f"  Writing warped mesh GLB (blue) from {warped_src.name}..."
                        )
                        mesh = trimesh.load(str(warped_src), force="mesh")
                        if isinstance(mesh, trimesh.Scene):
                            mesh = trimesh.util.concatenate(
                                [
                                    g
                                    for g in mesh.geometry.values()
                                    if isinstance(g, trimesh.Trimesh)
                                ]
                            )
                        mesh.visual = trimesh.visual.ColorVisuals(
                            mesh=mesh,
                            vertex_colors=np.tile(blue_rgba, (len(mesh.vertices), 1)),
                        )
                        mesh.export(str(warped_glb), file_type="glb")
                        print(f"    Saved: {warped_glb}")
                    if not (skip_existing and warped_png.exists()):
                        print("  Rendering warped_mesh (teaser, changed blue)...")
                        render_obj_with_blender(
                            str(warped_glb),
                            str(warped_png),
                            rotation=warped_mesh_trellis_rotation,
                            invisible_ground=True,
                            dist=2.0,
                            light_energy=2.5,
                        )
                        print(f"    Saved: {warped_png}")
                    else:
                        print(f"  Skipping existing: {warped_png.name}")
                else:
                    print(f"  Skipping existing: {warped_png.name} (+ GLB)")
                output_paths["warped_mesh"] = str(warped_png)
                output_paths["warped_mesh_glb"] = str(warped_glb)

        # === 4 & 5. Render abstractions ===
        original_json = result_folder / "abstraction.json"
        edited_json = None
        for name in [
            "edited_abstraction_final.json",
            "edited_abstraction.json",
            "edited_abstraction_iter1.json",
        ]:
            candidate = result_folder / name
            if candidate.exists():
                edited_json = candidate
                break

        if original_json.exists() and edited_json is not None:
            print("  Loading abstractions...")
            original = load_abstraction(str(original_json))
            edited = load_abstraction(str(edited_json))
            print(f"    Original: {len(original)} superquadrics")
            print(f"    Edited: {len(edited)} superquadrics")

            # Compare abstractions to get categories
            categories = compare_abstractions(original, edited)

            unchanged_indices = {sq["index"] for sq in categories.get("unchanged", [])}
            changed_indices = {
                sq["index"] for sq in categories.get("changed_edited", [])
            }
            added_deleted_indices = {sq["index"] for sq in categories.get("added", [])}
            added_deleted_indices.update(
                {sq["index"] for sq in categories.get("deleted", [])}
            )

            print(
                f"    Categories: unchanged={len(unchanged_indices)}, changed={len(changed_indices)}, added/deleted={len(added_deleted_indices)}"
            )

            # === 4. Original abstraction (gray) + GLB =======================================
            original_abstraction_png = folder_output_dir / "original_abstraction.png"
            original_abstraction_glb = folder_output_dir / "original_abstraction.glb"
            orig_abs_orient = teaser_mesh_orientation("original_abstraction")
            need_orig_abs = not (
                skip_existing
                and original_abstraction_png.exists()
                and original_abstraction_glb.exists()
            )
            if need_orig_abs:
                gray_original = []
                for sq in original:
                    sq_copy = sq.copy()
                    sq_copy["color"] = COLOR_GRAY
                    gray_original.append(sq_copy)

                gray_mesh = mesh_from_abstraction(gray_original, resolution=30)
                gray_mesh = _apply_teaser_orientation_mesh(gray_mesh, orig_abs_orient)

                if gray_mesh is not None:
                    if not (skip_existing and original_abstraction_glb.exists()):
                        omsg = "  Writing original abstraction GLB"
                        if orig_abs_orient is not None:
                            omsg += f" (orientation {orig_abs_orient}: {ORIENTATION_TRANSFORMS[orig_abs_orient][0]})"
                        omsg += "..."
                        print(omsg)
                        gray_mesh.export(str(original_abstraction_glb), file_type="glb")
                        print(f"    Saved: {original_abstraction_glb}")
                    if not (skip_existing and original_abstraction_png.exists()):
                        _on, _ = ORIENTATION_TRANSFORMS[
                            TEASER_ABSTRACTION_VIEW_ORIENT_IDX
                        ]
                        print(
                            f"  Rendering original abstraction (gray); view orient "
                            f"{TEASER_ABSTRACTION_VIEW_ORIENT_IDX} ({_on})..."
                        )
                        render_obj_with_blender(
                            str(original_abstraction_glb),
                            str(original_abstraction_png),
                            rotation_matrix=_teaser_abstraction_blender_view_matrix(),
                            force_glb_vertex_color=True,
                            invisible_ground=True,
                            dist=2.0,
                            light_energy=2.5,
                            shade_smooth=False,
                        )
                        print(f"    Saved: {original_abstraction_png}")
                    else:
                        print(f"  Skipping existing: {original_abstraction_png.name}")
                else:
                    print("  Warning: original abstraction mesh is empty")
            else:
                print(f"  Skipping existing: {original_abstraction_png.name} (+ GLB)")
            output_paths["original_abstraction"] = str(original_abstraction_png)
            output_paths["original_abstraction_glb"] = str(original_abstraction_glb)

            # === 5. Edited abstraction (categorized) + GLB =================================
            edited_abstraction_png = (
                folder_output_dir / "edited_abstraction_categorized.png"
            )
            edited_abstraction_glb = (
                folder_output_dir / "edited_abstraction_categorized.glb"
            )
            edited_abs_orient = teaser_mesh_orientation(
                "edited_abstraction_categorized"
            )
            need_edited_abs = not (
                skip_existing
                and edited_abstraction_png.exists()
                and edited_abstraction_glb.exists()
            )
            if need_edited_abs:
                recolored_edited = []
                for sq in edited:
                    sq_copy = sq.copy()
                    idx = sq.get("index", -1)

                    if idx in unchanged_indices:
                        sq_copy["color"] = COLOR_UNCHANGED
                    elif idx in changed_indices:
                        sq_copy["color"] = COLOR_BLUE
                    elif idx in added_deleted_indices:
                        sq_copy["color"] = COLOR_PURPLE
                    else:
                        sq_copy["color"] = COLOR_PURPLE

                    recolored_edited.append(sq_copy)

                recolored_mesh = mesh_from_abstraction(recolored_edited, resolution=30)
                recolored_mesh = _apply_teaser_orientation_mesh(
                    recolored_mesh, edited_abs_orient
                )

                if recolored_mesh is not None:
                    if not (skip_existing and edited_abstraction_glb.exists()):
                        emsg = "  Writing edited abstraction (categorized) GLB"
                        if edited_abs_orient is not None:
                            emsg += f" (orientation {edited_abs_orient}: {ORIENTATION_TRANSFORMS[edited_abs_orient][0]})"
                        emsg += "..."
                        print(emsg)
                        recolored_mesh.export(
                            str(edited_abstraction_glb), file_type="glb"
                        )
                        print(f"    Saved: {edited_abstraction_glb}")
                    if not (skip_existing and edited_abstraction_png.exists()):
                        _on, _ = ORIENTATION_TRANSFORMS[
                            TEASER_ABSTRACTION_VIEW_ORIENT_IDX
                        ]
                        print(
                            f"  Rendering edited abstraction (categorized); view orient "
                            f"{TEASER_ABSTRACTION_VIEW_ORIENT_IDX} ({_on})..."
                        )
                        render_obj_with_blender(
                            str(edited_abstraction_glb),
                            str(edited_abstraction_png),
                            rotation_matrix=_teaser_abstraction_blender_view_matrix(),
                            force_glb_vertex_color=True,
                            invisible_ground=True,
                            dist=2.0,
                            light_energy=2.5,
                            shade_smooth=False,
                        )
                        print(f"    Saved: {edited_abstraction_png}")
                    else:
                        print(f"  Skipping existing: {edited_abstraction_png.name}")
                else:
                    print("  Warning: edited abstraction mesh is empty")
            else:
                print(f"  Skipping existing: {edited_abstraction_png.name} (+ GLB)")
            output_paths["edited_abstraction_categorized"] = str(edited_abstraction_png)
            output_paths["edited_abstraction_categorized_glb"] = str(
                edited_abstraction_glb
            )
        else:
            if not original_json.exists():
                print("  Warning: abstraction.json not found")
            if edited_json is None:
                print("  Warning: edited abstraction not found")

        all_outputs[str(result_folder)] = output_paths

    print(f"\n{'=' * 60}")
    print("Teaser rendering complete!")
    print(f"Processed {len(all_outputs)} result folders")
    print(f"{'=' * 60}")

    return all_outputs


def _teaser_rotation_candidate_matrices(r_base: np.ndarray, m3: np.ndarray) -> list:
    """Several compose orders of R_base and M to compare (column-vector convention)."""
    inv = np.linalg.inv(m3)
    return [
        ("01_Rb_invM", r_base @ inv),
        ("02_invM_Rb", inv @ r_base),
        ("03_Rb_only", r_base.copy()),
        ("04_identity", np.eye(3)),
        ("05_invM_only", inv),
        ("06_M_Rb", m3 @ r_base),
        ("07_Rb_M", r_base @ m3),
    ]


def _teaser_input_shape_view_candidates() -> list:
    """Original normalized.obj has no dataset M; try common view rotations."""
    rb_glb = _blender_euler_xyz_deg_to_matrix_3x3(-90, 0, 0)
    rb_abs = _blender_euler_xyz_deg_to_matrix_3x3(90, 0, 0)
    return [
        ("01_identity", np.eye(3)),
        ("02_R_glb_neg90_x", rb_glb),
        ("03_R_abs_pos90_x", rb_abs),
        ("04_R_0_neg90_0", _blender_euler_xyz_deg_to_matrix_3x3(0, -90, 0)),
        ("05_R_0_0_neg90", _blender_euler_xyz_deg_to_matrix_3x3(0, 0, -90)),
    ]


def render_teaser_view_sweep(
    result_folder,
    skip_existing: bool = False,
    output_subdir: str = "view_options",
) -> None:
    """
    Render multiple Blender view rotations for teaser assets so you can pick the right formula.

    Writes PNGs under ``<result_folder>/teaser_renders/<output_subdir>/`` with filenames like
    ``appearance_edited_mesh__01_Rb_invM.png``. See option labels in
    ``_teaser_rotation_candidate_matrices`` / ``_teaser_input_shape_view_candidates``.

    Expects a single tredit result directory (with ``teaser_renders/*.glb`` already produced).
    """
    result_folder = Path(result_folder)
    teaser_dir = result_folder / "teaser_renders"
    out_dir = teaser_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    glb_base = _blender_euler_xyz_deg_to_matrix_3x3(-90, 0, 0)
    abs_base = _blender_euler_xyz_deg_to_matrix_3x3(90, 0, 0)
    _, m_glb = ORIENTATION_TRANSFORMS[TEASER_MESH_ORIENTATIONS["original_slat"]]
    m_glb_3 = np.asarray(m_glb[:3, :3], dtype=np.float64)
    _, m_abs = ORIENTATION_TRANSFORMS[TEASER_MESH_ORIENTATIONS["original_abstraction"]]
    m_abs_3 = np.asarray(m_abs[:3, :3], dtype=np.float64)

    glb_candidates = _teaser_rotation_candidate_matrices(glb_base, m_glb_3)
    abs_candidates = _teaser_rotation_candidate_matrices(abs_base, m_abs_3)

    render_kw_base = dict(invisible_ground=True, dist=2.0, light_energy=2.5)

    jobs = []

    for stem, rel_name, cands in (
        ("original_slat", "original_slat.glb", glb_candidates),
        ("appearance_edited_mesh", "appearance_edited_mesh.glb", glb_candidates),
        ("warped_mesh", "warped_mesh.glb", glb_candidates),
        ("original_abstraction", "original_abstraction.glb", abs_candidates),
        (
            "edited_abstraction_categorized",
            "edited_abstraction_categorized.glb",
            abs_candidates,
        ),
    ):
        p = teaser_dir / rel_name
        if p.exists():
            jobs.append((stem, p, cands))

    # Original input OBJ (no dataset orientation baked in)
    input_obj = None
    if (result_folder / "input_mesh" / "normalized.obj").exists():
        input_obj = result_folder / "input_mesh" / "normalized.obj"
    elif (result_folder / "from_shapenet" / "models" / "model_normalized.obj").exists():
        input_obj = result_folder / "from_shapenet" / "models" / "model_normalized.obj"
    if input_obj is not None:
        jobs.append(
            ("original_shape", input_obj, _teaser_input_shape_view_candidates())
        )

    if not jobs:
        print(
            f"No teaser meshes found under {teaser_dir} (and no input OBJ). Run teaser export first."
        )
        return

    print(f"Teaser view sweep -> {out_dir}")
    for stem, mesh_path, candidates in jobs:
        print(f"  {stem} ({mesh_path.name}):")
        render_kw = dict(render_kw_base)
        if stem in ("original_abstraction", "edited_abstraction_categorized"):
            render_kw["force_glb_vertex_color"] = True
            render_kw["shade_smooth"] = False
        elif stem == "warped_mesh":
            render_kw["force_glb_vertex_color"] = True
        for label, rot_mat in candidates:
            png_name = f"{stem}__{label}.png"
            out_png = out_dir / png_name
            if skip_existing and out_png.exists():
                print(f"    skip existing {png_name}")
                continue
            render_obj_with_blender(
                str(mesh_path), str(out_png), rotation_matrix=rot_mat, **render_kw
            )
            print(f"    wrote {png_name}")

    print(
        "Done. Compare PNGs in the folder above and note the label suffix (e.g. 03_Rb_only) you prefer."
    )


def _teaser_abstraction_extended_view_candidates() -> list:
    """
    Extra view rotations for abstraction GLBs (R_base = +90° X, M = flip_z): standalone
    orientations from ORIENTATION_TRANSFORMS and products with inv(M) and R_base.
    """
    r_abs = _blender_euler_xyz_deg_to_matrix_3x3(90, 0, 0)
    _, m_full = ORIENTATION_TRANSFORMS[TEASER_MESH_ORIENTATIONS["original_abstraction"]]
    m3 = np.asarray(m_full[:3, :3], dtype=np.float64)
    inv = np.linalg.inv(m3)
    cands = []
    cands.extend(_teaser_rotation_candidate_matrices(r_abs, m3))
    cands.append(("ref08_invM_Rb", inv @ r_abs))
    for idx in sorted(ORIENTATION_TRANSFORMS.keys()):
        oname, mat = ORIENTATION_TRANSFORMS[idx]
        mk = np.asarray(mat[:3, :3], dtype=np.float64)
        safe = "".join(c if c.isalnum() or c == "_" else "_" for c in oname)
        cands.append((f"a{idx:02d}_{safe}", mk))
        cands.append((f"b{idx:02d}_invMxO", inv @ mk))
        cands.append((f"c{idx:02d}_OxinvM", mk @ inv))
        cands.append((f"d{idx:02d}_RbOx", r_abs @ mk))
        cands.append((f"e{idx:02d}_OxRb", mk @ r_abs))
    return cands


def render_teaser_abstraction_view_sweep(
    result_folder,
    skip_existing: bool = False,
    output_subdir: str = "view_options_abstraction",
) -> None:
    """
    Like ``render_teaser_view_sweep`` but only ``original_abstraction.glb`` and
    ``edited_abstraction_categorized.glb``, with an expanded candidate set (see
    ``_teaser_abstraction_extended_view_candidates``). Uses vertex-color shading so
    category colors are visible.
    """
    result_folder = Path(result_folder)
    teaser_dir = result_folder / "teaser_renders"
    out_dir = teaser_dir / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _teaser_abstraction_extended_view_candidates()
    render_kw = dict(
        invisible_ground=True,
        dist=2.0,
        light_energy=2.5,
        force_glb_vertex_color=True,
        shade_smooth=False,
    )

    jobs = [
        ("original_abstraction", teaser_dir / "original_abstraction.glb"),
        (
            "edited_abstraction_categorized",
            teaser_dir / "edited_abstraction_categorized.glb",
        ),
    ]

    print(f"Abstraction-only view sweep -> {out_dir}")
    for stem, glb_path in jobs:
        if not glb_path.exists():
            print(f"  skip (missing): {glb_path.name}")
            continue
        print(f"  {stem}:")
        for label, rot_mat in candidates:
            png_name = f"{stem}__{label}.png"
            out_png = out_dir / png_name
            if skip_existing and out_png.exists():
                print(f"    skip existing {png_name}")
                continue
            render_obj_with_blender(
                str(glb_path), str(out_png), rotation_matrix=rot_mat, **render_kw
            )
            print(f"    wrote {png_name}")

    print(
        "Done. Set TEASER_ABSTRACTION_VIEW_ORIENT_IDX to the sweep index (e.g. a02 -> 2), "
        "or extend _teaser_abstraction_blender_view_matrix for a custom matrix."
    )


def _sanitize_sweep_label(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(name))


def render_noisy_pointcloud_orientation_sweep(
    result_folder: Path,
    output_dir: Path,
    threshold: float = 0.5,
    skip_existing: bool = True,
    seed: int = 42,
    noise_std: float = 15.0,
) -> None:
    """
    For the warped mesh and the edited-abstraction mesh, voxelize with each
    :data:`ORIENTATION_TRANSFORMS` (applied **after** bbox normalization, before
    discretization) and render a short noisy point-cloud run so you can compare
    axis conventions. By default only **one** noise level (t≈0) is rendered to keep
    the sweep quick.

    Outputs under
    ``output_dir / "noisy_voxels_pointcloud" / "orientation_sweep" /`` in subfolders
    ``warped_mesh/`` and ``edited_abstraction/`` with filenames
    ``o<idx>_<oname>_<step>_t<...>.png`` plus a combined grid image per subfolder.
    """
    result_folder = Path(result_folder)
    out_root = output_dir / "noisy_voxels_pointcloud" / "orientation_sweep"
    wdir = out_root / "warped_mesh"
    adir = out_root / "edited_abstraction"
    wdir.mkdir(parents=True, exist_ok=True)
    adir.mkdir(parents=True, exist_ok=True)

    print("Noisy point cloud orientation sweep (voxelization convention)")
    print(f"  -> {out_root.resolve()}")
    print("  (4x4 from ORIENTATION_TRANSFORMS is applied to normalized mesh vertices)")

    # --- resolve warped mesh (same as visualize_noisy_voxels_pointcloud) ---
    transformed_dir = result_folder / "transformed_meshes"
    mesh_path = None
    if transformed_dir.is_dir():
        preferred = transformed_dir / "transformed_mesh_sq2.obj"
        if preferred.is_file():
            mesh_path = preferred
        else:
            for p in sorted(transformed_dir.glob("*.obj")) + sorted(
                transformed_dir.glob("*.ply")
            ):
                mesh_path = p
                break
    if mesh_path is None or not mesh_path.is_file():
        print(
            "  Warning: no warped mesh under transformed_meshes/; skipping warped_mesh sweep"
        )
    else:
        print(f"  Warped mesh: {mesh_path.name}")

    # --- resolve edited-abstraction JSON ---
    edited_json = None
    for name in (
        "edited_abstraction_final.json",
        "edited_abstraction.json",
        "edited_abstraction_iter1.json",
    ):
        p = result_folder / name
        if p.is_file():
            edited_json = p
            break
    tmp_abs: Optional[Path] = None
    if edited_json is not None:
        from abstraction import load_abstraction, mesh_from_abstraction

        edited = load_abstraction(str(edited_json))
        abs_mesh = mesh_from_abstraction(edited, resolution=30)
        if abs_mesh is None or abs_mesh.is_empty:
            print(
                "  Warning: empty edited abstraction mesh; skipping edited_abstraction sweep"
            )
        else:
            tmp_abs = adir / "_tmp_edited_abs_for_sweep.obj"
            abs_mesh.export(str(tmp_abs), file_type="obj")
            print(
                f"  Edited abstraction: {edited_json.name} -> temp mesh for voxelization"
            )
    else:
        print(
            "  Warning: no edited_abstraction_*.json; skipping edited_abstraction sweep"
        )

    num_steps = (
        1  # only need t=0 to judge axes; increase if you want a strip per candidate
    )

    def run_one_sweep(
        name: str,
        mesh_file: Path,
        dest_dir: Path,
        pcolor: np.ndarray,
    ) -> None:
        n_ok = 0
        for idx in sorted(ORIENTATION_TRANSFORMS.keys()):
            oname, mat = ORIENTATION_TRANSFORMS[idx]
            safe = _sanitize_sweep_label(oname)
            stem = f"o{int(idx):02d}_{safe}"
            expected_png = dest_dir / f"{stem}_00_t0.000.png"
            if skip_existing and expected_png.is_file():
                print(f"    skip existing {name} idx={idx} {oname}")
                n_ok += 1
                continue
            try:
                vox = _voxelize_mesh_path_cpu(
                    str(mesh_file),
                    post_normalize_transform_4x4=mat,
                )
            except Exception as e:
                print(f"    {name} idx={idx} {oname}: voxelize error: {e}")
                continue
            if not (vox > threshold).any().item():
                print(f"    {name} idx={idx} {oname}: empty voxels, skip")
                continue
            _ = _render_noisy_voxels_pointcloud_single(
                voxels_tensor=vox,
                output_dir=dest_dir,
                threshold=threshold,
                num_steps=num_steps,
                skip_existing=False,
                seed=seed,
                noise_std=noise_std,
                prefix=stem,
                desc=f"{name} o{idx:02d} {oname}",
                unmasked_uniform_color=pcolor,
            )
            n_ok += 1
        if n_ok:
            grid_png = (
                dest_dir.parent / f"{_sanitize_sweep_label(name)}_sweep_overview.png"
            )
            pngs = sorted(dest_dir.glob("o*.png"))
            if pngs:
                create_grid_visualization(
                    image_paths=[str(p) for p in pngs],
                    output_path=str(grid_png),
                    cols=5,
                    title=f"Orientation sweep: {name} (ORIENTATION_TRANSFORMS index, post-normalize)",
                )
                print(f"  Wrote overview grid: {grid_png.name}")

    if mesh_path is not None and mesh_path.is_file():
        run_one_sweep("warped_mesh", mesh_path, wdir, PCOLOR_BLUE)
    if tmp_abs is not None and tmp_abs.is_file():
        run_one_sweep("edited_abstraction", tmp_abs, adir, PCOLOR_PURPLE)
        try:
            tmp_abs.unlink()
        except OSError:
            pass

    print(
        "Done. Compare PNGs (and *_sweep_overview.png) and note the best ORIENTATION_TRANSFORMS index oname."
    )


def create_grid_visualization(
    image_paths: list,
    output_path: str,
    cols: int = 5,
    title: str = "Structure Inversion Timesteps",
):
    """
    Create a grid visualization of all timestep images.

    Args:
        image_paths: List of paths to rendered images
        output_path: Path to save the grid image
        cols: Number of columns in the grid
        title: Title for the figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    n_images = len(image_paths)
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    fig.suptitle(title, fontsize=16, fontweight="bold")

    # Flatten axes for easy indexing
    if rows == 1:
        axes = [axes] if cols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, ax in enumerate(axes):
        if i < n_images:
            img = mpimg.imread(image_paths[i])
            ax.imshow(img)
            # Extract timestep info from filename
            filename = Path(image_paths[i]).stem
            parts = filename.split("_")
            step_num = parts[1] if len(parts) > 1 else str(i)
            t_val = parts[2] if len(parts) > 2 else ""
            ax.set_title(f"Step {step_num}\n{t_val}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved grid visualization: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create visualizations for slides and papers"
    )
    parser.add_argument(
        "--result-folder",
        type=str,
        required=True,
        help="Path to the tredit result folder",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: result_folder/visualizations)",
    )
    parser.add_argument(
        "--timesteps",
        action="store_true",
        help="Visualize decoded latents at inversion timesteps",
    )
    parser.add_argument(
        "--noisy",
        action="store_true",
        help="Visualize noisy versions of original voxels at different noise levels",
    )
    parser.add_argument(
        "--noisy-pointcloud",
        action="store_true",
        help="Visualize noisy voxels using point cloud interpolation (voxels move from original to random Gaussian positions)",
    )
    parser.add_argument(
        "--abstraction",
        action="store_true",
        help="Visualize voxelized abstraction with colors",
    )
    parser.add_argument(
        "--categorized",
        action="store_true",
        help="Visualize abstraction colored by edit category (orange=unchanged, blue=edited, purple=added/deleted)",
    )
    parser.add_argument(
        "--categorized-mesh-resolution",
        type=int,
        default=1024,
        metavar="N",
        help="Pixel width and height for categorized_abstraction.png (recolored SQ mesh only; default: 1024). "
        "Use 512 for legacy output size.",
    )
    parser.add_argument(
        "--vlm-slide",
        action="store_true",
        help="Render all abstraction JSONs with JSON vertex colors: default + left/front/right/back views "
        "under <output>/vlm_slide/<json_stem>/ (see abstraction.render_multiview for cardinals).",
    )
    parser.add_argument(
        "--vlm-slide-resolution",
        type=int,
        default=30,
        metavar="N",
        help="Mesh resolution for --vlm-slide mesh_from_abstraction (default: 30).",
    )
    parser.add_argument(
        "--vlm-slide-default-sweep",
        action="store_true",
        help="With --vlm-slide: also write default_pose_sweep/ with extra vertex-rotation candidates (default: off).",
    )
    parser.add_argument(
        "--categorized-and-unit-sq",
        action="store_true",
        help="Shorthand for --categorized --include-unit-sq (mesh unit superquadric in blue, same view as categorized).",
    )
    parser.add_argument(
        "--include-unit-sq",
        action="store_true",
        help="With --categorized: also render unit_sq.ply and unit_sq.png (canonical superquadric, same Blender settings as categorized_abstraction).",
    )
    parser.add_argument(
        "--render-for-teaser",
        action="store_true",
        help="Run teaser rendering in addition to any other selected modes. "
        "By default, teaser rendering runs when no other mode is selected.",
    )
    parser.add_argument(
        "--no-teaser",
        action="store_true",
        help="Skip teaser rendering (use with --timesteps, --noisy, etc.). "
        "Has no effect if another mode is already selected (teaser is off in that case unless combined with --render-for-teaser).",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Create a grid visualization of timesteps/noisy voxels",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of columns in grid visualization (default: 5)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of noise levels for --noisy and --noisy-pointcloud (default: 10)",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=5.0,
        help="Standard deviation of Gaussian noise for --noisy-pointcloud in voxel units (default: 15.0)",
    )
    parser.add_argument(
        "--noisy-pointcloud-modes",
        type=str,
        default=None,
        metavar="LIST",
        help="Comma-separated subset of point-cloud outputs to render (default: all). "
        "Names: original, original_orange, inpainted, inpainted_gray, inpainted_no_purple, "
        "inpainted_no_blue, inpainted_no_blue_purple, warped_mesh, edited_abstraction, "
        "edited_abstraction_categorized, mask_changed, mask_added_deleted. "
        "Example: inpainted_no_blue.",
    )
    parser.add_argument(
        "--noisy-pc-added-deleted-extra-dilation",
        type=int,
        default=0,
        metavar="N",
        help="Extra 3D binary dilation iterations on added/deleted after the default 2 (mask-based point clouds).",
    )
    parser.add_argument(
        "--noisy-pc-changed-extra-dilation",
        type=int,
        default=0,
        metavar="N",
        help="Extra 3D binary dilation iterations on changed/edited (blue) mask after the default 2.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for voxel occupancy (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for noisy visualization (default: 42)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rendering if output already exists",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Force re-rendering of existing artifacts (overrides --skip-existing)",
    )
    parser.add_argument(
        "--color-unchanged",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        help="RGB color [0-255] for unchanged parts in categorized + teaser (default: 190 120 40 orange)",
    )
    parser.add_argument(
        "--color-changed",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        help="RGB color [0-255] for changed/edited parts in categorized view (default: 40 90 180 saturated blue)",
    )
    parser.add_argument(
        "--color-added-deleted",
        type=int,
        nargs=3,
        metavar=("R", "G", "B"),
        help="RGB color [0-255] for added/deleted parts in categorized view (default: 150 50 170 saturated purple)",
    )
    parser.add_argument(
        "--teaser-view-sweep",
        action="store_true",
        help="Render several Blender view-rotation candidates under teaser_renders/view_options/ "
        "so you can pick the correct one (exits after; requires existing teaser GLBs/OBJ).",
    )
    parser.add_argument(
        "--teaser-abstraction-view-sweep",
        action="store_true",
        help="Abstraction GLBs only: extended rotation candidates under teaser_renders/view_options_abstraction/ "
        "(vertex colors on; exits after).",
    )
    parser.add_argument(
        "--noisy-pointcloud-orientation-sweep",
        action="store_true",
        help="Voxelize warped mesh + edited abstraction with each ORIENTATION_TRANSFORMS (post-normalize), "
        "then render a single noisy point-cloud frame per case under "
        "noisy_voxels_pointcloud/orientation_sweep/ (exits after).",
    )
    parser.add_argument(
        "--teaser-only",
        type=str,
        choices=["abstractions"],
        default=None,
        help="With teaser rendering: only rebuild/render abstraction GLBs and PNGs (skip input mesh and pipeline GLBs).",
    )

    args = parser.parse_args()

    if args.categorized_and_unit_sq:
        args.categorized = True
        args.include_unit_sq = True

    explicit_modes = (
        args.timesteps
        or args.noisy
        or args.noisy_pointcloud
        or args.abstraction
        or args.categorized
        or args.noisy_pointcloud_orientation_sweep
        or args.vlm_slide
    )
    run_teaser = (args.render_for_teaser or not explicit_modes) and not args.no_teaser
    if not explicit_modes and args.no_teaser:
        print(
            "Error: --no-teaser was given but no other visualization mode was selected "
            "(--timesteps, --noisy, --noisy-pointcloud, --noisy-pointcloud-orientation-sweep, "
            "--abstraction, --categorized, or --vlm-slide).",
            file=sys.stderr,
        )
        sys.exit(1)

    result_folder = Path(args.result_folder)
    output_dir = Path(args.output) if args.output else result_folder / "visualizations"
    threshold = args.threshold

    # Determine skip_existing behavior: --override forces re-rendering
    skip_existing = args.skip_existing and not args.override

    if args.teaser_view_sweep:
        render_teaser_view_sweep(result_folder, skip_existing=skip_existing)
        sys.exit(0)

    if args.teaser_abstraction_view_sweep:
        render_teaser_abstraction_view_sweep(result_folder, skip_existing=skip_existing)
        sys.exit(0)

    if args.noisy_pointcloud_orientation_sweep:
        render_noisy_pointcloud_orientation_sweep(
            result_folder,
            output_dir,
            threshold=threshold,
            skip_existing=skip_existing,
            seed=args.seed,
            noise_std=args.noise_std,
        )
        sys.exit(0)

    print(f"Using threshold: {threshold}")
    if args.override:
        print("Override mode: will re-render existing artifacts")

    pipeline = None

    # Visualize inversion timesteps (decoded latents)
    if args.timesteps:
        timestep_output = output_dir / "timesteps"
        output_paths, pipeline = visualize_inversion_timesteps(
            result_folder=str(result_folder),
            output_dir=str(timestep_output),
            threshold=threshold,
            pipeline=pipeline,
            skip_existing=skip_existing,
        )

        # Create grid if requested
        if args.grid and output_paths:
            grid_path = timestep_output / "timesteps_grid.png"
            create_grid_visualization(
                image_paths=output_paths,
                output_path=str(grid_path),
                cols=args.cols,
                title="Structure Inversion: Decoded Voxels at Each Timestep",
            )

    # Visualize noisy voxels
    if args.noisy:
        noisy_output = output_dir / "noisy_voxels"
        output_paths = visualize_noisy_voxels(
            result_folder=str(result_folder),
            output_dir=str(noisy_output),
            threshold=threshold,
            num_steps=args.num_steps,
            skip_existing=skip_existing,
            seed=args.seed,
        )

        # Create grid if requested
        if args.grid and output_paths:
            grid_path = noisy_output / "noisy_grid.png"
            create_grid_visualization(
                image_paths=output_paths,
                output_path=str(grid_path),
                cols=args.cols,
                title="Noisy Voxels: Original + Noise at Different Levels",
            )

    # Visualize noisy voxels with point cloud interpolation
    if args.noisy_pointcloud:
        noisy_pc_output = output_dir / "noisy_voxels_pointcloud"
        _pc_only = None
        if args.noisy_pointcloud_modes:
            _pc_only = {
                s.strip() for s in args.noisy_pointcloud_modes.split(",") if s.strip()
            }
        all_output_paths = visualize_noisy_voxels_pointcloud(
            result_folder=str(result_folder),
            output_dir=str(noisy_pc_output),
            threshold=threshold,
            num_steps=args.num_steps,
            skip_existing=skip_existing,
            seed=args.seed,
            noise_std=args.noise_std,
            only_modes=_pc_only,
            added_deleted_extra_dilation=args.noisy_pc_added_deleted_extra_dilation,
            changed_extra_dilation=args.noisy_pc_changed_extra_dilation,
        )

        # Create grids if requested - one for each output mode
        if args.grid:
            if all_output_paths.get("original"):
                grid_path = noisy_pc_output / "original" / "noisy_pointcloud_grid.png"
                create_grid_visualization(
                    image_paths=all_output_paths["original"],
                    output_path=str(grid_path),
                    cols=args.cols,
                    title="Noisy Original Voxels (Point Cloud, Gray)",
                )
            if all_output_paths.get("original_orange"):
                grid_path = (
                    noisy_pc_output / "original_orange" / "noisy_pointcloud_grid.png"
                )
                create_grid_visualization(
                    image_paths=all_output_paths["original_orange"],
                    output_path=str(grid_path),
                    cols=args.cols,
                    title="Noisy Original Voxels (Point Cloud, Orange)",
                )
            if all_output_paths.get("inpainted"):
                grid_path = noisy_pc_output / "inpainted" / "noisy_pointcloud_grid.png"
                create_grid_visualization(
                    image_paths=all_output_paths["inpainted"],
                    output_path=str(grid_path),
                    cols=args.cols,
                    title="Noisy Inpainted Voxels (Colored)",
                )
            if all_output_paths.get("inpainted_gray"):
                grid_path = (
                    noisy_pc_output / "inpainted_gray" / "noisy_pointcloud_grid.png"
                )
                create_grid_visualization(
                    image_paths=all_output_paths["inpainted_gray"],
                    output_path=str(grid_path),
                    cols=args.cols,
                    title="Noisy Inpainted Voxels (Gray)",
                )
            if all_output_paths.get("inpainted_no_purple"):
                grid_path = (
                    noisy_pc_output
                    / "inpainted_no_purple"
                    / "noisy_pointcloud_grid.png"
                )
                create_grid_visualization(
                    image_paths=all_output_paths["inpainted_no_purple"],
                    output_path=str(grid_path),
                    cols=args.cols,
                    title="Noisy Inpainted Voxels (added/deleted as gray)",
                )
            if all_output_paths.get("inpainted_no_blue"):
                grid_path = (
                    noisy_pc_output / "inpainted_no_blue" / "noisy_pointcloud_grid.png"
                )
                create_grid_visualization(
                    image_paths=all_output_paths["inpainted_no_blue"],
                    output_path=str(grid_path),
                    cols=args.cols,
                    title="Noisy Inpainted Voxels (no blue: orange, purple, gray for changed)",
                )
            if all_output_paths.get("inpainted_no_blue_purple"):
                grid_path = (
                    noisy_pc_output
                    / "inpainted_no_blue_purple"
                    / "noisy_pointcloud_grid.png"
                )
                create_grid_visualization(
                    image_paths=all_output_paths["inpainted_no_blue_purple"],
                    output_path=str(grid_path),
                    cols=args.cols,
                    title="Noisy Inpainted Voxels (changed & added/deleted as gray)",
                )
            if all_output_paths.get("warped_mesh"):
                grid_path = (
                    noisy_pc_output / "warped_mesh" / "noisy_pointcloud_grid.png"
                )
                create_grid_visualization(
                    image_paths=all_output_paths["warped_mesh"],
                    output_path=str(grid_path),
                    cols=args.cols,
                    title="Noisy Warped Mesh (Voxelized, Blue)",
                )
            if all_output_paths.get("edited_abstraction"):
                grid_path = (
                    noisy_pc_output / "edited_abstraction" / "noisy_pointcloud_grid.png"
                )
                create_grid_visualization(
                    image_paths=all_output_paths["edited_abstraction"],
                    output_path=str(grid_path),
                    cols=args.cols,
                    title="Noisy Edited Abstraction (Voxelized, Purple)",
                )
            if all_output_paths.get("edited_abstraction_categorized"):
                grid_path = (
                    noisy_pc_output
                    / "edited_abstraction_categorized"
                    / "noisy_pointcloud_grid.png"
                )
                create_grid_visualization(
                    image_paths=all_output_paths["edited_abstraction_categorized"],
                    output_path=str(grid_path),
                    cols=args.cols,
                    title="Noisy Edited Abstraction (Voxelized, Inpainted Mask Colors)",
                )
            if all_output_paths.get("mask_changed"):
                grid_path = noisy_pc_output / "mask_changed" / "mask_volume_grid.png"
                create_grid_visualization(
                    image_paths=all_output_paths["mask_changed"],
                    output_path=str(grid_path),
                    cols=args.cols,
                    title="Changed mask volume (blue)",
                )
            if all_output_paths.get("mask_added_deleted"):
                grid_path = (
                    noisy_pc_output / "mask_added_deleted" / "mask_volume_grid.png"
                )
                create_grid_visualization(
                    image_paths=all_output_paths["mask_added_deleted"],
                    output_path=str(grid_path),
                    cols=args.cols,
                    title="Added/deleted mask volume (purple)",
                )

    # Visualize abstraction voxels
    if args.abstraction:
        visualize_abstraction_voxels(
            result_folder=str(result_folder),
            output_dir=str(output_dir),
            skip_existing=skip_existing,
        )

    # Visualize categorized abstraction
    if args.categorized:
        visualize_categorized_abstraction(
            result_folder=str(result_folder),
            output_dir=str(output_dir),
            skip_existing=skip_existing,
            color_unchanged=args.color_unchanged,
            color_changed=args.color_changed,
            color_added_deleted=args.color_added_deleted,
            include_unit_sq=bool(
                getattr(args, "include_unit_sq", False) or args.categorized_and_unit_sq
            ),
            categorized_mesh_render_resolution=args.categorized_mesh_resolution,
        )

    # VLM slide: multiview abstraction JSONs with JSON colors
    if args.vlm_slide:
        vlm_slide_dir = output_dir / "vlm_slide"
        render_vlm_slide(
            result_folder=str(result_folder),
            output_dir=str(vlm_slide_dir),
            skip_existing=skip_existing,
            mesh_resolution=args.vlm_slide_resolution,
            default_pose_sweep=args.vlm_slide_default_sweep,
        )

    # Teaser rendering (default when no other mode is selected; combine with --render-for-teaser for multi-mode runs)
    if run_teaser:
        render_for_teaser(
            teaser_folder=str(result_folder),
            output_dir=args.output,
            skip_existing=skip_existing,
            color_unchanged=args.color_unchanged,
            color_changed=args.color_changed,
            color_added_deleted=args.color_added_deleted,
            teaser_only=args.teaser_only,
        )
