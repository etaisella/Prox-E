#!/usr/bin/env python3
"""
Prepare data for inpainting: compare abstractions, generate masks and transformations.

This module compares original and edited superquadric abstractions to produce
masks and transformation matrices needed for the latent-space inpainting pipeline.

Categories:
- unchanged: superquadrics identical in both
- added_deleted: superquadrics in one but not the other
- changed_original: original version of modified superquadrics
- changed_edited: edited version of modified superquadrics
"""

import json
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.interpolate import RegularGridInterpolator

# Import mesh generation from abstraction module
from prox_e.abstraction import mesh_from_abstraction, load_abstraction
from prox_e.utils import transform_voxels_to_trellis


def superquadric_to_key(sq: dict, tolerance: float = 0.01) -> tuple:
    """Create a hashable key from superquadric parameters for comparison."""

    def round_val(v):
        return round(v / tolerance) * tolerance

    scale = tuple(round_val(s) for s in sq["scale"])
    trans = tuple(round_val(t) for t in sq["translation"])
    exponents = tuple(round_val(e) for e in sq["exponents"])
    rot = tuple(round_val(r) for row in sq["rotation"] for r in row)

    return (scale, trans, exponents, rot)


def exponents_match(sq1: dict, sq2: dict, tolerance: float = 0.01) -> bool:
    """Check if two superquadrics have matching exponents within tolerance."""
    e1 = sq1["exponents"]
    e2 = sq2["exponents"]
    return abs(e1[0] - e2[0]) <= tolerance and abs(e1[1] - e2[1]) <= tolerance


def get_next_available_index(all_superquadrics: List[dict]) -> int:
    """Get the next available index for a new superquadric."""
    if not all_superquadrics:
        return 0
    max_index = max(sq.get("index", 0) for sq in all_superquadrics)
    return max_index + 1


def compare_abstractions(
    original: List[dict], edited: List[dict], tolerance: float = 0.01
) -> Dict[str, List[dict]]:
    """
    Compare original and edited abstractions to categorize superquadrics.

    Note: If exponents change for a superquadric, it cannot be transformed
    (the shape itself changed). Such SQs are treated as deleted+added instead
    of changed, with the edited SQ getting a new unique index.
    """
    orig_by_key = {superquadric_to_key(sq, tolerance): sq for sq in original}
    edit_by_key = {superquadric_to_key(sq, tolerance): sq for sq in edited}

    orig_keys = set(orig_by_key.keys())
    edit_keys = set(edit_by_key.keys())

    unchanged_keys = orig_keys & edit_keys
    unchanged = [orig_by_key[k] for k in unchanged_keys]

    deleted_keys = orig_keys - edit_keys
    deleted = [orig_by_key[k] for k in deleted_keys]

    added_keys = edit_keys - orig_keys
    added = [edit_by_key[k] for k in added_keys]

    changed_original = []
    changed_edited = []
    truly_added = []
    truly_deleted = []

    orig_by_idx = {sq["index"]: sq for sq in deleted}
    edit_by_idx = {sq["index"]: sq for sq in added}

    matched_orig_indices = set()
    matched_edit_indices = set()

    # Track next available index for SQs with changed exponents
    next_index = get_next_available_index(original + edited)

    for idx in orig_by_idx:
        if idx in edit_by_idx:
            sq_orig = orig_by_idx[idx]
            sq_edit = edit_by_idx[idx]

            # Check if exponents changed - if so, treat as delete + add
            if not exponents_match(sq_orig, sq_edit, tolerance):
                print(
                    f"  [compare_abstractions] SQ {idx} has changed exponents: "
                    f"{sq_orig['exponents']} -> {sq_edit['exponents']}"
                )
                print(
                    f"    -> Treating as deleted (idx {idx}) + added (new idx {next_index})"
                )

                # Original goes to deleted
                truly_deleted.append(sq_orig)

                # Create a copy of edited SQ with new index
                sq_edit_new = sq_edit.copy()
                sq_edit_new["index"] = next_index
                truly_added.append(sq_edit_new)
                next_index += 1
            else:
                # Exponents match - can compute transformation
                changed_original.append(sq_orig)
                changed_edited.append(sq_edit)

            matched_orig_indices.add(idx)
            matched_edit_indices.add(idx)

    for sq in deleted:
        if sq["index"] not in matched_orig_indices:
            truly_deleted.append(sq)

    for sq in added:
        if sq["index"] not in matched_edit_indices:
            truly_added.append(sq)

    return {
        "unchanged": unchanged,
        "added": truly_added,
        "deleted": truly_deleted,
        "added_deleted": truly_added + truly_deleted,
        "changed_original": changed_original,
        "changed_edited": changed_edited,
    }


def get_superquadric_matrix(sq: dict) -> np.ndarray:
    """Build a 4x4 transformation matrix for a superquadric (Scale -> Rotate -> Translate)."""
    S = np.eye(4)
    S[0, 0], S[1, 1], S[2, 2] = sq["scale"]

    R = np.eye(4)
    R[:3, :3] = np.array(sq["rotation"])

    T = np.eye(4)
    T[:3, 3] = sq["translation"]

    return T @ R @ S


def points_inside_superquadric(
    points: np.ndarray, sq: dict, margin: float = 1.1
) -> np.ndarray:
    """
    Test which points are inside (or near) a superquadric.

    Args:
        points: (N, 3) array of world-space points
        sq: Superquadric dict with scale, rotation, translation, exponents
        margin: Multiplier for the inside test (1.0 = exact boundary, 1.1 = 10% larger)

    Returns:
        (N,) boolean array, True if point is inside
    """
    M = get_superquadric_matrix(sq)
    M_inv = np.linalg.inv(M)

    # Transform points to local superquadric coordinates (unit sphere space)
    points_homo = np.hstack([points, np.ones((len(points), 1))])
    local_pts = (M_inv @ points_homo.T).T[:, :3]

    e1, e2 = sq["exponents"]
    x, y, z = local_pts[:, 0], local_pts[:, 1], local_pts[:, 2]

    # Superquadric implicit function: (|x|^(2/e2) + |y|^(2/e2))^(e2/e1) + |z|^(2/e1) <= 1
    eps = 1e-8
    exp_xy = 2.0 / (e2 + eps)
    exp_z = 2.0 / (e1 + eps)
    exp_outer = e2 / (e1 + eps)

    term_xy = (np.abs(x) ** exp_xy + np.abs(y) ** exp_xy) ** exp_outer
    term_z = np.abs(z) ** exp_z
    f_val = term_xy + term_z

    return f_val <= (margin**exp_z)


def make_voxel_to_normalized(resolution: int) -> np.ndarray:
    """
    Create 4x4 matrix to convert voxel indices to normalized world coordinates.
    Voxel index v maps to: (v + 0.5) / resolution - 0.5
    """
    offset = 0.5 / resolution - 0.5
    M = np.eye(4)
    M[0, 0] = M[1, 1] = M[2, 2] = 1.0 / resolution
    M[0, 3] = M[1, 3] = M[2, 3] = offset
    return M


def make_normalized_to_voxel(resolution: int) -> np.ndarray:
    """
    Create 4x4 matrix to convert normalized world coordinates to voxel indices.
    Inverse of voxel_to_normalized.
    """
    offset = 0.5 / resolution - 0.5
    M = np.eye(4)
    M[0, 0] = M[1, 1] = M[2, 2] = resolution
    M[0, 3] = M[1, 3] = M[2, 3] = -offset * resolution
    return M


def make_normalized_to_original(
    global_center: np.ndarray, global_scale: float
) -> np.ndarray:
    """
    Create 4x4 matrix to convert normalized world to original world coordinates.
    orig = norm * global_scale + global_center

    Convention: global_scale = bbox_size (direct, not inverse)
    """
    M = np.eye(4)
    M[0, 0] = M[1, 1] = M[2, 2] = global_scale
    M[0, 3], M[1, 3], M[2, 3] = global_center
    return M


def make_original_to_normalized(
    global_center: np.ndarray, global_scale: float
) -> np.ndarray:
    """
    Create 4x4 matrix to convert original world to normalized world coordinates.
    norm = (orig - global_center) / global_scale

    Convention: global_scale = bbox_size (direct, not inverse)
    """
    M = np.eye(4)
    M[0, 0] = M[1, 1] = M[2, 2] = 1.0 / global_scale
    M[0, 3] = -global_center[0] / global_scale
    M[1, 3] = -global_center[1] / global_scale
    M[2, 3] = -global_center[2] / global_scale
    return M


def build_voxel_transformation(
    sq_orig: dict,
    sq_edit: dict,
    global_center: np.ndarray,
    global_scale: float,
    resolution: int,
) -> np.ndarray:
    """
    Build a 4x4 transformation matrix that maps voxel indices from edited space to original space.

    The full chain: voxel_edit → normalized → original → transform → original → normalized → voxel_orig
    """
    V2N = make_voxel_to_normalized(resolution)
    N2O = make_normalized_to_original(global_center, global_scale)
    O2N = make_original_to_normalized(global_center, global_scale)
    N2V = make_normalized_to_voxel(resolution)

    Mat_orig = get_superquadric_matrix(sq_orig)
    Mat_edit = get_superquadric_matrix(sq_edit)
    Mat_edit_inv = np.linalg.inv(Mat_edit)

    # Transform in original space: edited point → original point
    M_edit_to_orig = Mat_orig @ Mat_edit_inv

    # Full chain: voxel → normalized → original → transform → original → normalized → voxel
    return N2V @ O2N @ M_edit_to_orig @ N2O @ V2N


def interpolate_grid_values(
    source_coordinates: np.ndarray,
    grid_values: np.ndarray,
    transformation_matrix: np.ndarray,
    order: int = 1,
) -> np.ndarray:
    """
    Interpolate values from a grid at transformed source coordinates.

    Args:
        source_coordinates: Array of shape (N, 3) with source coordinates.
        grid_values: 3D or 4D numpy array containing values to interpolate from.
        transformation_matrix: 4x4 transformation matrix to apply to source coordinates.
        order: Interpolation order (0 = nearest neighbor, 1 = linear).

    Returns:
        Array of shape (N,) for 3D grid or (N, C) for 4D grid with interpolated values.
    """
    N = source_coordinates.shape[0]

    # Convert to homogeneous coordinates [x, y, z, 1]
    ones = np.ones((N, 1))
    coords_homogeneous = np.hstack((source_coordinates, ones))  # (N, 4)

    # Apply transformation
    transformed_coords = (transformation_matrix @ coords_homogeneous.T).T  # (N, 4)
    transformed_coords = transformed_coords[:, :3]  # (N, 3)

    # Get grid dimensions
    if grid_values.ndim == 3:
        resolution = grid_values.shape[0]
    else:  # 4D
        resolution = grid_values.shape[0]

    # Clip transformed coordinates to valid grid bounds
    transformed_coords = np.clip(transformed_coords, 0, resolution - 1)

    # Create grid axes (0 to resolution-1)
    x = np.arange(resolution)
    y = np.arange(resolution)
    z = np.arange(resolution)

    # Choose interpolation method based on order
    method = "nearest" if order == 0 else "linear"

    if grid_values.ndim == 3:
        # Create interpolator
        interpolator = RegularGridInterpolator(
            (x, y, z), grid_values, method=method, bounds_error=False, fill_value=0.0
        )
        interpolated = interpolator(transformed_coords)
    elif grid_values.ndim == 4:
        num_channels = grid_values.shape[-1]
        interpolated = np.zeros((N, num_channels), dtype=np.float32)
        for c in range(num_channels):
            interpolator = RegularGridInterpolator(
                (x, y, z),
                grid_values[:, :, :, c],
                method=method,
                bounds_error=False,
                fill_value=0.0,
            )
            interpolated[:, c] = interpolator(transformed_coords)
    else:
        raise ValueError(f"grid_values must be 3D or 4D, got {grid_values.ndim}D")

    return interpolated


def create_color_grid(resolution: int = 64) -> np.ndarray:
    """Create a 3D color grid where each position maps to a color."""
    grid = np.zeros((resolution, resolution, resolution, 3), dtype=np.float32)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                normalized_pos = np.array([i, j, k]) / (resolution - 1)
                grid[i, j, k] = position_to_color(normalized_pos)
    return grid


def create_mesh_from_superquadrics(
    superquadrics: List[dict], resolution: int = 30
) -> Optional[trimesh.Trimesh]:
    """Create a merged mesh from a list of superquadrics."""
    if not superquadrics:
        return None
    return mesh_from_abstraction(superquadrics, resolution=resolution)


def trimesh_to_open3d(mesh: trimesh.Trimesh) -> o3d.geometry.TriangleMesh:
    """Convert trimesh to open3d mesh."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh


def voxelize_mesh(
    mesh: Union[trimesh.Trimesh, o3d.geometry.TriangleMesh], resolution: int = 64
) -> np.ndarray:
    """Voxelize a mesh to a binary 3D grid."""
    if isinstance(mesh, trimesh.Trimesh):
        o3d_mesh = trimesh_to_open3d(mesh)
    else:
        o3d_mesh = mesh

    vertices = np.clip(np.asarray(o3d_mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)

    voxel_size = 1.0 / resolution
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        o3d_mesh,
        voxel_size=voxel_size,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )

    voxels = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    for voxel in voxel_grid.get_voxels():
        idx = voxel.grid_index
        if (
            0 <= idx[0] < resolution
            and 0 <= idx[1] < resolution
            and 0 <= idx[2] < resolution
        ):
            voxels[idx[0], idx[1], idx[2]] = 1.0

    return voxels


def position_to_color(normalized_pos: np.ndarray) -> np.ndarray:
    """Convert a normalized position (0-1 range) to a vibrant, distinct color using HSV."""
    x, y, z = normalized_pos
    hue = (x * 0.4 + y * 0.35 + z * 0.25) % 1.0
    saturation = 0.85
    value = 0.7 + 0.3 * ((x + y + z) / 3.0)

    h_i = int(hue * 6)
    f = hue * 6 - h_i
    p = value * (1 - saturation)
    q = value * (1 - f * saturation)
    t = value * (1 - (1 - f) * saturation)

    if h_i == 0:
        r, g, b = value, t, p
    elif h_i == 1:
        r, g, b = q, value, p
    elif h_i == 2:
        r, g, b = p, value, t
    elif h_i == 3:
        r, g, b = p, q, value
    elif h_i == 4:
        r, g, b = t, p, value
    else:
        r, g, b = value, p, q

    return np.array([r, g, b])


def voxels_to_cube_mesh(
    voxels: np.ndarray,
    colors: Optional[np.ndarray] = None,
    use_position_colors: bool = True,
) -> o3d.geometry.TriangleMesh:
    """
    Convert a binary voxel grid to a mesh of cubes.

    Args:
        voxels: Binary 3D numpy array
        colors: Optional (N, 3) array of colors for each occupied voxel
        use_position_colors: If True and colors is None, use position-based colors
    """
    resolution = voxels.shape[0]
    voxel_size = 1.0 / resolution

    occupied = np.argwhere(voxels > 0.5)
    if len(occupied) == 0:
        return o3d.geometry.TriangleMesh()

    template = o3d.geometry.TriangleMesh.create_box(
        width=voxel_size, height=voxel_size, depth=voxel_size
    )
    template.translate((-voxel_size / 2, -voxel_size / 2, -voxel_size / 2))
    num_verts_per_cube = len(template.vertices)

    merged_mesh = o3d.geometry.TriangleMesh()
    all_colors = []

    for i, idx in enumerate(occupied):
        grid_pos = (idx / resolution) - 0.5 + (voxel_size / 2)
        pos = np.array(
            [grid_pos[0], -grid_pos[2], grid_pos[1]]
        )  # Axis swap for rendering

        cube = o3d.geometry.TriangleMesh(template)
        cube.translate(pos)
        merged_mesh += cube

        if colors is not None:
            cube_color = colors[i]
        elif use_position_colors:
            normalized_pos = idx / (resolution - 1)
            cube_color = position_to_color(normalized_pos)
        else:
            cube_color = np.array([0.5, 0.5, 0.5])

        all_colors.extend([cube_color] * num_verts_per_cube)

    merged_mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(all_colors))
    merged_mesh.compute_vertex_normals()

    return merged_mesh


def compute_global_normalization(
    original: List[dict], edited: List[dict], mesh_resolution: int = 30
) -> Tuple[np.ndarray, float, Optional[trimesh.Trimesh]]:
    """
    Compute global center and scale from all superquadrics.

    Returns:
        Tuple of (global_center, global_scale, reference_mesh)
    """
    all_superquadrics = original + edited
    reference_mesh = create_mesh_from_superquadrics(
        all_superquadrics, resolution=mesh_resolution
    )
    if reference_mesh is None:
        return None, None, None

    vertices = reference_mesh.vertices
    aabb = np.stack([vertices.min(0), vertices.max(0)])
    global_center = (aabb[0] + aabb[1]) / 2
    global_scale = (aabb[1] - aabb[0]).max() + 1e-6  # bbox_size (direct convention)

    return global_center, global_scale, reference_mesh


def prepare_edit_data(
    original_json: str,
    edited_json: str,
    output_folder: str,
    latent_resolution: int = 16,
    voxel_resolution: int = 64,
    mesh_resolution: int = 30,
    render_fn=None,
    save_visualizations: bool = True,
    run_verification: bool = False,
    create_transformed_meshes: bool = False,
    original_mesh_path: str = None,
    external_center: np.ndarray = None,
    external_scale: float = None,
    no_transformed_in_scale: bool = True,
) -> Optional[Dict]:
    """
    Unified function to prepare all data needed for editing pipeline.

    Compares original and edited abstractions, generates masks at both latent
    and voxel resolution, optionally saves visualizations, and returns all
    data needed for latent merging.

    Args:
        original_json: Path to original abstraction JSON
        edited_json: Path to edited abstraction JSON
        output_folder: Base output folder for saving outputs
        latent_resolution: Resolution for latent-space masks (default 16)
        voxel_resolution: Resolution for voxel-space masks (default 64)
        mesh_resolution: Resolution for superquadric meshing
        render_fn: Optional render function (mesh_path, output_path)
        save_visualizations: Whether to save voxel mesh visualizations
        run_verification: Whether to run transformation/interpolation checks
        create_transformed_meshes: If True, create world-space transformed meshes
        original_mesh_path: Path to original mesh (required if create_transformed_meshes=True)
        external_center: If provided, use this center instead of computing from superquadrics
        external_scale: If provided, use this scale instead of computing from superquadrics.
                        NOTE: This uses the INVERSE convention (1/bbox_size) to match internal usage.

    Returns:
        Dict containing:
        - changed_edited_masks: List of dicts with 'mask', 'mask_voxel', 'M_latent', 'sq_index'
        - changed_original_mask: Union mask for changed original superquadrics (latent res)
        - changed_original_mask_voxel: Union mask at voxel resolution
        - added_deleted_mask: Union mask for added/deleted superquadrics (latent res)
        - added_deleted_mask_voxel: Union mask at voxel resolution
        - num_changed: Number of changed superquadrics
        - num_added_deleted: Number of added/deleted superquadrics
        - latent_resolution: Resolution used for latent masks
        - voxel_resolution: Resolution used for voxel masks
        - global_center: Global normalization center
        - global_scale: Global normalization scale
        - categories: Dict of categorized superquadrics
        - transformed_mesh_path: Path to combined transformed mesh (if created)
    """
    output_path = Path(output_folder)
    masks_folder = output_path / "latent_masks"
    masks_folder.mkdir(parents=True, exist_ok=True)

    # === STEP 1: Load and compare abstractions ===
    print("[PrepareEditData] Loading abstractions...")
    original = load_abstraction(original_json)
    edited = load_abstraction(edited_json)
    print(f"  Original: {len(original)} superquadrics")
    print(f"  Edited: {len(edited)} superquadrics")

    print("[PrepareEditData] Comparing abstractions...")
    categories = compare_abstractions(original, edited)
    for cat, sqs in categories.items():
        print(f"  {cat}: {len(sqs)} superquadrics")

    changed_orig_list = categories.get("changed_original", [])
    changed_edit_list = categories.get("changed_edited", [])
    added_deleted_list = categories.get("added_deleted", [])
    unchanged_list = categories.get("unchanged", [])

    # === STEP 2: Create transformed meshes FIRST (needed for proper normalization) ===
    transformed_mesh_paths = {}
    combined_transformed_mesh_path = None

    if create_transformed_meshes and changed_orig_list and changed_edit_list:
        if original_mesh_path is None:
            raise ValueError(
                "original_mesh_path is required when create_transformed_meshes=True"
            )

        print("[PrepareEditData] Creating world-space transformed meshes...")

        # Load original mesh
        original_mesh = trimesh.load(original_mesh_path, force="mesh")
        original_vertices = np.array(original_mesh.vertices)

        # List to collect all meshes for combined visualization
        all_meshes = []
        offset_step = 1.1

        # Add original mesh at position 0 with gray color
        original_copy = original_mesh.copy()
        gray_color = np.array([180, 180, 180, 255], dtype=np.uint8)
        original_copy.visual = trimesh.visual.ColorVisuals(
            mesh=original_copy,
            vertex_colors=np.tile(gray_color, (len(original_copy.vertices), 1)),
        )
        all_meshes.append(original_copy)

        transformed_meshes_folder = output_path / "transformed_meshes"
        transformed_meshes_folder.mkdir(parents=True, exist_ok=True)

        # Piecewise warped mesh: each vertex warped only by its local sq transformation
        piecewise_vertices = original_vertices.copy()
        vertex_warped = np.zeros(len(original_vertices), dtype=bool)

        for i, (sq_orig, sq_edit) in enumerate(
            zip(changed_orig_list, changed_edit_list)
        ):
            # Get world-space transformation matrices
            M_orig = get_superquadric_matrix(sq_orig)
            M_edit = get_superquadric_matrix(sq_edit)

            # Transformation from original to edited: M_edit @ M_orig_inv
            M_orig_inv = np.linalg.inv(M_orig)
            M_transform = M_edit @ M_orig_inv

            sq_index = sq_orig.get("index", i)
            sq_color = sq_edit.get(
                "color", [128, 128, 128]
            )  # Use color from abstraction JSON
            print(
                f"  SQ {sq_index}: orig_trans={sq_orig['translation']}, edit_trans={sq_edit['translation']}, color={sq_color}"
            )

            # === Individual fully-transformed mesh ===
            # Transform ALL vertices
            vertices_homo = np.hstack(
                [original_vertices, np.ones((len(original_vertices), 1))]
            )
            transformed_vertices = (M_transform @ vertices_homo.T).T[:, :3]

            # Save individual transformed mesh
            individual_mesh = trimesh.Trimesh(
                vertices=transformed_vertices.copy(), faces=original_mesh.faces
            )
            individual_mesh_path = (
                transformed_meshes_folder / f"transformed_mesh_sq{sq_index}.obj"
            )
            individual_mesh.export(str(individual_mesh_path))
            transformed_mesh_paths[sq_index] = str(individual_mesh_path)
            print(f"    Saved: {individual_mesh_path}")

            # Offset for combined visualization
            x_offset = (i + 1) * offset_step
            transformed_vertices[:, 0] += x_offset
            transformed_mesh = trimesh.Trimesh(
                vertices=transformed_vertices, faces=original_mesh.faces
            )
            # Assign light gray color to transformed mesh
            light_gray = np.array([200, 200, 200, 255], dtype=np.uint8)
            transformed_mesh.visual = trimesh.visual.ColorVisuals(
                mesh=transformed_mesh,
                vertex_colors=np.tile(light_gray, (len(transformed_mesh.vertices), 1)),
            )
            all_meshes.append(transformed_mesh)

            # Add edited superquadric mesh at same offset (uses colors from mesh_from_abstraction)
            sq_edit_mesh = create_mesh_from_superquadrics(
                [sq_edit], resolution=mesh_resolution
            )
            if sq_edit_mesh is not None:
                sq_edit_mesh.vertices[:, 0] += x_offset
                visual_type = type(sq_edit_mesh.visual).__name__
                has_colors = (
                    hasattr(sq_edit_mesh.visual, "vertex_colors")
                    and sq_edit_mesh.visual.vertex_colors is not None
                )
                print(
                    f"    SQ mesh visual type: {visual_type}, has vertex colors: {has_colors}"
                )
                if has_colors:
                    print(
                        f"    First vertex color: {sq_edit_mesh.visual.vertex_colors[0]}"
                    )
                all_meshes.append(sq_edit_mesh)

            # === Piecewise warping: only vertices inside this sq ===
            inside_mask = points_inside_superquadric(
                original_vertices, sq_orig, margin=1.5
            )
            to_warp = inside_mask & ~vertex_warped

            if np.any(to_warp):
                pts_to_warp = original_vertices[to_warp]
                pts_homo = np.hstack([pts_to_warp, np.ones((len(pts_to_warp), 1))])
                warped_pts = (M_transform @ pts_homo.T).T[:, :3]

                piecewise_vertices[to_warp] = warped_pts
                vertex_warped[to_warp] = True
                print(f"    Piecewise: warped {np.sum(to_warp)} vertices")

        print(
            f"  Piecewise total: {np.sum(vertex_warped)} / {len(original_vertices)} vertices warped"
        )

        # Save piecewise mesh
        piecewise_mesh = trimesh.Trimesh(
            vertices=piecewise_vertices, faces=original_mesh.faces
        )
        piecewise_path = transformed_meshes_folder / "piecewise_warped_mesh.obj"
        piecewise_mesh.export(str(piecewise_path))
        print(f"    Saved: {piecewise_path}")

        # Add to combined visualization with offset
        piecewise_offset = (len(changed_orig_list) + 1) * offset_step
        piecewise_vis = piecewise_mesh.copy()
        piecewise_vis.vertices[:, 0] += piecewise_offset
        cyan_color = np.array([100, 200, 200, 255], dtype=np.uint8)
        piecewise_vis.visual = trimesh.visual.ColorVisuals(
            mesh=piecewise_vis,
            vertex_colors=np.tile(cyan_color, (len(piecewise_vis.vertices), 1)),
        )
        all_meshes.append(piecewise_vis)

        # Save combined mesh (PLY format to preserve vertex colors)
        combined_mesh = trimesh.util.concatenate(all_meshes)
        combined_path = output_path / "transformed_original_meshes.ply"
        combined_mesh.export(str(combined_path), file_type="ply")
        combined_transformed_mesh_path = str(combined_path)
        print(f"[PrepareEditData] Saved combined mesh: {combined_path}")
        print(f"  Visual type: {type(combined_mesh.visual).__name__}")
        if (
            hasattr(combined_mesh.visual, "vertex_colors")
            and combined_mesh.visual.vertex_colors is not None
        ):
            print(f"  Vertex colors shape: {combined_mesh.visual.vertex_colors.shape}")

    # === STEP 3: Compute global normalization (from meshes, not superquadrics) ===
    # This ensures transformed meshes fit within the normalized bounds
    if external_center is not None and external_scale is not None:
        print("[PrepareEditData] Using external normalization...")
        global_center = external_center
        global_scale = external_scale
    else:
        print("[PrepareEditData] Computing global normalization from meshes...")
        # Collect all mesh paths for normalization
        mesh_paths_for_norm = []
        if original_mesh_path is not None:
            mesh_paths_for_norm.append(original_mesh_path)
        # Add edited abstraction mesh (from superquadrics)
        abstraction_mesh = create_mesh_from_superquadrics(
            edited, resolution=mesh_resolution
        )
        if abstraction_mesh is not None:
            abstraction_temp_path = output_path / "temp_abstraction_for_norm.obj"
            abstraction_mesh.export(str(abstraction_temp_path))
            mesh_paths_for_norm.append(str(abstraction_temp_path))
        # Add transformed meshes
        if not no_transformed_in_scale:
            for sq_index, mesh_path in transformed_mesh_paths.items():
                if Path(mesh_path).exists():
                    mesh_paths_for_norm.append(mesh_path)

        # Compute unified bounding box from all meshes
        all_vertices = []
        for mesh_path in mesh_paths_for_norm:
            try:
                mesh = trimesh.load(mesh_path, force="mesh")
                all_vertices.append(np.array(mesh.vertices))
            except Exception as e:
                print(f"  Warning: Could not load {mesh_path}: {e}")

        combined_vertices = np.concatenate(all_vertices, axis=0)
        bbox_min = combined_vertices.min(axis=0)
        bbox_max = combined_vertices.max(axis=0)
        print(f"  Bbox min: {bbox_min}")
        print(f"  Bbox max: {bbox_max}")
        global_center = (bbox_min + bbox_max) / 2
        bbox_size = (bbox_max - bbox_min).max()
        global_scale = bbox_size + 1e-6  # Direct scale (bbox_size convention)
        print(f"  Computed from {len(mesh_paths_for_norm)} meshes")

    print(f"  Global center: {global_center}")
    print(f"  Global scale: {global_scale}")

    # Store reference mesh for visualization (from superquadrics)
    _, _, reference_mesh = compute_global_normalization(
        original, edited, mesh_resolution
    )

    # === STEP 4: Create per-superquadric masks for changed_edited ===
    print("[PrepareEditData] Creating per-superquadric masks...")
    changed_edited_masks = []

    for i, (sq_orig, sq_edit) in enumerate(zip(changed_orig_list, changed_edit_list)):
        sq_mesh = create_mesh_from_superquadrics([sq_edit], resolution=mesh_resolution)
        if sq_mesh is None:
            continue

        # Normalize: (vertices - center) / scale
        sq_mesh.vertices -= global_center
        sq_mesh.vertices /= global_scale

        # Voxelize at both resolutions
        sq_voxels_latent = voxelize_mesh(sq_mesh, resolution=latent_resolution)
        sq_voxels_voxel = voxelize_mesh(sq_mesh, resolution=voxel_resolution)

        # Build transformation matrix for latent space
        M_latent = build_voxel_transformation(
            sq_orig, sq_edit, global_center, global_scale, latent_resolution
        )

        # Compute world-space transformation (edited→original) for SLAT injection
        M_orig = get_superquadric_matrix(sq_orig)
        M_edit = get_superquadric_matrix(sq_edit)
        M_transform_inv = M_orig @ np.linalg.inv(M_edit)  # edited → original

        sq_index = sq_edit.get("index", i)

        changed_edited_masks.append(
            {
                "mask": sq_voxels_latent,
                "mask_voxel": sq_voxels_voxel,
                "M_latent": M_latent,
                "M_transform_inv": M_transform_inv,  # world-space: edited → original
                "sq_index": sq_index,
                "transformed_mesh_path": transformed_mesh_paths.get(sq_index),
            }
        )

        # Save visualization
        if save_visualizations:
            occupied = np.sum(sq_voxels_latent > 0.5)
            print(f"  [changed_edited_sq{sq_index}] Occupied: {int(occupied)}")
            if occupied > 0:
                cube_mesh = voxels_to_cube_mesh(sq_voxels_latent)
                mesh_path = masks_folder / f"changed_edited_sq{sq_index}.ply"
                o3d.io.write_triangle_mesh(
                    str(mesh_path), cube_mesh, write_vertex_colors=True
                )
                if render_fn:
                    try:
                        render_path = masks_folder / f"changed_edited_sq{sq_index}.png"
                        render_fn(str(mesh_path), str(render_path))
                    except Exception as e:
                        print(f"    Render failed: {e}")

    # === STEP 5: Create union masks for changed_original ===
    changed_original_mask = np.zeros((latent_resolution,) * 3, dtype=np.float32)
    changed_original_mask_voxel = np.zeros((voxel_resolution,) * 3, dtype=np.float32)

    for sq in changed_orig_list:
        sq_mesh = create_mesh_from_superquadrics([sq], resolution=mesh_resolution)
        if sq_mesh is None:
            continue
        sq_mesh.vertices -= global_center
        sq_mesh.vertices /= global_scale
        sq_voxels_latent = voxelize_mesh(sq_mesh, resolution=latent_resolution)
        sq_voxels_voxel = voxelize_mesh(sq_mesh, resolution=voxel_resolution)
        changed_original_mask = np.logical_or(
            changed_original_mask, sq_voxels_latent
        ).astype(np.float32)
        changed_original_mask_voxel = np.logical_or(
            changed_original_mask_voxel, sq_voxels_voxel
        ).astype(np.float32)

    # Save changed_original visualization
    if save_visualizations:
        occupied = np.sum(changed_original_mask > 0.5)
        print(f"  [changed_original] Occupied: {int(occupied)}")
        if occupied > 0:
            cube_mesh = voxels_to_cube_mesh(changed_original_mask)
            mesh_path = masks_folder / "changed_original.ply"
            o3d.io.write_triangle_mesh(
                str(mesh_path), cube_mesh, write_vertex_colors=True
            )
            if render_fn:
                try:
                    render_path = masks_folder / "changed_original.png"
                    render_fn(str(mesh_path), str(render_path))
                except Exception as e:
                    print(f"    Render failed: {e}")

    # === STEP 6: Create union masks for added_deleted ===
    added_deleted_mask = np.zeros((latent_resolution,) * 3, dtype=np.float32)
    added_deleted_mask_voxel = np.zeros((voxel_resolution,) * 3, dtype=np.float32)

    for sq in added_deleted_list:
        sq_mesh = create_mesh_from_superquadrics([sq], resolution=mesh_resolution)
        if sq_mesh is None:
            continue
        sq_mesh.vertices -= global_center
        sq_mesh.vertices /= global_scale
        sq_voxels_latent = voxelize_mesh(sq_mesh, resolution=latent_resolution)
        sq_voxels_voxel = voxelize_mesh(sq_mesh, resolution=voxel_resolution)
        added_deleted_mask = np.logical_or(added_deleted_mask, sq_voxels_latent).astype(
            np.float32
        )
        added_deleted_mask_voxel = np.logical_or(
            added_deleted_mask_voxel, sq_voxels_voxel
        ).astype(np.float32)

    # Save added_deleted visualization
    if save_visualizations:
        occupied = np.sum(added_deleted_mask > 0.5)
        print(f"  [added_deleted] Occupied: {int(occupied)}")
        if occupied > 0:
            cube_mesh = voxels_to_cube_mesh(added_deleted_mask)
            mesh_path = masks_folder / "added_deleted.ply"
            o3d.io.write_triangle_mesh(
                str(mesh_path), cube_mesh, write_vertex_colors=True
            )
            if render_fn:
                try:
                    render_path = masks_folder / "added_deleted.png"
                    render_fn(str(mesh_path), str(render_path))
                except Exception as e:
                    print(f"    Render failed: {e}")

    # === STEP 6b: Create union masks for unchanged superquadrics ===
    unchanged_mask = np.zeros((latent_resolution,) * 3, dtype=np.float32)
    unchanged_mask_voxel = np.zeros((voxel_resolution,) * 3, dtype=np.float32)

    for sq in unchanged_list:
        sq_mesh = create_mesh_from_superquadrics([sq], resolution=mesh_resolution)
        if sq_mesh is None:
            continue
        sq_mesh.vertices -= global_center
        sq_mesh.vertices /= global_scale
        sq_voxels_latent = voxelize_mesh(sq_mesh, resolution=latent_resolution)
        sq_voxels_voxel = voxelize_mesh(sq_mesh, resolution=voxel_resolution)
        unchanged_mask = np.logical_or(unchanged_mask, sq_voxels_latent).astype(
            np.float32
        )
        unchanged_mask_voxel = np.logical_or(
            unchanged_mask_voxel, sq_voxels_voxel
        ).astype(np.float32)

    # Save unchanged visualization
    if save_visualizations:
        occupied = np.sum(unchanged_mask > 0.5)
        print(f"  [unchanged] Occupied: {int(occupied)}")
        if occupied > 0:
            cube_mesh = voxels_to_cube_mesh(unchanged_mask)
            mesh_path = masks_folder / "unchanged.ply"
            o3d.io.write_triangle_mesh(
                str(mesh_path), cube_mesh, write_vertex_colors=True
            )
            if render_fn:
                try:
                    render_path = masks_folder / "unchanged.png"
                    render_fn(str(mesh_path), str(render_path))
                except Exception as e:
                    print(f"    Render failed: {e}")

    # === STEP 7: Run verification (optional) ===
    if run_verification and changed_edited_masks:
        _run_latent_verification(
            changed_orig_list,
            changed_edit_list,
            changed_edited_masks,
            global_center,
            global_scale,
            latent_resolution,
            mesh_resolution,
            masks_folder,
            render_fn,
        )

    # === STEP 8: Transform masks to Trellis orientation ===
    print("[PrepareEditData] Transforming masks to Trellis orientation...")
    for mask_data in changed_edited_masks:
        mask_data["mask"] = transform_voxels_to_trellis(mask_data["mask"])
        mask_data["mask_voxel"] = transform_voxels_to_trellis(mask_data["mask_voxel"])

    changed_original_mask = transform_voxels_to_trellis(changed_original_mask)
    changed_original_mask_voxel = transform_voxels_to_trellis(
        changed_original_mask_voxel
    )
    added_deleted_mask = transform_voxels_to_trellis(added_deleted_mask)
    added_deleted_mask_voxel = transform_voxels_to_trellis(added_deleted_mask_voxel)
    unchanged_mask = transform_voxels_to_trellis(unchanged_mask)
    unchanged_mask_voxel = transform_voxels_to_trellis(unchanged_mask_voxel)

    print(f"[PrepareEditData] Complete. Masks saved to: {masks_folder}")

    return {
        "changed_edited_masks": changed_edited_masks,
        "changed_original_mask": changed_original_mask,
        "changed_original_mask_voxel": changed_original_mask_voxel,
        "added_deleted_mask": added_deleted_mask,
        "added_deleted_mask_voxel": added_deleted_mask_voxel,
        "unchanged_mask": unchanged_mask,
        "unchanged_mask_voxel": unchanged_mask_voxel,
        "num_changed": len(changed_edited_masks),
        "num_added_deleted": len(added_deleted_list),
        "num_unchanged": len(unchanged_list),
        "latent_resolution": latent_resolution,
        "voxel_resolution": voxel_resolution,
        "global_center": global_center,
        "global_scale": global_scale,
        "categories": categories,
        "transformed_mesh_path": combined_transformed_mesh_path,
    }


def _run_latent_verification(
    changed_orig_list: List[dict],
    changed_edit_list: List[dict],
    changed_edited_masks: List[Dict],
    global_center: np.ndarray,
    global_scale: float,
    latent_resolution: int,
    mesh_resolution: int,
    masks_folder: Path,
    render_fn,
):
    """Run transformation and interpolation verification at latent resolution."""
    print("\n" + "=" * 60)
    print("LATENT VERIFICATION: TRANSFORMATION AND INTERPOLATION CHECKS")
    print(f"Resolution: {latent_resolution}x{latent_resolution}x{latent_resolution}")
    print("=" * 60)

    # Create color grid at latent resolution
    color_grid = create_color_grid(latent_resolution)

    # Create union of all edited masks
    edit_voxels_union = np.zeros((latent_resolution,) * 3, dtype=np.float32)
    for sq_data in changed_edited_masks:
        edit_voxels_union = np.logical_or(edit_voxels_union, sq_data["mask"]).astype(
            np.float32
        )

    occupied = np.argwhere(edit_voxels_union > 0.5)
    print(f"  Found {len(occupied)} occupied voxels in edited masks union")

    if len(occupied) == 0:
        print("  No occupied voxels, skipping verification")
        return

    # For each voxel, find which superquadric it belongs to
    voxel_to_sq = []
    for idx in occupied:
        best_sq_idx = 0
        for sq_idx, sq_data in enumerate(changed_edited_masks):
            if sq_data["mask"][idx[0], idx[1], idx[2]] > 0.5:
                best_sq_idx = sq_idx
                break
        voxel_to_sq.append(best_sq_idx)

    # 1. Transformation check (nearest neighbor)
    print("\n  Creating transformed visualization (nearest neighbor)...")
    transformed_voxels = np.zeros((latent_resolution,) * 3, dtype=np.float32)

    for i, idx in enumerate(occupied):
        sq_data = changed_edited_masks[voxel_to_sq[i]]
        M_latent = sq_data["M_latent"]

        coord_homogeneous = np.array([idx[0], idx[1], idx[2], 1.0])
        orig_coord = M_latent @ coord_homogeneous

        i_orig, j_orig, k_orig = [int(round(c)) for c in orig_coord[:3]]
        if (
            0 <= i_orig < latent_resolution
            and 0 <= j_orig < latent_resolution
            and 0 <= k_orig < latent_resolution
        ):
            transformed_voxels[i_orig, j_orig, k_orig] = 1.0

    transformed_mesh = voxels_to_cube_mesh(transformed_voxels)
    transformed_path = masks_folder / "verification_transformed.ply"
    o3d.io.write_triangle_mesh(
        str(transformed_path), transformed_mesh, write_vertex_colors=True
    )
    print(f"  Saved: {transformed_path}")
    print(f"  Transformed voxels count: {int(np.sum(transformed_voxels > 0.5))}")

    if render_fn:
        try:
            render_path = masks_folder / "verification_transformed.png"
            render_fn(str(transformed_path), str(render_path))
            print(f"  Rendered: {render_path}")
        except Exception as e:
            print(f"  Render failed: {e}")

    # 2. Interpolation check (trilinear)
    print("\n  Creating interpolated visualization (trilinear colors)...")
    interpolated_colors = []
    debug_data = []

    for i, idx in enumerate(occupied):
        sq_data = changed_edited_masks[voxel_to_sq[i]]
        M_latent = sq_data["M_latent"]

        coord_homogeneous = np.array([idx[0], idx[1], idx[2], 1.0])
        transformed_coord = M_latent @ coord_homogeneous
        transformed_pos = transformed_coord[:3]

        color = interpolate_grid_values(
            idx.reshape(1, 3).astype(np.float32), color_grid, M_latent, order=1
        )[0]

        in_bounds = all(0 <= c < latent_resolution for c in transformed_pos)

        debug_data.append(
            {
                "voxel_index": i,
                "sq_index": voxel_to_sq[i],
                "original_position": idx.tolist(),
                "transformed_position": transformed_pos.tolist(),
                "in_bounds": in_bounds,
                "interpolated_color": color.tolist(),
                "color_magnitude": float(np.linalg.norm(color)),
            }
        )

        interpolated_colors.append(np.clip(color, 0, 1))

    # Save debug JSON
    debug_json_path = masks_folder / "interpolation_debug.json"
    with open(debug_json_path, "w") as f:
        json.dump(debug_data, f, indent=2)
    print(f"  Saved debug data: {debug_json_path}")

    out_of_bounds = [d for d in debug_data if not d["in_bounds"]]
    black_voxels = [d for d in debug_data if d["color_magnitude"] < 0.01]
    print(f"  Out of bounds voxels: {len(out_of_bounds)} / {len(debug_data)}")
    print(f"  Black/near-black voxels: {len(black_voxels)} / {len(debug_data)}")

    if black_voxels:
        print("  First 5 black voxels:")
        for d in black_voxels[:5]:
            print(
                f"    Voxel {d['voxel_index']}: pos={d['original_position']} -> {d['transformed_position']}, in_bounds={d['in_bounds']}"
            )

    interp_mesh = voxels_to_cube_mesh(
        edit_voxels_union, colors=np.array(interpolated_colors)
    )
    interp_path = masks_folder / "verification_interpolated.ply"
    o3d.io.write_triangle_mesh(str(interp_path), interp_mesh, write_vertex_colors=True)
    print(f"  Saved: {interp_path}")

    if render_fn:
        try:
            render_path = masks_folder / "verification_interpolated.png"
            render_fn(str(interp_path), str(render_path))
            print(f"  Rendered: {render_path}")
        except Exception as e:
            print(f"  Render failed: {e}")

    # 3. Save reference color grid
    print("\n  Saving reference color grid...")
    orig_voxels = np.zeros((latent_resolution,) * 3, dtype=np.float32)
    for sq in changed_orig_list:
        sq_mesh = create_mesh_from_superquadrics([sq], resolution=mesh_resolution)
        if sq_mesh is None:
            continue
        sq_mesh.vertices -= global_center
        sq_mesh.vertices /= global_scale
        sq_voxels = voxelize_mesh(sq_mesh, resolution=latent_resolution)
        orig_voxels = np.logical_or(orig_voxels, sq_voxels).astype(np.float32)

    orig_occupied = np.argwhere(orig_voxels > 0.5)
    if len(orig_occupied) > 0:
        orig_colors = []
        for idx in orig_occupied:
            color = color_grid[idx[0], idx[1], idx[2]]
            orig_colors.append(np.clip(color, 0, 1))

        orig_color_mesh = voxels_to_cube_mesh(orig_voxels, colors=np.array(orig_colors))
        orig_color_path = masks_folder / "verification_original_colors.ply"
        o3d.io.write_triangle_mesh(
            str(orig_color_path), orig_color_mesh, write_vertex_colors=True
        )
        print(f"  Saved: {orig_color_path}")

        if render_fn:
            try:
                render_path = masks_folder / "verification_original_colors.png"
                render_fn(str(orig_color_path), str(render_path))
                print(f"  Rendered: {render_path}")
            except Exception as e:
                print(f"  Render failed: {e}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


# === BACKWARD COMPATIBILITY ALIASES ===
# These allow old code to still work during transition


def generate_voxel_masks(
    original_json: str,
    edited_json: str,
    output_folder: str,
    render_fn=None,
    resolution: int = 64,
    mesh_resolution: int = 30,
    render_masks: bool = True,
    run_verification: bool = False,
) -> Dict[str, str]:
    """Deprecated: Use prepare_edit_data instead."""
    print("Warning: generate_voxel_masks is deprecated, use prepare_edit_data instead")
    result = prepare_edit_data(
        original_json=original_json,
        edited_json=edited_json,
        output_folder=output_folder,
        voxel_resolution=resolution,
        mesh_resolution=mesh_resolution,
        render_fn=render_fn,
        save_visualizations=render_masks,
        run_verification=run_verification,
    )
    return {} if result is None else {}


def get_latent_merging_data(
    original_json: str,
    edited_json: str,
    latent_resolution: int = 16,
    voxel_resolution: int = 64,
    mesh_resolution: int = 30,
    output_folder: str = None,
    render_fn=None,
    run_verification: bool = False,
    no_interp_mode: bool = False,
    original_mesh_path: str = None,
) -> Optional[Dict]:
    """Deprecated: Use prepare_edit_data instead."""
    print(
        "Warning: get_latent_merging_data is deprecated, use prepare_edit_data instead"
    )
    return prepare_edit_data(
        original_json=original_json,
        edited_json=edited_json,
        output_folder=output_folder or ".",
        latent_resolution=latent_resolution,
        voxel_resolution=voxel_resolution,
        mesh_resolution=mesh_resolution,
        render_fn=render_fn,
        save_visualizations=output_folder is not None,
        run_verification=run_verification,
        create_transformed_meshes=no_interp_mode,
        original_mesh_path=original_mesh_path,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare edit data from abstraction comparisons"
    )
    parser.add_argument(
        "--original", type=str, required=True, help="Path to original abstraction JSON"
    )
    parser.add_argument(
        "--edited", type=str, required=True, help="Path to edited abstraction JSON"
    )
    parser.add_argument("--output", type=str, required=True, help="Output folder")
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help="Original mesh path (for transformed meshes)",
    )
    parser.add_argument(
        "--resolution", type=int, default=64, help="Voxel grid resolution"
    )
    parser.add_argument("--render", action="store_true", help="Render visualizations")
    parser.add_argument("--verify", action="store_true", help="Run verification checks")
    parser.add_argument(
        "--transform-meshes", action="store_true", help="Create transformed meshes"
    )

    args = parser.parse_args()

    render_fn = None
    if args.render:
        from prox_e.utils import render_obj_with_blender

        render_fn = render_obj_with_blender

    result = prepare_edit_data(
        original_json=args.original,
        edited_json=args.edited,
        output_folder=args.output,
        voxel_resolution=args.resolution,
        render_fn=render_fn,
        run_verification=args.verify,
        create_transformed_meshes=args.transform_meshes,
        original_mesh_path=args.mesh,
    )

    if result:
        print("\nResult summary:")
        print(f"  Changed superquadrics: {result['num_changed']}")
        print(f"  Added/deleted superquadrics: {result['num_added_deleted']}")
