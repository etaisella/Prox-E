"""
Superquadric abstraction utilities using SuperDec.

Main entry point: generate_abstraction()
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import trimesh
from omegaconf import OmegaConf
import open3d as o3d

# Add submodules to path
REPO_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(REPO_ROOT / "submodules" / "superdec"))

from superdec.superdec import SuperDec
from superdec.data.dataloader import (
    denormalize_outdict,
    denormalize_points,
    normalize_points,
)
from superdec.utils.predictions_handler import PredictionHandler

from prox_e.utils import render_obj_with_blender


# Module-level model cache
_superdec_model = None
_superdec_device = None


def _to_serializable(val):
    """Convert value to JSON-serializable format."""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().tolist()
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, (float, int, str, bool, list, dict, type(None))):
        return val
    else:
        return str(val)


def _round_floats(obj, decimals: int = 3):
    """Recursively round all float values in a nested structure to specified decimal places."""
    if isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, dict):
        return {k: _round_floats(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_round_floats(item, decimals) for item in obj]
    else:
        return obj


def _get_superdec_model(device: str = None):
    """Get or load the SuperDec model (cached)."""
    global _superdec_model, _superdec_device

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _superdec_model is not None and _superdec_device == device:
        return _superdec_model, device

    checkpoints_dir = (
        REPO_ROOT / "submodules" / "superdec" / "checkpoints" / "normalized"
    )
    ckp_path = checkpoints_dir / "ckpt.pt"
    config_path = checkpoints_dir / "config.yaml"

    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    with open(config_path) as f:
        configs = OmegaConf.load(f)

    model = SuperDec(configs.superdec).to(device)
    model.lm_optimization = False
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _superdec_model = model
    _superdec_device = device

    print(f"SuperDec model loaded on {device}")
    return model, device


def generate_abstraction(
    mesh_file: Union[str, Path],
    output_folder: Union[str, Path],
    category: str = "object",
    resolution: int = 30,
    save_superdec_json: bool = True,
) -> Dict:
    """
    Generate superquadric abstraction from a mesh file using SuperDec.

    This is the main entry point for abstraction generation. It:
    1. Loads and samples the mesh to a point cloud
    2. Runs SuperDec inference
    3. Extracts the superquadric mesh
    4. Saves all outputs (OBJ, abstraction JSON, optionally raw SuperDec JSON)

    Args:
        mesh_file: Path to input mesh (OBJ file)
        output_folder: Folder to save outputs
        category: Category name for the object (default: "object")
        resolution: Mesh extraction resolution (default: 30)
        save_superdec_json: Whether to save raw SuperDec output JSON (default: True)

    Returns:
        Dict with keys:
            - 'mesh': trimesh.Trimesh of the superquadric abstraction
            - 'abstraction': List of superquadric dicts
            - 'obj_path': Path to saved OBJ file
            - 'abstraction_json_path': Path to saved abstraction JSON
            - 'superdec_json_path': Path to raw SuperDec JSON (if saved)
    """
    mesh_file = Path(mesh_file)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load SuperDec model
    model, device = _get_superdec_model()

    # Load point cloud from mesh
    pc = o3d.io.read_triangle_mesh(str(mesh_file))
    pc_sampled = pc.sample_points_uniformly(number_of_points=4096)
    points_tmp = np.asarray(pc_sampled.points)
    n_points = points_tmp.shape[0]

    # Subsample to 4096 points (SuperDec input size)
    if n_points != 4096:
        replace = n_points < 4096
        idxs = np.random.choice(n_points, 4096, replace=replace)
        points = points_tmp[idxs]
    else:
        points = points_tmp

    # Normalize points
    points, translation, scale = normalize_points(points)

    # Convert to tensor
    points_tensor = torch.from_numpy(points).unsqueeze(0).to(device).float()

    # Run SuperDec inference
    print("Running SuperDec inference...")
    with torch.no_grad():
        outdict = model(points_tensor)
        print("SuperDec inference completed")

        # Save raw SuperDec output if requested
        superdec_json_path = None
        if save_superdec_json:
            outdict_json = {k: _to_serializable(v) for k, v in outdict.items()}
            superdec_json_path = output_folder / "superdec.json"
            with open(superdec_json_path, "w") as f:
                json.dump(outdict_json, f, indent=2)
            print(f"Saved SuperDec output to {superdec_json_path}")

        # Move tensors to CPU and denormalize
        for key in outdict:
            if isinstance(outdict[key], torch.Tensor):
                outdict[key] = outdict[key].cpu()
        translation_arr = np.array([translation])
        scale_arr = np.array([scale])
        outdict = denormalize_outdict(outdict, translation_arr, scale_arr, z_up=False)
        points_tensor = denormalize_points(
            points_tensor.cpu(), translation_arr, scale_arr, z_up=False
        )

    # Extract mesh from SuperDec output
    print("Extracting superquadric mesh...")
    pred_handler = _AbstractionHandler.from_outdict(outdict, points_tensor, [category])
    superdec_mesh = pred_handler.get_meshes(resolution=resolution)[0]

    # Export mesh to OBJ file
    obj_path = output_folder / "superdec.obj"
    superdec_mesh.export(str(obj_path))
    print(f"SuperDec mesh exported to: {obj_path}")

    # Save simplified abstraction JSON (after get_meshes so colors are assigned)
    abstraction_json_path = output_folder / "abstraction.json"
    pred_handler.save_abstraction_json(0, str(abstraction_json_path))
    print(f"Saved abstraction to: {abstraction_json_path}")

    # Get abstraction data
    abstraction = pred_handler.get_abstraction(0)

    return {
        "mesh": superdec_mesh,
        "abstraction": abstraction,
        "obj_path": obj_path,
        "abstraction_json_path": abstraction_json_path,
        "superdec_json_path": superdec_json_path,
    }


def load_abstraction(filepath: Union[str, Path]) -> List[Dict]:
    """
    Load an abstraction from a JSON file.

    Args:
        filepath: Path to the abstraction JSON file

    Returns:
        List of superquadric dicts
    """
    with open(filepath, "r") as f:
        return json.load(f)


def mesh_from_abstraction(
    abstraction: List[Dict], resolution: int = 30
) -> trimesh.Trimesh:
    """
    Create a mesh from abstraction data.

    Args:
        abstraction: List of superquadric dicts (from load_abstraction or generate_abstraction)
        resolution: Mesh resolution for each superquadric

    Returns:
        trimesh.Trimesh with vertex colors
    """
    return _AbstractionHandler.mesh_from_abstraction(abstraction, resolution)


def render_multiview(
    obj_path: Union[str, Path],
    output_folder: Union[str, Path],
    prefix: str = "",
    single_view: bool = False,
    elev: float = 10.0,
    **render_kwargs,
) -> List[Path]:
    """
    Render an OBJ file from multiple views (or single view).

    Args:
        obj_path: Path to the OBJ file to render
        output_folder: Folder to save rendered images
        prefix: Optional prefix for output filenames (e.g. "superdec_")
        single_view: If True, render only one view; if False, render 4 cardinal views
        elev: Elevation angle for multi-view renders (default: 10.0)
        **render_kwargs: Additional kwargs passed to render_obj_with_blender

    Returns:
        List of paths to rendered images
    """
    obj_path = Path(obj_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    rendered_paths = []

    if single_view:
        render_path = (
            output_folder / f"{prefix}render.png"
            if prefix
            else output_folder / "render.png"
        )
        render_obj_with_blender(
            obj_path=str(obj_path), output_path=str(render_path), **render_kwargs
        )
        rendered_paths.append(render_path)
    else:
        azimuths = [0, 90, 180, 270]
        direction_names = ["left", "front", "right", "back"]
        for direction_name, azimuth in zip(direction_names, azimuths):
            render_path = output_folder / f"{prefix}{direction_name}.png"
            render_obj_with_blender(
                obj_path=str(obj_path),
                output_path=str(render_path),
                azim=azimuth,
                elev=elev,
                **render_kwargs,
            )
            rendered_paths.append(render_path)

    return rendered_paths


# ============================================================================
# Internal classes (not part of public API)
# ============================================================================


class _AbstractionHandler(PredictionHandler):
    """Extended PredictionHandler with abstraction JSON export/import."""

    def get_abstraction(self, index: int) -> List[Dict]:
        """Get simplified abstraction representation for a single sample."""
        P = self.scale.shape[1]
        abstraction = []

        sq_index = 0
        for p in range(P):
            # print(f"exist: {self.exist[index, p]}")
            if self.exist[index, p] > 0.5:  # change to 0.5 later
                superquad = {
                    "index": sq_index,
                    "scale": self.scale[index, p].tolist(),
                    "translation": self.translation[index, p].tolist(),
                    "rotation": self.rotation[index, p].tolist(),
                    "exponents": self.exponents[index, p].tolist(),
                    "color": self.colors[p].tolist(),
                }
                abstraction.append(superquad)
                sq_index += 1

        return abstraction

    def save_abstraction_json(self, index: int, filepath: str) -> None:
        """Save the simplified abstraction for a single sample to a JSON file."""
        abstraction = self.get_abstraction(index)
        # Round all float values to 3 decimal places
        abstraction = _round_floats(abstraction, decimals=3)
        with open(filepath, "w") as f:
            json.dump(abstraction, f, indent=2)

    @staticmethod
    def mesh_from_abstraction(
        abstraction: List[Dict], resolution: int = 30
    ) -> trimesh.Trimesh:
        """Create a mesh from an abstraction JSON list."""
        vertices = []
        faces = []
        v_colors = []
        f_colors = []
        vertex_offset = 0

        for sq in abstraction:
            scale = np.array(sq["scale"])
            exponents = np.array(sq["exponents"])
            rotation = np.array(sq["rotation"])
            translation = np.array(sq["translation"])
            color = np.array(sq["color"])

            verts, tris = _AbstractionHandler._superquadric_mesh_static(
                scale, exponents, rotation, translation, resolution
            )

            vertices.append(verts)
            faces.append(tris + vertex_offset)
            v_colors.append(np.ones((verts.shape[0], 3)) * color)
            f_colors.append(np.ones((tris.shape[0], 3)) * color)

            vertex_offset += len(verts)

        vertices = np.concatenate(vertices)
        faces = np.concatenate(faces)
        v_colors = np.concatenate(v_colors) / 255.0
        f_colors = np.concatenate(f_colors) / 255.0

        return trimesh.Trimesh(
            vertices, faces, face_colors=f_colors, vertex_colors=v_colors
        )

    @staticmethod
    def _superquadric_mesh_static(scale, exponents, rotation, translation, N):
        """Generate a single superquadric mesh."""

        def f(o, m):
            return np.sign(np.sin(o)) * np.abs(np.sin(o)) ** m

        def g(o, m):
            return np.sign(np.cos(o)) * np.abs(np.cos(o)) ** m

        u = np.linspace(-np.pi, np.pi, N, endpoint=True)
        v = np.linspace(-np.pi / 2.0, np.pi / 2.0, N, endpoint=True)
        u = np.tile(u, N)
        v = np.repeat(v, N)

        if np.linalg.det(rotation) < 0:
            u = u[::-1]

        x = scale[0] * g(v, exponents[0]) * g(u, exponents[1])
        y = scale[1] * g(v, exponents[0]) * f(u, exponents[1])
        z = scale[2] * f(v, exponents[0])

        x[:N] = 0.0
        x[-N:] = 0.0

        vertices = np.concatenate(
            [np.expand_dims(x, 1), np.expand_dims(y, 1), np.expand_dims(z, 1)], axis=1
        )
        vertices = (rotation @ vertices.T).T + translation

        triangles = []
        for i in range(N - 1):
            for j in range(N - 1):
                triangles.append([i * N + j, i * N + j + 1, (i + 1) * N + j])
                triangles.append(
                    [(i + 1) * N + j, i * N + j + 1, (i + 1) * N + (j + 1)]
                )
        for i in range(N - 1):
            triangles.append([i * N + (N - 1), i * N, (i + 1) * N + (N - 1)])
            triangles.append([(i + 1) * N + (N - 1), i * N, (i + 1) * N])

        triangles.append([(N - 1) * N + (N - 1), (N - 1) * N, (N - 1)])
        triangles.append([(N - 1), (N - 1) * N, 0])

        return np.array(vertices), np.array(triangles)
