"""Mesh orientation helpers for custom input meshes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


ORIENTATION_TRANSFORMS: dict[int, tuple[str, np.ndarray]] = {
    0: ("none", np.eye(4)),
    1: ("flip_z", np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])),
    2: ("flip_y", np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
    3: ("flip_x", np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])),
    4: (
        "flip_y_and_z",
        np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
    ),
    5: ("swap_y_z", np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])),
    6: (
        "swap_y_z_flip_z",
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
    ),
    7: (
        "swap_y_z_flip_y",
        np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
    ),
    8: (
        "swap_y_z_flip_both",
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
    ),
    9: (
        "rot_180_x",
        np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
    ),
    10: (
        "rot_180_y",
        np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
    ),
    11: (
        "rot_180_z",
        np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    ),
    12: (
        "rot_90_x",
        np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
    ),
    13: (
        "rot_-90_x",
        np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]),
    ),
    14: (
        "rot_90_y",
        np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]),
    ),
    15: (
        "rot_-90_y",
        np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
    ),
}

TARGET_MAX_DIM = 0.75


def get_orientation_transform(index: int) -> tuple[str, np.ndarray]:
    if index not in ORIENTATION_TRANSFORMS:
        valid = ", ".join(str(i) for i in sorted(ORIENTATION_TRANSFORMS))
        raise ValueError(f"Unknown orientation index {index}. Valid indices: {valid}")
    name, matrix = ORIENTATION_TRANSFORMS[index]
    return name, matrix.copy()


def load_mesh_union(path: Path | str) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), force="mesh")
    if isinstance(mesh, trimesh.Scene):
        meshes = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in {path}")
        mesh = trimesh.util.concatenate(meshes)
    if mesh.is_empty or len(mesh.vertices) == 0:
        raise ValueError(f"Empty mesh: {path}")
    return mesh


def apply_legacy_input_orientation(mesh: trimesh.Trimesh, *, edit3dbench: bool) -> str:
    """Preserve the existing direct-mesh behavior when no orientation index is supplied."""
    if not edit3dbench:
        rot_x = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
        mesh.apply_transform(rot_x)
        label = "legacy_rot_x_-90_then_rot_y_180"
    else:
        label = "legacy_edit3dbench_rot_y_180"
    rot_y = trimesh.transformations.rotation_matrix(np.radians(180), [0, 1, 0])
    mesh.apply_transform(rot_y)
    return label


def apply_orientation(
    mesh: trimesh.Trimesh, orientation_index: int | None, *, edit3dbench: bool
) -> str:
    if orientation_index is None:
        return apply_legacy_input_orientation(mesh, edit3dbench=edit3dbench)
    name, matrix = get_orientation_transform(orientation_index)
    mesh.apply_transform(matrix)
    return f"{orientation_index}:{name}"


def normalize_mesh(
    mesh: trimesh.Trimesh,
    *,
    target_max_dim: float = TARGET_MAX_DIM,
) -> dict[str, Any]:
    vertices = mesh.vertices
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    center = (bbox_min + bbox_max) / 2
    bbox_size = float((bbox_max - bbox_min).max())
    if bbox_size <= 0:
        raise ValueError("Cannot normalize a mesh with a zero-size bounding box")
    scale_factor = float(target_max_dim / bbox_size)
    mesh.vertices = (vertices - center) * scale_factor
    return {
        "bbox_min": bbox_min.tolist(),
        "bbox_max": bbox_max.tolist(),
        "center": center.tolist(),
        "max_dim": bbox_size,
        "target_max_dim": target_max_dim,
        "scale_factor": scale_factor,
    }


def expected_orientation_metadata(
    source_path: Path | str,
    orientation_index: int | None,
    *,
    edit3dbench: bool,
) -> dict[str, Any]:
    return {
        "source_path": str(Path(source_path)),
        "orientation_index": orientation_index,
        "edit3dbench": bool(edit3dbench),
        "target_max_dim": TARGET_MAX_DIM,
    }


def metadata_matches(metadata_path: Path, expected: dict[str, Any]) -> bool:
    try:
        saved = json.loads(metadata_path.read_text())
    except Exception:
        return False
    return all(saved.get(k) == v for k, v in expected.items())


def load_oriented_normalized_mesh(
    source_path: Path | str,
    output_path: Path | str,
    *,
    orientation_index: int | None = None,
    edit3dbench: bool = False,
    metadata_path: Path | str | None = None,
) -> dict[str, Any]:
    source_path = Path(source_path)
    output_path = Path(output_path)
    metadata_path = (
        Path(metadata_path)
        if metadata_path is not None
        else output_path.with_suffix(".orientation.json")
    )

    mesh = load_mesh_union(source_path)
    orientation_label = apply_orientation(
        mesh, orientation_index, edit3dbench=edit3dbench
    )
    stats = normalize_mesh(mesh)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))

    metadata = expected_orientation_metadata(
        source_path, orientation_index, edit3dbench=edit3dbench
    )
    metadata.update({"orientation_label": orientation_label, "normalization": stats})
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
