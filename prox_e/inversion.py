#!/usr/bin/env python3
"""
VoxHammer-based inversion for sparse structure in TRELLIS.

This module uses VoxHammer's sample_sparse_structure_inverse function directly
for more accurate inversion than custom implementations.
"""

import os
import sys
import time
import torch
import numpy as np
import open3d as o3d
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from types import MethodType
from prox_e.appearance_editing import (
    render_for_conditioning,
    preprocess_single_image,
    load_trellis_image_pipeline,
)
from prox_e.utils import transform_voxels_to_trellis

# Add VoxHammer submodule to path
VOXHAMMER_PATH = Path(__file__).parent / "submodules" / "voxhammer"
sys.path.insert(0, str(VOXHAMMER_PATH))

# Add supergen submodule to path (for trellis)
SUPERGEN_PATH = Path(__file__).parent / "submodules" / "supergen"
sys.path.insert(0, str(SUPERGEN_PATH))

# Import VoxHammer components
from voxhammer.edit_pipeline import (
    InversionFlowEulerGuidanceIntervalSampler,
    ss_flow_forward,
    ss_trsfmr_forward,
    ss_attn_forward,
    feats_to_slat,
    slat_flow_forward,
    slat_trsfmr_forward,
    slat_attn_forward,
)

# Use local rendering module with flat shading fix (instead of voxhammer.bpy_render)
from prox_e.rendering import render_3d_model
import trellis.modules.sparse as sp
from trellis.utils import postprocessing_utils

# Import trellis pipeline
from trellis.pipelines import TrellisTextTo3DPipeline
from tqdm import tqdm
import json
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import utils3d


def extract_features_single_thread(
    output_dir, model="dinov2_vitl14_reg", batch_size=10
):
    """
    Single-threaded version of VoxHammer's extract_features.
    Avoids ThreadPoolExecutor which can cause issues with multi-GPU setups.
    """
    # Clear CUDA cache before loading DINOv2 to free memory
    torch.cuda.empty_cache()
    import gc

    gc.collect()

    # Load DINOv2 model from local cache to avoid network issues
    dinov2_repo_path = os.path.expanduser(
        "~/.cache/torch/hub/facebookresearch_dinov2_main"
    )
    if os.path.exists(dinov2_repo_path):
        dinov2_model = torch.hub.load(
            dinov2_repo_path, model, source="local", pretrained=True
        )
    else:
        dinov2_model = torch.hub.load("facebookresearch/dinov2", model, pretrained=True)
    dinov2_model.eval().cuda()

    transform_fn = transforms.Compose(
        [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    n_patch = 518 // 14

    transforms_path = os.path.join(output_dir, "transforms.json")
    mesh_path = os.path.join(output_dir, "mesh.ply")
    voxels_path = os.path.join(output_dir, "voxels.ply")

    if not os.path.exists(transforms_path):
        raise ValueError(f"Transforms file not found: {transforms_path}")
    if not os.path.exists(mesh_path):
        raise ValueError(f"Mesh file not found: {mesh_path}")

    # Voxelize mesh if needed
    if not os.path.exists(voxels_path):
        print("Voxelizing mesh...")
        mesh = o3d.io.read_triangle_mesh(mesh_path)
        vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
            mesh,
            voxel_size=1 / 64,
            min_bound=(-0.5, -0.5, -0.5),
            max_bound=(0.5, 0.5, 0.5),
        )
        voxel_vertices = np.array(
            [voxel.grid_index for voxel in voxel_grid.get_voxels()]
        )
        assert np.all(voxel_vertices >= 0) and np.all(voxel_vertices < 64), (
            "Some vertices are out of bounds"
        )
        voxel_vertices = (voxel_vertices + 0.5) / 64 - 0.5
        utils3d.io.write_ply(voxels_path, voxel_vertices)
        print(f"Voxelized mesh saved to: {voxels_path}")

    # Load transforms
    with open(transforms_path, "r") as f:
        metadata = json.load(f)
    frames = metadata["frames"]

    # Load all images sequentially (no threading). Retry with center crop if all blank.
    for center_crop_ratio in [None, 0.5]:
        data = []
        skipped_blank = 0
        for view in tqdm(frames, desc="Loading images"):
            image_path = os.path.join(output_dir, view["file_path"])
            try:
                image = Image.open(image_path)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue
            if center_crop_ratio is not None:
                w, h = image.size
                cw, ch = int(w * center_crop_ratio), int(h * center_crop_ratio)
                left, top = (w - cw) // 2, (h - ch) // 2
                image = image.crop((left, top, left + cw, top + ch))
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255

            # Skip blank images (fully transparent or very low alpha coverage)
            alpha = (
                image[:, :, 3] if image.shape[2] == 4 else np.ones_like(image[:, :, 0])
            )
            alpha_coverage = np.mean(alpha)
            if alpha_coverage < 0.01:  # Less than 1% visible pixels
                skipped_blank += 1
                continue

            image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            image = transform_fn(image)

            c2w = torch.tensor(view["transform_matrix"])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = view["camera_angle_x"]
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(
                torch.tensor(fov), torch.tensor(fov)
            )

            data.append(
                {"image": image, "extrinsics": extrinsics, "intrinsics": intrinsics}
            )

        if skipped_blank > 0:
            print(f"Skipped {skipped_blank} blank images (< 1% alpha coverage)")
        if len(data) > 0:
            if center_crop_ratio is not None:
                print(
                    "WARNING: Used center-crop fallback (50% crop) for blank renders."
                )
            break
        if center_crop_ratio == 0.5:
            raise ValueError(
                f"All {len(frames)} rendered images are blank! "
                f"This usually indicates the mesh is not properly centered in the render. "
                f"Check the normalization parameters (center/scale) passed to rendering."
            )

    if len(data) < 10:
        print(
            f"WARNING: Only {len(data)} valid images out of {len(frames)} - feature quality may be poor"
        )

    # Load positions
    positions = utils3d.io.read_ply(voxels_path)[0]
    positions = torch.from_numpy(positions).float().cuda()
    indices = ((positions + 0.5) * 64).long()
    assert torch.all(indices >= 0) and torch.all(indices < 64), (
        "Some vertices are out of bounds"
    )

    n_views = len(data)
    pack = {"indices": indices.cpu().numpy().astype(np.uint8)}

    patchtokens_lst = []
    uv_lst = []

    # Use no_grad to prevent gradient accumulation (critical for memory!)
    with torch.no_grad():
        for i in tqdm(range(0, n_views, batch_size), desc="Processing image batches"):
            batch_data = data[i : i + batch_size]
            bs = len(batch_data)
            batch_images = torch.stack([d["image"] for d in batch_data]).cuda()
            batch_extrinsics = torch.stack([d["extrinsics"] for d in batch_data]).cuda()
            batch_intrinsics = torch.stack([d["intrinsics"] for d in batch_data]).cuda()

            features = dinov2_model(batch_images, is_training=True)
            uv = (
                utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[
                    0
                ]
                * 2
                - 1
            )
            patchtokens = (
                features["x_prenorm"][:, dinov2_model.num_register_tokens + 1 :]
                .permute(0, 2, 1)
                .reshape(bs, 1024, n_patch, n_patch)
            )

            # Detach and move to CPU immediately to free GPU memory
            patchtokens_lst.append(patchtokens.detach().cpu())
            uv_lst.append(uv.detach().cpu())

            # Clear GPU memory between batches
            del (
                features,
                batch_images,
                batch_extrinsics,
                batch_intrinsics,
                patchtokens,
                uv,
            )
            torch.cuda.empty_cache()

    # Delete DINOv2 model to free GPU memory before final processing
    del dinov2_model
    torch.cuda.empty_cache()

    # Concatenate on CPU
    patchtokens = torch.cat(patchtokens_lst, dim=0)
    uv = torch.cat(uv_lst, dim=0)

    # Move to GPU for grid_sample, then back to CPU
    patchtokens = patchtokens.cuda()
    uv = uv.cuda()

    # Save features
    pack["patchtokens"] = (
        F.grid_sample(
            patchtokens, uv.unsqueeze(1), mode="bilinear", align_corners=False
        )
        .squeeze(2)
        .permute(0, 2, 1)
        .cpu()
        .numpy()
    )
    pack["patchtokens"] = np.mean(pack["patchtokens"], axis=0).astype(np.float16)

    # Cleanup
    del patchtokens, uv, patchtokens_lst, uv_lst
    torch.cuda.empty_cache()

    save_path = os.path.join(output_dir, "features.npz")
    # Use temp file to avoid S3/network filesystem seek issues with zipfile
    # Then copy bytes directly (shutil.move fails on cross-filesystem to S3)
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        np.savez_compressed(tmp_path, **pack)
        # Read and write bytes directly (S3 filesystems can handle simple writes)
        with open(tmp_path, "rb") as src:
            data = src.read()
        with open(save_path, "wb") as dst:
            dst.write(data)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    print(f"Features saved to: {save_path}")


class ConfigurableStepsSampler(InversionFlowEulerGuidanceIntervalSampler):
    """
    Sampler with configurable number of steps (VoxHammer hardcodes 25).
    """

    def __init__(self, sigma_min, steps: int = 12):
        super().__init__(sigma_min)
        self.steps = steps

    @torch.no_grad()
    def sample(
        self,
        model,
        stage,
        noise,
        cond,
        cfg_strength,
        latent=None,
        latent_mask=None,
        kv=None,
        self_kv_mask=None,
        cross_kv_mask=None,
        skip_step=None,
        noise_init=None,
        is_text=False,
        skip_latent_replacement=True,
        full_mask=True,
    ):
        """Sample with configurable steps instead of hardcoded 25."""
        steps = self.steps  # Use configurable steps
        rescale_t = 3.0
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        if latent_mask is None:
            inverse_bool = True
        else:
            inverse_bool = False

        if skip_step is not None:
            t_seq = t_seq[skip_step:]
            steps = steps - skip_step

        if noise_init is not None:
            noise_randn = torch.randn_like(noise)
            t_init = t_seq[0]
            sample = noise_init * (1 - t_init) + noise_randn * t_init

        if inverse_bool:
            t_seq = t_seq[::-1]
            desc = "Inversing"
            latent = {}
            kv = {}
        else:
            desc = "Sampling"

        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        for t_curr, t_prev in tqdm(t_pairs, desc=desc, disable=False):
            if inverse_bool:
                t_latent = t_prev
            else:
                t_latent = t_curr
                if not skip_latent_replacement:
                    if stage == 1:
                        sample = sample * latent_mask + latent[f"{t_latent}"].cuda() * (
                            1 - latent_mask
                        )
                    elif stage == 2:
                        match_1 = (
                            sample.coords.unsqueeze(1) == latent_mask.unsqueeze(0)
                        ).all(dim=-1)
                        match_2 = (
                            latent[f"{t_latent}"].coords.cuda().unsqueeze(1)
                            == latent_mask.unsqueeze(0)
                        ).all(dim=-1)
                        idx_1 = match_1.float().argmax(0)
                        idx_2 = match_2.float().argmax(0)
                        feats = sample.feats.clone()
                        feats[idx_1] = latent[f"{t_latent}"].feats.cuda()[idx_2]
                        sample = sample.replace(feats)
            sample = self.sample_once(
                model,
                sample,
                t_curr,
                t_prev,
                cond,
                cfg_strength,
                kv,
                self_kv_mask,
                cross_kv_mask,
                t_latent,
                is_text,
            )
            if inverse_bool:
                latent[f"{t_latent}"] = sample.cpu()
        return sample, latent, kv


def sample_sparse_structure_inverse_configurable(
    pipeline,
    cond_src,
    voxel_src,
    cfg_strength_stage_1_inverse,
    skip_step,
    is_text,
    num_steps: int = 12,
):
    """
    VoxHammer's sample_sparse_structure_inverse with configurable steps.

    Args:
        num_steps: Number of inversion steps (default 12 to match SuperGen)
    """
    stage = 1
    flow_model = pipeline.models["sparse_structure_flow_model"]
    encoder = pipeline.models["sparse_structure_encoder"]
    z_s = encoder(voxel_src)
    sigma_min = pipeline.sparse_structure_sampler.sigma_min

    # Use our configurable sampler instead of VoxHammer's hardcoded one
    sparse_structure_sampler = ConfigurableStepsSampler(sigma_min, steps=num_steps)

    if cfg_strength_stage_1_inverse is None:
        cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
    else:
        cfg_strength = cfg_strength_stage_1_inverse

    noise, ss_latent, ss_kv = sparse_structure_sampler.sample(
        flow_model,
        stage,
        z_s,
        cond_src,
        cfg_strength,
        skip_step=skip_step,
        is_text=is_text,
    )
    return noise, ss_latent, ss_kv


def sample_sparse_structure_denoise_configurable(
    pipeline,
    cond_tgt,
    noise,
    voxel_src,
    voxel_mask,
    ss_latent,
    ss_latent_mask,
    ss_kv,
    ss_self_kv_mask,
    ss_cross_kv_mask,
    cfg_strength_stage_1_forward,
    skip_step,
    re_init,
    is_text,
    skip_latent_replacement=True,
    full_mask=True,
    num_steps: int = 12,
):
    """
    VoxHammer's sample_sparse_structure_denoise with configurable steps.
    """
    stage = 1
    flow_model = pipeline.models["sparse_structure_flow_model"]
    sigma_min = pipeline.sparse_structure_sampler.sigma_min

    # Use our configurable sampler
    sparse_structure_sampler = ConfigurableStepsSampler(sigma_min, steps=num_steps)

    if cfg_strength_stage_1_forward is None:
        cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
        print(f"cfg_strength: {cfg_strength}")
    else:
        cfg_strength = cfg_strength_stage_1_forward

    if re_init:
        encoder = pipeline.models["sparse_structure_encoder"]
        noise_init = encoder(voxel_src)
    else:
        noise_init = None

    z_s, ss_latent, ss_kv = sparse_structure_sampler.sample(
        flow_model,
        stage,
        noise,
        cond_tgt,
        cfg_strength,
        ss_latent,
        ss_latent_mask,
        ss_kv,
        ss_self_kv_mask,
        ss_cross_kv_mask,
        skip_step,
        noise_init,
        is_text=is_text,
        skip_latent_replacement=skip_latent_replacement,
        full_mask=full_mask,
    )

    decoder = pipeline.models["sparse_structure_decoder"]
    voxel = decoder(z_s)
    voxel = voxel * voxel_mask + voxel_src * (1 - voxel_mask)
    return voxel


def sample_slat_inverse_configurable(
    pipeline,
    cond_src,
    slat_src,
    coords_mask,
    cfg_strength_stage_2_inverse,
    is_text,
    num_steps: int = 25,
):
    """
    SLAT inversion with configurable steps (adapted from VoxHammer's sample_slat_inverse).

    Args:
        pipeline: Trellis pipeline
        cond_src: Source conditioning
        slat_src: Source SLAT tensor
        coords_mask: Coordinates to invert (use all coords to invert everything)
        cfg_strength_stage_2_inverse: CFG strength for inversion
        is_text: Whether using text conditioning
        num_steps: Number of inversion steps

    Returns:
        slat_latent: Dictionary of intermediate latents
        slat_kv: Dictionary of KV caches
    """
    stage = 2
    coords_mask = torch.cat(
        [torch.zeros(coords_mask.shape[0], 1).int().cuda(), coords_mask], dim=1
    )
    sparse_tensor_mask = torch.zeros(
        slat_src.coords.shape[0], dtype=torch.bool, device="cuda"
    )
    for coord in coords_mask:
        sparse_tensor_mask[torch.all(slat_src.coords == coord, dim=1)] = True

    # Check if we have any matching coordinates
    num_matches = sparse_tensor_mask.sum().item()
    if num_matches == 0:
        raise ValueError(
            f"SLAT inversion failed: no coordinates matched between voxels and SLAT features. "
            f"coords_mask has {coords_mask.shape[0]} coords, slat_src has {slat_src.coords.shape[0]} coords. "
            f"This usually indicates a mismatch in normalization between voxelization and rendering."
        )

    slat_inverse = slat_src.replace(
        slat_src.feats[sparse_tensor_mask], slat_src.coords[sparse_tensor_mask]
    )

    flow_model = pipeline.models["slat_flow_model"]
    std = torch.tensor(pipeline.slat_normalization["std"], device=pipeline.device)[None]
    mean = torch.tensor(pipeline.slat_normalization["mean"], device=pipeline.device)[
        None
    ]
    slat_inverse = (slat_inverse - mean) / std

    sigma_min = pipeline.slat_sampler.sigma_min
    slat_sampler = ConfigurableStepsSampler(sigma_min, steps=num_steps)

    if cfg_strength_stage_2_inverse is None:
        cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
    else:
        cfg_strength = cfg_strength_stage_2_inverse

    noise, slat_latent, slat_kv = slat_sampler.sample(
        flow_model, stage, slat_inverse, cond_src, cfg_strength, is_text=is_text
    )
    return noise, slat_latent, slat_kv


def sample_slat_denoise_configurable(
    pipeline,
    cond_tgt,
    coords_tgt,
    slat_src,
    coords_mask,
    slat_latent,
    slat_kv,
    slat_self_kv_mask,
    slat_cross_kv_mask,
    cfg_strength_stage_2_forward,
    is_text,
    skip_latent_replacement=True,
    full_mask=True,
    num_steps: int = 25,
):
    """
    SLAT denoising (forward) with configurable steps (adapted from VoxHammer's sample_slat_denoise).
    """
    stage = 2
    coords_mask = torch.cat(
        [torch.zeros(coords_mask.shape[0], 1).int().cuda(), coords_mask], dim=1
    )
    flow_model = pipeline.models["slat_flow_model"]
    noise = sp.SparseTensor(
        feats=torch.randn(coords_tgt.shape[0], flow_model.in_channels).to(
            pipeline.device
        ),
        coords=coords_tgt,
    )

    sigma_min = pipeline.slat_sampler.sigma_min
    slat_sampler = ConfigurableStepsSampler(sigma_min, steps=num_steps)

    if cfg_strength_stage_2_forward is None:
        cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
    else:
        cfg_strength = cfg_strength_stage_2_forward

    slat, slat_latent, slat_kv = slat_sampler.sample(
        flow_model,
        stage,
        noise,
        cond_tgt,
        cfg_strength,
        slat_latent,
        coords_mask,
        slat_kv,
        slat_self_kv_mask,
        slat_cross_kv_mask,
        is_text=is_text,
        skip_latent_replacement=skip_latent_replacement,
        full_mask=full_mask,
    )

    std = torch.tensor(pipeline.slat_normalization["std"], device=pipeline.device)[None]
    mean = torch.tensor(pipeline.slat_normalization["mean"], device=pipeline.device)[
        None
    ]
    slat = slat * std + mean

    # Merge with original SLAT at masked coordinates
    match_1 = (coords_tgt.unsqueeze(1) == coords_mask.unsqueeze(0)).all(dim=-1)
    match_2 = (slat_src.coords.unsqueeze(1) == coords_mask.unsqueeze(0)).all(dim=-1)
    idx_1 = match_1.float().argmax(0)
    idx_2 = match_2.float().argmax(0)
    feats = slat.feats.clone()
    feats[idx_1] = slat_src.feats[idx_2]
    slat = slat.replace(feats)
    return slat


# Import voxel visualization from inpaint_data_preparation
from prox_e.inpaint_data_preparation import voxels_to_cube_mesh


def run_3d_rendering(input_model_path: str, render_dir: str, **render_kwargs) -> dict:
    """
    Step 1: Render 3D model to generate multi-view images (VoxHammer style).

    Args:
        input_model_path: Path to input 3D model file
        render_dir: Directory to save rendered images
        **render_kwargs: Additional rendering parameters

    Returns:
        Dictionary containing rendering results
    """
    print("=" * 50)
    print("STEP 1: 3D Model Rendering")
    print("=" * 50)

    # Check if already rendered
    transforms_path = os.path.join(render_dir, "transforms.json")
    mesh_path = os.path.join(render_dir, "mesh.ply")
    if os.path.exists(transforms_path) and os.path.exists(mesh_path):
        print(f"Render directory {render_dir} already exists, skipping rendering")
        return {
            "rendered": True,
            "num_views": 150,
            "output_dir": render_dir,
            "transforms_file": transforms_path,
            "mesh_file": mesh_path,
        }

    default_params = {
        "num_views": 150,
        "scale": 1.0,
        "offset": None,
        "resolution": 512,
        "engine": "BLENDER_EEVEE",
        "geo_mode": False,
        "split_normal": False,
        "save_mesh": True,
    }
    default_params.update(render_kwargs)

    print(f"Input model: {input_model_path}")
    print(f"Output directory: {render_dir}")
    print(
        f"Rendering parameters: num_views={default_params['num_views']}, resolution={default_params['resolution']}"
    )

    os.makedirs(render_dir, exist_ok=True)
    result = render_3d_model(
        file_path=input_model_path, output_dir=render_dir, **default_params
    )
    print("Rendering completed successfully!")
    print(f"Generated {result['num_views']} views")
    print(f"Transforms file: {result['transforms_file']}")
    if result.get("mesh_file"):
        print(f"Mesh file: {result['mesh_file']}")
    return result


def run_feature_extraction(
    render_dir: str, max_retries: int = 5, **feature_kwargs
) -> dict:
    """
    Step 2: Extract DinoV2 features from rendered images.

    Args:
        render_dir: Directory containing rendered images
        max_retries: Maximum number of retry attempts (default: 5)
        **feature_kwargs: Additional feature extraction parameters

    Returns:
        Dictionary containing feature extraction results
    """
    import time as time_module
    import traceback

    print("=" * 50)
    print("STEP 2: Feature Extraction (DinoV2)")
    print("=" * 50)

    # Check if already extracted
    features_path = os.path.join(render_dir, "features.npz")
    if os.path.exists(features_path):
        print(f"Features already exist at {features_path}, skipping extraction")
        return {"features_path": features_path}

    default_params = {"model": "dinov2_vitl14_reg", "batch_size": 4}
    default_params.update(feature_kwargs)
    print(f"Render directory: {render_dir}")
    print(
        f"Feature extraction parameters: model={default_params['model']}, batch_size={default_params['batch_size']}"
    )

    error_log_path = os.path.join(render_dir, "feature_extraction_errors.txt")

    for attempt in range(1, max_retries + 1):
        try:
            extract_features_single_thread(render_dir, **default_params)
            if os.path.exists(features_path):
                print("Feature extraction completed successfully!")
                print(f"Features saved to: {features_path}")
                return {"features_path": features_path}
            else:
                print(
                    f"Feature extraction failed - features file not created (attempt {attempt}/{max_retries})"
                )
        except Exception as e:
            error_msg = f"Attempt {attempt}/{max_retries} failed: {e}\n{traceback.format_exc()}\n"
            print(f"Feature extraction failed (attempt {attempt}/{max_retries}): {e}")

            # Log error (just print - S3 filesystems have write issues)
            print(f"Error details: {error_msg}")

        if attempt < max_retries:
            print("Waiting 5 seconds before retry...")
            time_module.sleep(5)
        else:
            raise RuntimeError(
                f"Feature extraction failed after {max_retries} attempts. See {error_log_path}"
            )


def voxelize_mesh(
    mesh_path: str,
    resolution: int = 64,
    center: np.ndarray = None,
    scale: float = None,
    remove_out_of_bounds: bool = False,
) -> torch.Tensor:
    """
    Convert a mesh to a voxel grid matching VoxHammer's format.

    Args:
        mesh_path: Path to mesh file (OBJ, PLY, etc.)
        resolution: Voxel grid resolution (default 64)
        center: Optional global center for normalization. If None, computed from mesh.
        scale: Optional global scale for normalization. If None, computed from mesh.
        remove_out_of_bounds: If True, removes vertices outside [-0.5, 0.5]^3 instead of
            raising an error. Default False.

    Returns:
        Voxel tensor of shape (1, 1, resolution, resolution, resolution) on CUDA

    Raises:
        ValueError: If any vertices are outside [-0.5, 0.5]^3 after normalization
            (only when remove_out_of_bounds is False)
    """
    import trimesh

    # Load mesh
    if mesh_path.endswith(".ply"):
        mesh = o3d.io.read_triangle_mesh(mesh_path)
    else:
        tm = trimesh.load(mesh_path, force="mesh")
        # Handle Scene objects (when OBJ has multiple meshes)
        if isinstance(tm, trimesh.Scene):
            meshes = [g for g in tm.geometry.values() if isinstance(g, trimesh.Trimesh)]
            if meshes:
                tm = trimesh.util.concatenate(meshes)
            else:
                raise ValueError(f"No valid meshes found in {mesh_path}")
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(tm.faces)

    mesh.compute_vertex_normals()

    # Get vertices
    vertices = np.asarray(mesh.vertices)

    # Compute center and scale if not provided
    if center is None:
        center = (vertices.max(0) + vertices.min(0)) / 2
    if scale is None:
        scale = (vertices.max(0) - vertices.min(0)).max()

    # Normalize mesh
    vertices = (vertices - center) / scale

    # Check bounds - handle vertices outside valid region
    # min_coords = vertices.min(0)
    # max_coords = vertices.max(0)
    # if np.any(min_coords < -0.5) or np.any(max_coords > 0.5):
    #    raise ValueError(
    #        f"Vertices out of bounds after normalization!\n"
    #        f"  Min coords: {min_coords} (should be >= -0.5)\n"
    #        f"  Max coords: {max_coords} (should be <= 0.5)\n"
    #        f"  Center used: {center}\n"
    #        f"  Scale used: {scale}\n"
    #        f"  Mesh path: {mesh_path}"
    #    )

    mesh.vertices = o3d.utility.Vector3dVector(vertices)

    # Voxelize
    voxel_size = 1.0 / resolution
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=voxel_size,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )

    # Convert to dense tensor matching VoxHammer format
    voxels = torch.zeros(1, 1, resolution, resolution, resolution, dtype=torch.float32)
    for voxel in voxel_grid.get_voxels():
        idx = voxel.grid_index
        if (
            0 <= idx[0] < resolution
            and 0 <= idx[1] < resolution
            and 0 <= idx[2] < resolution
        ):
            voxels[0, 0, idx[0], idx[1], idx[2]] = 1.0

    return voxels.cuda()


def setup_pipeline_for_inversion(pipeline, setup_slat: bool = True):
    """
    Monkey-patch the pipeline's flow models for VoxHammer's inversion.
    This modifies the forward methods to support KV caching and masking.

    Args:
        pipeline: Trellis pipeline
        setup_slat: If True, also setup SLAT flow model (requires image pipeline)
    """
    # Setup sparse structure flow model
    ss_flow = pipeline.models["sparse_structure_flow_model"]
    ss_flow.forward = MethodType(ss_flow_forward, ss_flow)

    for block in ss_flow.blocks:
        trsfmr_obj = block
        trsfmr_obj.forward = MethodType(ss_trsfmr_forward, trsfmr_obj)
        self_attn_obj = block.self_attn
        self_attn_obj.forward = MethodType(ss_attn_forward, self_attn_obj)
        cross_attn_obj = block.cross_attn
        cross_attn_obj.forward = MethodType(ss_attn_forward, cross_attn_obj)

    # Setup SLAT flow model (for appearance inversion)
    if setup_slat and "slat_flow_model" in pipeline.models:
        slat_flow = pipeline.models["slat_flow_model"]
        slat_flow.forward = MethodType(slat_flow_forward, slat_flow)

        for block in slat_flow.blocks:
            trsfmr_obj = block
            trsfmr_obj.forward = MethodType(slat_trsfmr_forward, trsfmr_obj)
            self_attn_obj = block.self_attn
            self_attn_obj.forward = MethodType(slat_attn_forward, self_attn_obj)
            cross_attn_obj = block.cross_attn
            cross_attn_obj.forward = MethodType(slat_attn_forward, cross_attn_obj)


def load_trellis_text_pipeline():
    """Load the TRELLIS text-to-3D pipeline."""
    print("Loading TrellisTextTo3DPipeline...")
    pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-large")
    pipeline.cuda()
    return pipeline


def apply_denormalization_to_mesh(
    mesh: o3d.geometry.TriangleMesh,
    center: np.ndarray,
    scale: float,
    is_voxel_mesh: bool = False,
) -> o3d.geometry.TriangleMesh:
    """
    Apply inverse normalization to an Open3D mesh.

    The forward normalization was: vertices = (vertices - center) / scale
    So the inverse is: vertices = vertices * scale + center

    Args:
        mesh: Open3D TriangleMesh to denormalize
        center: Center that was used for normalization
        scale: Scale that was used for normalization
        is_voxel_mesh: If True, accounts for axis swap done in voxels_to_cube_mesh

    Returns:
        The denormalized mesh (modified in place)
    """
    vertices = np.asarray(mesh.vertices)

    if is_voxel_mesh:
        # voxels_to_cube_mesh applies: pos = [x, -z, y]
        # We need to swap center accordingly: [cx, cy, cz] -> [cx, -cz, cy]
        adjusted_center = np.array([center[0], -center[2], center[1]])
        vertices = vertices * scale + adjusted_center
    else:
        vertices = vertices * scale + center

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


def save_voxels_as_ply(
    voxels: torch.Tensor,
    output_path: str,
    threshold: float = 0.5,
    normalization: Dict = None,
):
    """Save voxel grid as PLY mesh with cube visualization.

    Args:
        voxels: Voxel tensor to save
        output_path: Path to save the PLY file
        threshold: Threshold for voxel occupancy
        normalization: Optional dict with 'center' and 'scale' for denormalization
    """
    if voxels.dim() == 5:
        voxels = voxels[0, 0]  # Remove batch and channel dims

    voxels_np = (voxels.detach().cpu().numpy() > threshold).astype(np.float32)
    cube_mesh = voxels_to_cube_mesh(voxels_np, use_position_colors=True)

    # Apply denormalization (with axis swap correction for voxel meshes)
    if normalization is not None:
        center = normalization["center"]
        scale = normalization["scale"]
        apply_denormalization_to_mesh(cube_mesh, center, scale, is_voxel_mesh=True)
        print(f"  Applied denormalization to voxels (scale={scale})")

    o3d.io.write_triangle_mesh(output_path, cube_mesh, write_vertex_colors=True)


def compute_global_normalization(
    mesh_paths: List[str],
) -> Dict[str, Any]:
    """
    Compute global normalization parameters for consistent voxelization.

    Both center and scale are computed from the unified bounding box of all meshes.

    Args:
        mesh_paths: List of mesh paths to include in normalization computation

    Returns:
        Dict with 'center' (np.ndarray) and 'scale' (float)
    """
    import trimesh

    def load_vertices(mesh_path: str) -> np.ndarray:
        """Load vertices from a mesh file."""
        if mesh_path.endswith(".ply"):
            mesh = o3d.io.read_triangle_mesh(mesh_path)
            return np.asarray(mesh.vertices)
        else:
            tm = trimesh.load(mesh_path, force="mesh")
            if isinstance(tm, trimesh.Scene):
                meshes = [
                    g for g in tm.geometry.values() if isinstance(g, trimesh.Trimesh)
                ]
                if meshes:
                    tm = trimesh.util.concatenate(meshes)
                else:
                    raise ValueError(f"No valid meshes found in {mesh_path}")
            return np.array(tm.vertices)

    # Collect all vertices
    all_vertices = []
    for mesh_path in mesh_paths:
        if mesh_path and os.path.exists(mesh_path):
            vertices = load_vertices(mesh_path)
            all_vertices.append(vertices)

    if not all_vertices:
        raise ValueError("No valid meshes found for normalization")

    # Concatenate all vertices
    combined_vertices = np.vstack(all_vertices)

    # Compute center and scale from unified bounding box
    center = (combined_vertices.max(0) + combined_vertices.min(0)) / 2
    bbox_size = combined_vertices.max(0) - combined_vertices.min(0)
    scale = bbox_size.max()

    print(f"[GlobalNormalization] Computed from {len(all_vertices)} mesh(es)")
    print(f"  Center (from unified bbox): {center}")
    print(f"  Scale (from unified bbox): {scale}")
    print(f"  Total vertices: {len(combined_vertices)}")

    return {
        "center": center,
        "scale": scale,
    }


def run_full_inversion(
    original_mesh_path: str,
    abstraction_mesh_path: str,
    output_dir: str,
    external_normalization: Dict,
    text_prompt: str,
    merging_data: Dict = None,
    verify: bool = False,
    render_fn=None,
    num_steps: int = 25,
    num_views: int = 150,
    run_rendering: bool = True,
    run_features: bool = True,
    shade_smooth: bool = False,
    text_pipeline=None,
    image_pipeline=None,
) -> Dict[str, Any]:
    """
    Run full inversion pipeline on original mesh, abstraction, and transformed meshes.

    Computes global normalization from all meshes, then inverts each one using
    the same coordinate frame.

    Args:
        original_mesh_path: Path to the original mesh
        abstraction_mesh_path: Path to the edited abstraction mesh
        output_dir: Directory to save outputs
        text_prompt: Text prompt for conditioning (e.g., "a chair")
        merging_data: Optional dict containing 'changed_edited_masks' with transformed mesh paths
        verify: If True, verify inversion by reconstruction
        render_fn: Optional render function for visualization
        num_steps: Number of inversion/sampling steps (default 25)
        num_views: Number of views to render for appearance inversion (default 150)
        external_normalization: If provided, use this {'center': np.array, 'scale': float}
                                instead of computing from meshes
        run_rendering: If True, render multi-view images for original mesh (default True)
        run_features: If True, extract features and run SLAT inversion for original mesh (default True)
        num_steps: Number of inversion/sampling steps (default 25)
        num_views: Number of views to render for appearance inversion (default 150)
        external_normalization: If provided, use this {'center': np.array, 'scale': float}
                                instead of computing from meshes

    Returns:
        Dict containing:
        - pipeline: The loaded Trellis pipeline (for reuse in subsequent steps)
        - original_inversion_data: Inversion data from the original mesh
        - normalization: Dict with 'center' and 'scale' used for normalization
    """
    output_path = Path(output_dir)

    # Use external normalization if provided, otherwise ERROR
    print("\n[Inversion] Using external normalization provided by previous step...")
    global_norm = external_normalization
    norm_center = global_norm["center"]
    norm_scale = global_norm["scale"]
    print(f"  Center: {norm_center}")
    print(f"  Scale: {norm_scale}")

    # Load pipeline if not provided
    pipeline = (
        text_pipeline if text_pipeline is not None else load_trellis_text_pipeline()
    )

    # Accumulate sub-timings across all inversion calls
    _total_structure_time = 0.0
    _total_slat_time = 0.0

    # Invert each transformed mesh
    if merging_data is not None and "changed_edited_masks" in merging_data:
        for mask_data in merging_data["changed_edited_masks"]:
            transformed_mesh_path = mask_data.get("transformed_mesh_path")
            sq_index = mask_data.get("sq_index", "unknown")

            if transformed_mesh_path and os.path.exists(transformed_mesh_path):
                print(f"\n  Inverting transformed mesh for SQ {sq_index}...")
                _inv_result = run_voxhammer_inversion(
                    pipeline=pipeline,
                    mesh_path=transformed_mesh_path,
                    text_prompt=text_prompt,
                    output_dir=output_dir,
                    filename=f"transformed_sq{sq_index}",
                    verify=verify,
                    render_fn=render_fn if verify else None,
                    num_steps=num_steps,
                    run_rendering=False,
                    run_features=False,
                    normalization_center=norm_center,
                    normalization_scale=norm_scale,
                    shade_smooth=shade_smooth,
                    image_pipeline=image_pipeline,
                )
                _total_structure_time += _inv_result.get("timing", {}).get(
                    "structure_inversion", 0
                )

    # Invert the edited abstraction mesh
    _abs_result = run_voxhammer_inversion(
        pipeline=pipeline,
        mesh_path=abstraction_mesh_path,
        text_prompt=text_prompt,
        output_dir=output_dir,
        filename="abstraction",
        verify=verify,
        render_fn=render_fn if verify else None,
        num_steps=num_steps,
        run_rendering=False,
        run_features=False,
        normalization_center=norm_center,
        normalization_scale=norm_scale,
        shade_smooth=shade_smooth,
        image_pipeline=image_pipeline,
    )
    _total_structure_time += _abs_result.get("timing", {}).get("structure_inversion", 0)

    # Invert the original mesh (with multi-view rendering for appearance)
    original_inversion_data = run_voxhammer_inversion(
        pipeline=pipeline,
        mesh_path=original_mesh_path,
        text_prompt=text_prompt,
        output_dir=output_dir,
        filename="original_shape",
        verify=verify,
        render_fn=render_fn if verify else None,
        num_steps=num_steps,
        run_rendering=run_rendering,
        run_features=run_features,
        num_views=num_views,
        normalization_center=norm_center,
        normalization_scale=norm_scale,
        shade_smooth=shade_smooth,
        image_pipeline=image_pipeline,
    )
    _orig_timing = original_inversion_data.get("timing", {})
    _total_structure_time += _orig_timing.get("structure_inversion", 0)
    _total_slat_time += _orig_timing.get("slat_inversion", 0)
    # Capture image_pipeline (may have been loaded during SLAT inversion)
    if image_pipeline is None:
        image_pipeline = original_inversion_data.get("image_pipeline")

    # === Create combined debug mesh with all voxelized components ===
    print("\n[Debug] Creating combined debug mesh...")
    inversion_dir = output_path / "inversion"
    combined_voxels = None

    # Collect all voxel tensors from inversion
    voxel_files = [
        ("original_shape", inversion_dir / "original_shape_voxels.pt"),
        ("abstraction", inversion_dir / "abstraction_voxels.pt"),
    ]

    # Add transformed mesh voxels
    if merging_data is not None and "changed_edited_masks" in merging_data:
        for mask_data in merging_data["changed_edited_masks"]:
            sq_index = mask_data.get("sq_index")
            voxel_path = inversion_dir / f"transformed_sq{sq_index}_voxels.pt"
            if voxel_path.exists():
                voxel_files.append((f"transformed_sq{sq_index}", voxel_path))

    # Load and combine voxels (all on CPU for debug mesh)
    for name, vpath in voxel_files:
        if vpath.exists():
            voxels = torch.load(vpath, weights_only=False)
            if voxels.dim() == 5:
                voxels = voxels[0, 0]  # Remove batch and channel dims
            voxels_binary = (voxels > 0.5).float().cpu()  # Ensure CPU
            if combined_voxels is None:
                combined_voxels = voxels_binary
            else:
                combined_voxels = torch.maximum(combined_voxels, voxels_binary)
            print(f"  Added {name}: {int(voxels_binary.sum().item())} voxels")

    # Add masks from merging_data (convert from latent to voxel resolution if needed)
    if merging_data is not None:
        # Add changed_original_mask_voxel
        if "changed_original_mask_voxel" in merging_data:
            mask = merging_data["changed_original_mask_voxel"]
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = (mask > 0.5).float().cpu()
            if combined_voxels is not None and mask.shape == combined_voxels.shape:
                combined_voxels = torch.maximum(combined_voxels, mask)
                print(f"  Added changed_original_mask: {int(mask.sum().item())} voxels")

        # Add added_deleted_mask_voxel
        if "added_deleted_mask_voxel" in merging_data:
            mask = merging_data["added_deleted_mask_voxel"]
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask)
            mask = (mask > 0.5).float().cpu()
            if combined_voxels is not None and mask.shape == combined_voxels.shape:
                combined_voxels = torch.maximum(combined_voxels, mask)
                print(f"  Added added_deleted_mask: {int(mask.sum().item())} voxels")

        # Add each changed_edited_mask_voxel
        if "changed_edited_masks" in merging_data:
            for i, mask_data in enumerate(merging_data["changed_edited_masks"]):
                if "mask_voxel" in mask_data:
                    mask = mask_data["mask_voxel"]
                    if isinstance(mask, np.ndarray):
                        mask = torch.from_numpy(mask)
                    mask = (mask > 0.5).float().cpu()
                    if (
                        combined_voxels is not None
                        and mask.shape == combined_voxels.shape
                    ):
                        combined_voxels = torch.maximum(combined_voxels, mask)
                        sq_index = mask_data.get("sq_index", i)
                        print(
                            f"  Added edited_mask_{sq_index}: {int(mask.sum().item())} voxels"
                        )

    # Save combined debug mesh
    if combined_voxels is not None:
        combined_ply = output_path / "debug_combined_voxels.ply"
        # Add batch/channel dims for save_voxels_as_ply
        combined_voxels_5d = combined_voxels.unsqueeze(0).unsqueeze(0)
        save_voxels_as_ply(
            combined_voxels_5d, str(combined_ply), normalization=global_norm
        )
        print(f"  Saved combined debug mesh: {combined_ply}")
        print(
            f"  Total voxels in combined mesh: {int((combined_voxels > 0.5).sum().item())}"
        )

    return {
        "pipeline": pipeline,
        "image_pipeline": image_pipeline,
        "original_inversion_data": original_inversion_data,
        "normalization": global_norm,
        "timings": {
            "structure_inversion": _total_structure_time,
            "slat_inversion": _total_slat_time,
        },
    }


def compute_reconstruction_error(
    original_voxels: torch.Tensor,
    reconstructed_voxels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute reconstruction error metrics."""
    orig = (original_voxels.detach().cpu() > threshold).float()
    recon = (reconstructed_voxels.detach().cpu() > threshold).float()

    # MSE
    mse = torch.mean((orig - recon) ** 2).item()

    # IoU
    intersection = (orig * recon).sum().item()
    union = ((orig + recon) > 0).float().sum().item()
    iou = intersection / (union + 1e-8)

    # Counts
    orig_count = orig.sum().item()
    recon_count = recon.sum().item()

    return {
        "mse": mse,
        "iou": iou,
        "original_voxels": int(orig_count),
        "reconstructed_voxels": int(recon_count),
    }


def run_voxhammer_inversion(
    pipeline,
    mesh_path: str,
    text_prompt: str,
    output_dir: str,
    filename: str,
    verify: bool = True,
    render_fn=None,
    skip_step: int = 0,
    cfg_strength: float = None,
    cfg_strength_slat: float = None,
    num_steps: int = 25,
    run_rendering: bool = True,
    run_features: bool = True,
    num_views: int = 150,
    normalization_center: np.ndarray = None,
    normalization_scale: float = None,
    shade_smooth: bool = False,
    image_pipeline=None,
) -> Dict[str, Any]:
    """
    Run VoxHammer-based inversion on a mesh (structure and appearance).

    Pipeline order:
    1. Voxelize mesh and run sparse structure inversion
    2. Render multi-view images (if features don't exist)
    3. Extract DinoV2 features (if features don't exist)
    4. Extract SLAT from features and run appearance inversion
    5. Optionally verify both inversions

    Args:
        pipeline: TrellisTextTo3DPipeline instance (will be modified in-place)
        mesh_path: Path to input mesh
        text_prompt: Text prompt for conditioning (e.g., "a chair")
        output_dir: Directory to save outputs
        filename: Filename to save outputs
        verify: If True, verify inversion by reconstruction
        render_fn: Optional render function for visualization
        skip_step: Number of steps to skip (VoxHammer default)
        cfg_strength: CFG strength for structure inversion (None = use default)
        cfg_strength_slat: CFG strength for appearance inversion (None = use default)
        num_steps: Number of inversion/sampling steps (default 25)
        run_rendering: If True, render multi-view images and extract features
        run_features: If True, extract SLAT and run appearance inversion
        num_views: Number of views to render (default 150)
        normalization_center: Global center for mesh normalization (None = compute from mesh)
        normalization_scale: Global scale for mesh normalization (None = compute from mesh)

    Returns:
        Dict containing inversion results
    """
    output_dir = Path(output_dir)
    inversion_dir = output_dir / "inversion"
    inversion_dir.mkdir(parents=True, exist_ok=True)
    render_dir = str(inversion_dir / f"{filename}_renders")
    features_path = os.path.join(render_dir, "features.npz")

    result = {}

    # ========================================
    # STEP 1: SPARSE STRUCTURE INVERSION
    # ========================================
    _ss_start = time.time()
    print("=" * 60)
    print("STEP 1: SPARSE STRUCTURE INVERSION")
    print("=" * 60)

    # Voxelize the mesh and transform to Trellis orientation
    print(f"Voxelizing mesh: {mesh_path}")
    if normalization_center is not None:
        print(f"  Using global center: {normalization_center}")
    if normalization_scale is not None:
        print(f"  Using global scale: {normalization_scale}")
    voxel_src = voxelize_mesh(
        mesh_path, center=normalization_center, scale=normalization_scale
    )
    voxel_src = transform_voxels_to_trellis(voxel_src)
    voxel_count = (voxel_src > 0.5).sum().item()
    print(f"  Voxel count: {int(voxel_count)} (transformed to Trellis orientation)")

    # Save original voxels as .pt (same format as inpainted_voxels.pt)
    original_voxels_pt = inversion_dir / f"{filename}_voxels.pt"
    torch.save(voxel_src, original_voxels_pt)
    print(f"  Saved voxels tensor: {original_voxels_pt}")

    # Save original voxels with denormalization
    original_ply = inversion_dir / f"{filename}_original_voxels.ply"
    normalization = {"center": normalization_center, "scale": normalization_scale}
    save_voxels_as_ply(voxel_src, str(original_ply), normalization=normalization)
    print(f"  Saved: {original_ply}")

    # Setup pipeline for inversion (monkey-patch forward methods)
    print("Setting up pipeline for VoxHammer inversion...")
    setup_pipeline_for_inversion(pipeline, setup_slat=run_features)

    # Get conditioning
    print(f"Encoding prompt: '{text_prompt}'")
    cond_src = pipeline.get_cond([text_prompt])

    # Run sparse structure inversion
    print(
        f"\nRunning sparse structure inversion (steps={num_steps}, skip_step={skip_step}, is_text=True)..."
    )
    noise, ss_latent, ss_kv = sample_sparse_structure_inverse_configurable(
        pipeline=pipeline,
        cond_src=cond_src,
        voxel_src=voxel_src,
        cfg_strength_stage_1_inverse=cfg_strength,
        skip_step=skip_step,
        is_text=True,
        num_steps=num_steps,
    )

    # Save structure inversion results
    latents_path = inversion_dir / f"{filename}_ss_latents.pt"
    torch.save(
        {
            "noise": noise.cpu() if hasattr(noise, "cpu") else noise,
            "ss_latent": ss_latent,
            "text_prompt": text_prompt,
            "mesh_path": mesh_path,
            "skip_step": skip_step,
            "cfg_strength": cfg_strength,
            "num_steps": num_steps,
        },
        latents_path,
    )
    print(f"Saved structure latents: {latents_path}")

    _ss_time = time.time() - _ss_start

    result.update(
        {
            "noise": noise,
            "ss_latent": ss_latent,
            "ss_kv": ss_kv,
            "latents_path": str(latents_path),
            "original_voxels": voxel_src,
            "cond_src": cond_src,
        }
    )

    # Verify structure inversion if requested
    if verify:
        print("\nVerifying structure inversion (reconstruction)...")

        # Create mask that preserves everything (no masking)
        voxel_mask = torch.zeros_like(voxel_src)
        ss_latent_mask = torch.zeros(1, 8, 16, 16, 16).cuda()

        # Reconstruct
        recon_voxels = sample_sparse_structure_denoise_configurable(
            pipeline=pipeline,
            cond_tgt=cond_src,
            noise=noise,
            voxel_src=voxel_src,
            voxel_mask=voxel_mask,
            ss_latent=ss_latent,
            ss_latent_mask=ss_latent_mask,
            ss_kv=ss_kv,
            ss_self_kv_mask=None,
            ss_cross_kv_mask=None,
            cfg_strength_stage_1_forward=cfg_strength,
            skip_step=skip_step,
            re_init=False,
            is_text=True,
            skip_latent_replacement=True,
            full_mask=True,
            num_steps=num_steps,
        )

        # Compute error
        errors = compute_reconstruction_error(voxel_src, recon_voxels)
        print(f"  Structure MSE: {errors['mse']:.6f}")
        print(f"  Structure IoU: {errors['iou']:.4f}")

        # Save reconstructed voxels
        recon_ply = inversion_dir / f"{filename}_reconstructed_voxels.ply"
        save_voxels_as_ply(recon_voxels, str(recon_ply))
        print(f"  Saved: {recon_ply}")

        result["reconstructed_voxels"] = recon_voxels
        result["structure_reconstruction_errors"] = errors

        # Render if function provided
        # Use rotation to align Trellis-oriented voxels with standard view
        trellis_rotation = (-90, 0, 0)
        if render_fn is not None:
            try:
                render_fn(
                    str(original_ply),
                    str(inversion_dir / f"{filename}_original_voxels.png"),
                    rotation=trellis_rotation,
                )
                render_fn(
                    str(recon_ply),
                    str(inversion_dir / f"{filename}_reconstructed_voxels.png"),
                    rotation=trellis_rotation,
                )
            except Exception as e:
                print(f"  Render failed: {e}")

    # ========================================
    # STEP 2: RENDERING AND FEATURE EXTRACTION
    # ========================================
    _slat_start = time.time()
    if run_rendering:
        print("\n" + "=" * 60)
        print("STEP 2: RENDERING AND FEATURE EXTRACTION")
        print("=" * 60)

        # Only render/extract if features don't exist
        if os.path.exists(features_path):
            print(f"Features already exist at {features_path}, skipping rendering")
            result["features"] = {"features_path": features_path}
        else:
            # Let the renderer use its default normalization (normalize_scene)
            # which correctly centers and scales the mesh for rendering.
            # We pass offset=None to use automatic normalization.
            # Note: This means SLAT coordinates will be in the renderer's normalized
            # space, which may differ from our voxelization normalization.
            render_results = run_3d_rendering(
                mesh_path,
                render_dir,
                num_views=num_views,
                shade_smooth=shade_smooth,
                # Use default normalization (offset=None triggers normalize_scene)
            )
            result["rendering"] = render_results

            # Extract features
            feature_results = run_feature_extraction(render_dir)
            result["features"] = feature_results

    # ========================================
    # STEP 3: APPEARANCE (SLAT) INVERSION
    # ========================================
    if run_features and os.path.exists(features_path):
        print("\n" + "=" * 60)
        print("STEP 3: APPEARANCE (SLAT) INVERSION")
        print("=" * 60)

        # Render the mesh for conditioning using VoxHammer-style rendering
        conditioning_image_path = output_dir / "conditioning_render.png"
        print(f"Rendering mesh for conditioning: {mesh_path}")
        render_for_conditioning(
            str(mesh_path), str(conditioning_image_path), shade_smooth=shade_smooth
        )
        preprocessed_image = preprocess_single_image(str(conditioning_image_path))

        # Extract SLAT from features
        print("Extracting SLAT from features...")
        slat_src = feats_to_slat(pipeline, features_path)
        print(f"  SLAT shape: {slat_src.feats.shape}")

        # Get coordinates for inversion (all coordinates)
        # Use SLAT's own coordinates (not transformed voxels) to ensure coordinate system match
        # slat_src.coords has shape [N, 4] where first column is batch index (always 0)
        # We need [N, 3] with [x, y, z] for the coords_mask
        coords_src = (
            slat_src.coords[:, 1:].int().cuda()
        )  # Remove batch dim, keep [x, y, z]

        # Run SLAT inversion
        print(f"Running SLAT inversion (steps={num_steps})...")
        if image_pipeline is None:
            image_pipeline = load_trellis_image_pipeline()
        cond_src = image_pipeline.get_cond([preprocessed_image])
        setup_pipeline_for_inversion(image_pipeline)
        flow_model = image_pipeline.models["slat_flow_model"]
        flow_model.forward = MethodType(slat_flow_forward, flow_model)
        for block in flow_model.blocks:
            trsfmr_obj = block
            trsfmr_obj.forward = MethodType(slat_trsfmr_forward, trsfmr_obj)
            self_attn_obj = block.self_attn
            self_attn_obj.forward = MethodType(slat_attn_forward, self_attn_obj)
            cross_attn_obj = block.cross_attn
            cross_attn_obj.forward = MethodType(slat_attn_forward, cross_attn_obj)

        slat_noise, slat_latent, slat_kv = sample_slat_inverse_configurable(
            pipeline=image_pipeline,
            cond_src=cond_src,
            slat_src=slat_src,
            coords_mask=coords_src,
            cfg_strength_stage_2_inverse=cfg_strength_slat,
            is_text=False,
            num_steps=num_steps,
        )

        # Save SLAT inversion results
        slat_latents_path = inversion_dir / f"{filename}_slat_latents.pt"
        torch.save(
            {
                "slat_noise": slat_noise.cpu()
                if hasattr(slat_noise, "cpu")
                else slat_noise,
                "slat_latent": slat_latent,
                # slat_kv excluded - very large
                "coords_src": coords_src.cpu(),
                "cfg_strength_slat": cfg_strength_slat,
            },
            slat_latents_path,
        )
        print(f"Saved SLAT latents: {slat_latents_path}")

        result.update(
            {
                "slat_src": slat_src,
                "slat_noise": slat_noise,
                "slat_latent": slat_latent,
                "slat_kv": slat_kv,
                "slat_latents_path": str(slat_latents_path),
                "coords_src": coords_src,
                "preprocessed_image": preprocessed_image,
                "image_cond": cond_src,
                "conditioning_image_path": str(conditioning_image_path),
            }
        )

        # Always decode and save original SLAT to main folder (regardless of verification)
        print("  Decoding and saving original SLAT...")
        try:
            torch.set_grad_enabled(True)
            from prox_e.utils import render_obj_with_blender
            import trimesh

            trellis_rotation = (-90, 0, 0)  # Aligns Trellis output with standard view

            # Decode and render original SLAT
            assets_orig = pipeline.decode_slat(slat_src, ["gaussian", "mesh"])
            glb_orig = postprocessing_utils.to_glb(
                assets_orig["gaussian"][0],
                assets_orig["mesh"][0],
                simplify=0.95,
                texture_size=1024,
            )
            # Save to main output folder (parent of inversion_dir)
            orig_glb_path = output_dir / "original_slat.glb"
            glb_orig.export(str(orig_glb_path))

            # Apply denormalization using RENDERER's normalization (not global).
            # The SLAT coords came from feats_to_slat which uses the renderer's normalized mesh.
            # Renderer's forward: normalized = original * scale + offset
            # Inverse: original = (normalized - offset) / scale
            transforms_path = os.path.join(render_dir, "transforms.json")
            if os.path.exists(transforms_path):
                with open(transforms_path, "r") as f:
                    transforms = json.load(f)
                renderer_scale = transforms["scale"]
                renderer_offset = np.array(transforms["offset"])

                # Build inverse transform: original = (normalized - offset) / scale
                inv_scale = 1.0 / renderer_scale
                inv_offset = -renderer_offset / renderer_scale
                transform = np.eye(4)
                transform[:3, :3] *= inv_scale
                transform[:3, 3] = inv_offset

                scene = trimesh.load(str(orig_glb_path))
                if isinstance(scene, trimesh.Scene):
                    for geom in scene.geometry.values():
                        if isinstance(geom, trimesh.Trimesh):
                            geom.apply_transform(transform)
                elif isinstance(scene, trimesh.Trimesh):
                    scene.apply_transform(transform)
                scene.export(str(orig_glb_path), file_type="glb")
                print(
                    f"    Applied renderer denormalization (scale={renderer_scale:.4f})"
                )
            else:
                print(
                    "    Warning: transforms.json not found, skipping denormalization"
                )

            print(f"    Saved: {orig_glb_path}")

            # Render original SLAT with rotation to align coordinate systems
            render_obj_with_blender(
                str(orig_glb_path),
                str(output_dir / "original_slat.png"),
                rotation=trellis_rotation,
                shade_smooth=shade_smooth,
            )
            print(f"    Saved render: {output_dir / 'original_slat.png'}")

            torch.set_grad_enabled(False)
        except Exception as e:
            print(f"  Warning: Original SLAT export failed: {e}")

        # Verify SLAT inversion if requested (reconstruction comparison)
        if verify:
            print("\nVerifying SLAT inversion (reconstruction)...")

            # Get target coords (same as source for reconstruction)
            coords_tgt = torch.cat(
                [torch.zeros(coords_src.shape[0], 1).int().cuda(), coords_src], dim=1
            )

            # Reconstruct SLAT using denoising
            recon_slat = sample_slat_denoise_configurable(
                pipeline=image_pipeline,
                cond_tgt=cond_src,
                coords_tgt=coords_tgt,
                slat_src=slat_src,
                coords_mask=coords_src,
                slat_latent=slat_latent,
                slat_kv=slat_kv,
                slat_self_kv_mask=None,
                slat_cross_kv_mask=None,
                cfg_strength_stage_2_forward=cfg_strength_slat,
                is_text=False,
                skip_latent_replacement=True,
                full_mask=True,
                num_steps=num_steps,
            )

            # Compute SLAT reconstruction error
            slat_mse = torch.mean((slat_src.feats - recon_slat.feats) ** 2).item()
            print(f"  SLAT MSE: {slat_mse:.6f}")

            result["reconstructed_slat"] = recon_slat
            result["slat_reconstruction_mse"] = slat_mse

            # Render reconstructed SLAT for comparison
            print("  Rendering reconstructed SLAT...")
            try:
                torch.set_grad_enabled(True)
                from prox_e.utils import render_obj_with_blender

                trellis_rotation = (
                    -90,
                    0,
                    0,
                )  # Aligns Trellis output with standard view

                assets_recon = pipeline.decode_slat(recon_slat, ["gaussian", "mesh"])
                glb_recon = postprocessing_utils.to_glb(
                    assets_recon["gaussian"][0],
                    assets_recon["mesh"][0],
                    simplify=0.95,
                    texture_size=1024,
                )
                recon_glb_path = inversion_dir / f"{filename}_reconstructed_slat.glb"
                glb_recon.export(str(recon_glb_path))
                print(f"    Saved: {recon_glb_path}")

                # Render reconstructed SLAT with same rotation
                render_obj_with_blender(
                    str(recon_glb_path),
                    str(inversion_dir / f"{filename}_reconstructed_slat.png"),
                    rotation=trellis_rotation,
                )
                print(f"    Saved renders to: {inversion_dir}")

                torch.set_grad_enabled(False)
            except Exception as e:
                print(f"  Warning: SLAT rendering failed: {e}")

    _slat_time = time.time() - _slat_start

    result["timing"] = {
        "structure_inversion": _ss_time,
        "slat_inversion": _slat_time,
    }
    result["image_pipeline"] = image_pipeline

    print("\nInversion complete!")
    return result


# Backward compatibility: alias for the main function
run_sparse_structure_inversion = run_voxhammer_inversion


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run VoxHammer inversion")
    parser.add_argument("--mesh", type=str, required=True, help="Path to input mesh")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--verify", action="store_true", help="Verify inversion")
    parser.add_argument("--render", action="store_true", help="Render visualizations")
    parser.add_argument("--skip_step", type=int, default=0, help="Steps to skip")
    parser.add_argument("--cfg", type=float, default=None, help="CFG strength")
    parser.add_argument(
        "--steps", type=int, default=25, help="Number of inversion steps (default 12)"
    )

    args = parser.parse_args()

    # Load pipeline
    pipeline = load_trellis_text_pipeline()

    render_fn = None
    if args.render:
        from prox_e.utils import render_obj_with_blender

        render_fn = render_obj_with_blender

    run_voxhammer_inversion(
        pipeline=pipeline,
        mesh_path=args.mesh,
        text_prompt=args.prompt,
        output_dir=args.output,
        verify=args.verify,
        render_fn=render_fn,
        skip_step=args.skip_step,
        cfg_strength=args.cfg,
        num_steps=args.steps,
    )
