import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Union, Dict
from types import MethodType

import rembg
from PIL import Image

# Add VoxHammer submodule to path
VOXHAMMER_PATH = Path(__file__).parent / "submodules" / "voxhammer"
sys.path.insert(0, str(VOXHAMMER_PATH))

# Add supergen submodule to path (for trellis)
SUPERGEN_PATH = Path(__file__).parent / "submodules" / "supergen"
sys.path.insert(0, str(SUPERGEN_PATH))

from voxhammer.edit_pipeline import (
    InversionFlowEulerGuidanceIntervalSampler,
    slat_flow_forward,
    slat_trsfmr_forward,
    slat_attn_forward,
)
import trellis.modules.sparse as sp
from trellis.utils import postprocessing_utils
from trellis.pipelines import TrellisImageTo3DPipeline

# Import bpyrenderer for VoxHammer-style rendering
import trimesh


def apply_denormalization_to_glb(
    glb_path: str, center: np.ndarray, scale: float, output_path: str = None
):
    """
    Apply inverse normalization to a GLB file to restore original scale and position.

    The forward normalization was: vertices = (vertices - center) / scale
    So the inverse is: vertices = vertices * scale + center

    Args:
        glb_path: Path to the normalized GLB file
        center: Center that was used for normalization
        scale: Scale that was used for normalization
        output_path: Optional output path (if None, overwrites input)

    Returns:
        Path to the denormalized GLB file
    """
    if output_path is None:
        output_path = glb_path

    # Load the GLB as a scene
    scene = trimesh.load(glb_path)

    # Create transformation matrix: scale then translate
    # vertices_denorm = vertices_norm * scale + center
    transform = np.eye(4)
    transform[:3, :3] *= scale  # Scale
    transform[:3, 3] = center  # Translate

    # Apply to all meshes in the scene
    if isinstance(scene, trimesh.Scene):
        for name, geom in scene.geometry.items():
            if isinstance(geom, trimesh.Trimesh):
                geom.apply_transform(transform)
    elif isinstance(scene, trimesh.Trimesh):
        scene.apply_transform(transform)

    # Export back to GLB
    scene.export(output_path, file_type="glb")
    print(f"Applied denormalization to mesh: {output_path}")
    print(f"  Scale: {scale}, Center: {center}")

    return output_path


def load_trellis_image_pipeline():
    """Load the TRELLIS image-to-3D pipeline."""
    print("Loading TrellisImageTo3DPipeline...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    return pipeline


def render_for_conditioning(
    input_model: str,
    output_path: str,
    elevation: float = 20,
    azimuth: float = 70,
    distance: float = 1.5,
    shade_smooth: bool = False,
    rotation=None,
):
    """
    Render a model for appearance conditioning using the same Blender render path
    as original_slat.png/output.png, while keeping the conditioning camera angles.

    For Trellis-decoded GLB (e.g. original_slat.glb), pass rotation equal to
    prox_e.utils.TRELLIS_DECODE_GLB_BLENDER_ROTATION_DEG so the result matches
    original_slat.png aside from camera. Plain pipeline meshes (normalized OBJ) should
    use rotation=None.

    Args:
        input_model: Path to the input model (OBJ, GLB, etc.)
        output_path: Path to save the rendered image
        elevation: Camera elevation in degrees (default: 20)
        azimuth: Camera azimuth in degrees (default: 70)
        distance: Camera distance from center (default: 1.5)
        shade_smooth: If True, apply smooth shading to mesh objects (default: False)
        rotation: Optional (rx, ry, rz) degrees for Blender vertex rotation, or None
    """
    from prox_e.utils import render_obj_with_blender

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(
        dist=distance,
        azim=azimuth,
        elev=elevation,
        shade_smooth=shade_smooth,
    )
    if rotation is not None:
        kwargs["rotation"] = rotation
    result = render_obj_with_blender(
        input_model,
        output_path,
        **kwargs,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Conditioning render failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    print(f"Rendered conditioning image: {output_path}")


def preprocess_single_image(
    img_path, remove_background=False, skip_rembg=False
) -> Image.Image:
    """
    Preprocess a single image for TRELLIS image conditioning.

    This is a simplified version of VoxHammer's preprocess_image that handles
    only a single image (no edit or mask).

    Args:
        img_path: Path to the input image
        remove_background: If True, run rembg (only used when not skipping rembg; see skip_rembg)
        skip_rembg: If True, skip rembg entirely. RGB images become opaque RGBA (use for edited
            RGB exports where u2net often mis-segments and yields a black premultiplied result).

    Returns:
        Preprocessed PIL Image (518x518, RGB, background removed)
    """
    # Load image and convert to RGB
    image = Image.open(img_path)
    if image.mode == "RGB":
        image_rgb = image
    else:
        image = image.convert("RGBA")
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image_rgb = background

    # Resize if too large
    max_size = max(image_rgb.size)
    scale = min(1, 1024 / max_size)
    if scale < 1:
        resize_size = (int(image_rgb.width * scale), int(image_rgb.height * scale))
        image_rgb = image_rgb.resize(resize_size, Image.Resampling.LANCZOS)

    # Remove background (RGB images from Blender need rembg; skip for e.g. Kontext RGB edits)
    if skip_rembg:
        pre_img = image_rgb.convert("RGBA") if image.mode == "RGB" else image
    elif remove_background or image.mode == "RGB":
        pre_img = rembg.remove(image_rgb, session=rembg.new_session("u2net"))
    else:
        pre_img = image
    pre_img_np = np.array(pre_img)

    # Get bounding box from alpha channel
    alpha = pre_img_np[:, :, 3]
    bbox_coords = np.argwhere(alpha > 0.8 * 255)
    if len(bbox_coords) == 0:
        # No alpha, use full image
        bbox = (0, 0, image_rgb.width, image_rgb.height)
    else:
        bbox = (
            np.min(bbox_coords[:, 1]),
            np.min(bbox_coords[:, 0]),
            np.max(bbox_coords[:, 1]),
            np.max(bbox_coords[:, 0]),
        )

    # Expand bbox slightly and make square
    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    bbox = (
        center[0] - size // 2,
        center[1] - size // 2,
        center[0] + size // 2,
        center[1] + size // 2,
    )

    # Crop and resize
    pre_img = pre_img.crop(bbox)
    pre_img = pre_img.resize((518, 518), Image.Resampling.LANCZOS)

    # Apply alpha compositing (multiply RGB by alpha)
    pre_img_np = np.array(pre_img).astype(np.float32) / 255
    pre_img_np = pre_img_np[:, :, :3] * pre_img_np[:, :, 3:4]
    pre_img = Image.fromarray((pre_img_np * 255).astype(np.uint8))

    # Save preprocessed image for debugging
    pre_img.save(img_path.replace(".png", "_preprocessed.png"))

    return pre_img


class AppearanceEditingSampler(InversionFlowEulerGuidanceIntervalSampler):
    """Sampler with appearance editing support and latent injection."""

    def __init__(self, sigma_min: float, steps: int = 25):
        super().__init__(sigma_min)
        self.steps = steps

    @torch.no_grad()
    def sample_appearance_editing(
        self,
        model,
        noise,
        cond,
        cfg_strength,
        slat_latent=None,
        preserve_coords_mask=None,
        injection_steps: int = 0,
        edit_injection_mappings=None,
        added_deleted_coords_mask=None,
        abstraction_injection_steps: int = 0,
    ):
        """
        Run appearance editing with optional latent injection.

        Args:
            model: SLAT flow model
            noise: Initial noise (SparseTensor)
            cond: Conditioning
            cfg_strength: CFG strength
            slat_latent: Dict of inverted SLAT latents at each timestep (from inversion)
            preserve_coords_mask: Coordinates where we should inject inverted latents (unchanged regions)
            injection_steps: Number of steps to inject latents (0 = no injection)
            edit_injection_mappings: List of precomputed dicts with 'sample_indices' and 'nn_indices' for edit regions
            added_deleted_coords_mask: Coordinates in added/deleted regions for abstraction injection
            abstraction_injection_steps: Number of steps to inject latents into added/deleted regions
        """
        steps = self.steps
        rescale_t = 3.0
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        kv = {}

        for i, (t_curr, t_prev) in enumerate(tqdm(t_pairs, desc="Appearance Editing")):
            sample = self.sample_once(
                model,
                sample,
                t_curr,
                t_prev,
                cond,
                cfg_strength,
                kv,
                None,
                None,
                t_curr,
                False,
            )

            # Inject inverted SLAT latents at preserved (unchanged) coordinates
            if (
                i < injection_steps
                and slat_latent is not None
                and preserve_coords_mask is not None
            ):
                t_latent = t_curr
                latent_key = f"{t_latent}"

                if latent_key in slat_latent:
                    print(f"Injecting SLAT latent at {latent_key}")
                    # Find matching coordinates between sample and mask
                    # preserve_coords_mask has batch dim prepended: (N, 4) with [0, x, y, z]
                    match_1 = (
                        sample.coords.unsqueeze(1) == preserve_coords_mask.unsqueeze(0)
                    ).all(dim=-1)
                    match_2 = (
                        slat_latent[latent_key].coords.cuda().unsqueeze(1)
                        == preserve_coords_mask.unsqueeze(0)
                    ).all(dim=-1)

                    # Get indices of matching coordinates
                    idx_1 = match_1.float().argmax(0)
                    idx_2 = match_2.float().argmax(0)

                    # Only inject where we have valid matches
                    has_match_1 = match_1.any(dim=0)
                    has_match_2 = match_2.any(dim=0)
                    valid_mask = has_match_1 & has_match_2

                    if valid_mask.sum() > 0:
                        print(f"  Preserved regions: {valid_mask.sum()} coords")
                        feats = sample.feats.clone()
                        feats[idx_1[valid_mask]] = slat_latent[latent_key].feats.cuda()[
                            idx_2[valid_mask]
                        ]
                        sample = sample.replace(feats)

                    # Inject into edit regions using precomputed NN mappings
                    if edit_injection_mappings is not None:
                        latent_feats = slat_latent[latent_key].feats.cuda()
                        for mapping in edit_injection_mappings:
                            sample_indices = mapping["sample_indices"]
                            nn_indices = mapping["nn_indices"]
                            sq_index = mapping.get("sq_index", "?")

                            if len(sample_indices) > 0:
                                feats = sample.feats.clone()
                                feats[sample_indices] = latent_feats[nn_indices]
                                sample = sample.replace(feats)
                                print(
                                    f"  Edit region SQ{sq_index}: {len(sample_indices)} coords"
                                )

                    # Inject into added/deleted regions (abstraction injection)
                    if (
                        i < abstraction_injection_steps
                        and added_deleted_coords_mask is not None
                    ):
                        # Find matching coordinates between sample and added_deleted mask
                        match_sample = (
                            sample.coords.unsqueeze(1)
                            == added_deleted_coords_mask.unsqueeze(0)
                        ).all(dim=-1)
                        match_latent = (
                            slat_latent[latent_key].coords.cuda().unsqueeze(1)
                            == added_deleted_coords_mask.unsqueeze(0)
                        ).all(dim=-1)

                        # Get indices of matching coordinates
                        idx_sample = match_sample.float().argmax(0)
                        idx_latent = match_latent.float().argmax(0)

                        # Only inject where we have valid matches in both
                        has_match_sample = match_sample.any(dim=0)
                        has_match_latent = match_latent.any(dim=0)
                        valid_mask = has_match_sample & has_match_latent

                        if valid_mask.sum() > 0:
                            print(
                                f"  Added/deleted injection: {valid_mask.sum()} coords"
                            )
                            feats = sample.feats.clone()
                            feats[idx_sample[valid_mask]] = slat_latent[
                                latent_key
                            ].feats.cuda()[idx_latent[valid_mask]]
                            sample = sample.replace(feats)

        return sample, slat_latent, kv


def run_appearance_editing(
    pipeline,
    inversion_data: Union[str, Path, Dict],
    output_dir: str,
    image_cond,
    num_steps: int = 25,
    sparse_structure_latent: torch.Tensor = None,
    render_fn=None,
    slat_latent: Dict = None,
    merging_data: Dict = None,
    appearance_injection_steps: int = 0,
    edit_dilation_steps: int = 1,
    normalization: Dict = None,
    do_edit_injection: bool = False,
    abstraction_injection_steps: int = 0,
    shade_smooth: bool = False,
) -> torch.Tensor:
    """
    Run appearance editing using image-guided conditioning.

    Args:
        pipeline: TrellisImageTo3DPipeline (image-guided pipeline)
        inversion_data: Either path to ss_latents.pt or the loaded dict
        output_dir: Directory to save outputs
        image_cond: Pre-computed image conditioning (REQUIRED)
        num_steps: Number of steps (matching inversion)
        sparse_structure_latent: Sparse structure latent tensor from inpainting
        render_fn: Optional render function for visualization
        slat_latent: Dict of inverted SLAT latents (from inversion)
        merging_data: Dict containing masks for determining injection regions
        appearance_injection_steps: Number of steps to inject appearance latents (0 = disabled)
        edit_dilation_steps: Dilation iterations for edit masks (should match structure inpainting)
        normalization: Dict with 'center' and 'scale' for reverting normalization on output mesh
        do_edit_injection: Whether to enable edit region injection (experimental, default False)
        abstraction_injection_steps: Number of steps to inject latents into added/deleted regions (default 0)

    Returns:
        Appearance edited GLB object
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load inversion data if path provided
    if isinstance(inversion_data, (str, Path)):
        inversion_data = torch.load(inversion_data, weights_only=False)

    # Get models
    flow_model = pipeline.models["slat_flow_model"]
    flow_model.forward = MethodType(slat_flow_forward, flow_model)
    for block in flow_model.blocks:
        trsfmr_obj = block
        trsfmr_obj.forward = MethodType(slat_trsfmr_forward, trsfmr_obj)
        self_attn_obj = block.self_attn
        self_attn_obj.forward = MethodType(slat_attn_forward, self_attn_obj)
        cross_attn_obj = block.cross_attn
        cross_attn_obj.forward = MethodType(slat_attn_forward, cross_attn_obj)
    sigma_min = pipeline.slat_sampler.sigma_min
    cfg_strength = 1.0

    # Get coordinates from sparse structure latent
    # Format: [batch, channel, x, y, z] -> extract [batch, x, y, z]
    # Note: sparse_structure_latent is already in Trellis orientation (transformed during inversion)
    coords_tgt = torch.argwhere(sparse_structure_latent > 0)[:, [0, 2, 3, 4]].int()

    noise = sp.SparseTensor(
        feats=torch.randn(coords_tgt.shape[0], flow_model.in_channels).to(
            pipeline.device
        ),
        coords=coords_tgt,
    )

    # Move noise to GPU
    if hasattr(noise, "cuda"):
        noise = noise.cuda()

    print("=" * 60)
    print("APPEARANCE EDITING (Image-Guided)")
    print("=" * 60)

    # Ensure pipeline is setup for inversion/inpainting
    # Import here to avoid circular import
    from prox_e.inversion import setup_pipeline_for_inversion

    setup_pipeline_for_inversion(pipeline)

    print("Using provided image conditioning")
    cond = image_cond

    # Compute preserve_coords_mask from merging_data if available
    # This mask contains coordinates OUTSIDE the edit regions (unchanged regions)
    # where we should inject inverted appearance latents
    preserve_coords_mask = None
    if (
        appearance_injection_steps > 0
        and merging_data is not None
        and slat_latent is not None
    ):
        print(
            f"Computing appearance injection mask (injection_steps={appearance_injection_steps})..."
        )
        from scipy.ndimage import binary_dilation

        # Get the masks from merging_data at voxel resolution (64x64x64)
        voxel_resolution = merging_data.get("voxel_resolution", 64)

        # Create union of all edit regions (dilated)
        edit_union = np.zeros(
            (voxel_resolution, voxel_resolution, voxel_resolution), dtype=np.float32
        )

        # Add changed_edited masks (dilated) - use mask_voxel for 64^3 resolution
        for edit_mask_data in merging_data.get("changed_edited_masks", []):
            mask = edit_mask_data.get("mask_voxel", edit_mask_data["mask"]).copy()
            if edit_dilation_steps > 0:
                mask = binary_dilation(
                    mask > 0.5, iterations=edit_dilation_steps
                ).astype(np.float32)
            edit_union = np.maximum(edit_union, mask)

        # Add changed_original mask (dilated) - use voxel resolution version
        changed_original = merging_data.get(
            "changed_original_mask_voxel",
            merging_data.get(
                "changed_original_mask", np.zeros((voxel_resolution,) * 3)
            ),
        )
        if edit_dilation_steps > 0:
            changed_original = binary_dilation(
                changed_original > 0.5, iterations=edit_dilation_steps
            ).astype(np.float32)
        edit_union = np.maximum(edit_union, changed_original)

        # Add added_deleted mask (dilated) - use voxel resolution version
        added_deleted = merging_data.get(
            "added_deleted_mask_voxel",
            merging_data.get("added_deleted_mask", np.zeros((voxel_resolution,) * 3)),
        )
        if edit_dilation_steps > 0:
            added_deleted = binary_dilation(
                added_deleted > 0.5, iterations=edit_dilation_steps
            ).astype(np.float32)
        edit_union = np.maximum(edit_union, added_deleted)

        # Preserve mask is everything OUTSIDE the edit union
        preserve_mask = (edit_union < 0.5).astype(np.float32)

        # Convert preserve mask to coordinates
        # Note: Masks are already in Trellis orientation (transformed in get_latent_merging_data)
        preserve_voxel_coords = np.argwhere(preserve_mask > 0.5)  # (N, 3)

        if len(preserve_voxel_coords) > 0:
            # Add batch dimension (0) as first column
            preserve_coords_mask = torch.cat(
                [
                    torch.zeros(len(preserve_voxel_coords), 1).int(),
                    torch.from_numpy(preserve_voxel_coords).int(),
                ],
                dim=1,
            ).cuda()

            print(
                f"  Preserve mask: {len(preserve_coords_mask)} coordinates (outside edit regions)"
            )

            # Save preserve_coords_mask for debugging/reuse
            preserve_mask_path = output_dir / "preserve_coords_mask.pt"
            torch.save(preserve_coords_mask, preserve_mask_path)
            print(f"  Saved preserve mask: {preserve_mask_path}")
        else:
            print(
                "  Warning: No coordinates to preserve (entire shape is in edit region)"
            )

    # Precompute edit region injection mappings (NN search done once, outside loop)
    # Only computed when do_edit_injection is True (experimental feature)
    edit_injection_mappings = None
    if (
        do_edit_injection
        and appearance_injection_steps > 0
        and merging_data is not None
        and slat_latent is not None
    ):
        print("[Edit Injection ENABLED] Computing edit region injection mappings...")
        from scipy.ndimage import binary_dilation

        voxel_resolution = merging_data.get("voxel_resolution", 64)
        global_center = merging_data.get("global_center")
        global_scale = merging_data.get("global_scale")

        # Get reference slat_latent coords (any timestep, coords are the same)
        first_key = list(slat_latent.keys())[0]
        latent_coords = slat_latent[
            first_key
        ].coords.cuda()  # (K, 4) with [batch, x, y, z]
        latent_coords_xyz = latent_coords[:, 1:].float()  # (K, 3)

        # Convert latent coords to world space for NN search
        # voxel → normalized: (v + 0.5) / res - 0.5
        latent_coords_normalized = (latent_coords_xyz + 0.5) / voxel_resolution - 0.5
        # normalized → world: n * scale + center
        latent_coords_world = (
            latent_coords_normalized * global_scale
            + torch.from_numpy(global_center).float().cuda()
        )

        edit_injection_mappings = []

        for edit_mask_info in merging_data.get("changed_edited_masks", []):
            mask_voxel = edit_mask_info.get("mask_voxel", edit_mask_info["mask"]).copy()
            M_transform_inv = edit_mask_info.get("M_transform_inv")
            sq_index = edit_mask_info.get("sq_index")

            if M_transform_inv is None:
                print(
                    f"  Warning: M_transform_inv not found for SQ{sq_index}, skipping"
                )
                continue

            # Dilate the mask (same as for other masks)
            if edit_dilation_steps > 0:
                mask_voxel = binary_dilation(
                    mask_voxel > 0.5, iterations=edit_dilation_steps
                ).astype(np.float32)

            # Get voxel coordinates in this edit mask
            edit_voxel_coords = np.argwhere(mask_voxel > 0.5)  # (N, 3)
            if len(edit_voxel_coords) == 0:
                continue

            # Add batch dimension for matching with sample.coords
            edit_coords_with_batch = torch.cat(
                [
                    torch.zeros(len(edit_voxel_coords), 1).int(),
                    torch.from_numpy(edit_voxel_coords).int(),
                ],
                dim=1,
            ).cuda()

            # Find which sample coords (coords_tgt) are in this edit mask
            match_sample = (
                coords_tgt.cuda().unsqueeze(1) == edit_coords_with_batch.unsqueeze(0)
            ).all(dim=-1)
            sample_in_mask = match_sample.any(dim=1)
            sample_indices = torch.where(sample_in_mask)[0]

            if len(sample_indices) == 0:
                continue

            # Get xyz coords of sample points in edit region
            coords_xyz = coords_tgt[sample_indices, 1:].float().cuda()  # (M, 3)

            # Transform: voxel → normalized → world → M_transform_inv → world → normalized → voxel
            # voxel → normalized
            coords_normalized = (coords_xyz + 0.5) / voxel_resolution - 0.5
            # normalized → world
            coords_world = (
                coords_normalized * global_scale
                + torch.from_numpy(global_center).float().cuda()
            )
            # Apply M_transform_inv (world space: edited → original)
            ones = torch.ones(coords_world.shape[0], 1, device=coords_world.device)
            coords_homo = torch.cat([coords_world, ones], dim=1)
            M_tensor = torch.from_numpy(M_transform_inv).float().cuda()
            transformed_world = (M_tensor @ coords_homo.T).T[:, :3]

            # Find nearest neighbors in latent_coords_world (vectorized)
            diff = transformed_world.unsqueeze(1) - latent_coords_world.unsqueeze(
                0
            )  # (M, K, 3)
            dist_sq = (diff**2).sum(dim=2)  # (M, K)
            nn_indices = dist_sq.argmin(dim=1)  # (M,)

            edit_injection_mappings.append(
                {
                    "sample_indices": sample_indices,
                    "nn_indices": nn_indices,
                    "sq_index": sq_index,
                }
            )
            print(
                f"  Edit mask SQ{sq_index}: {len(sample_indices)} coords → precomputed NN mapping"
            )

            # Debug: show first 5 coordinates and their NN
            num_debug = min(5, len(sample_indices))
            print(f"    First {num_debug} coordinate mappings:")
            for j in range(num_debug):
                src_voxel = coords_xyz[j].cpu().numpy()
                src_world = coords_world[j].cpu().numpy()
                tgt_world = transformed_world[j].cpu().numpy()
                nn_idx = nn_indices[j].item()
                nn_world = latent_coords_world[nn_idx].cpu().numpy()
                nn_voxel = latent_coords_xyz[nn_idx].cpu().numpy()
                dist = torch.sqrt(dist_sq[j, nn_idx]).item()
                print(
                    f"      [{j}] voxel {src_voxel} → world {src_world} → transformed {tgt_world}"
                )
                print(
                    f"          → NN idx {nn_idx}: voxel {nn_voxel}, world {nn_world}, dist={dist:.4f}"
                )

        if not edit_injection_mappings:
            edit_injection_mappings = None

    # Compute added_deleted_coords_mask for abstraction injection
    added_deleted_coords_mask = None
    if (
        abstraction_injection_steps > 0
        and appearance_injection_steps > 0
        and merging_data is not None
        and slat_latent is not None
    ):
        print(
            f"Computing added/deleted coords mask (abstraction_injection_steps={abstraction_injection_steps})..."
        )
        from scipy.ndimage import binary_dilation

        voxel_resolution = merging_data.get("voxel_resolution", 64)

        # Get added_deleted mask and dilate it
        added_deleted = merging_data.get(
            "added_deleted_mask_voxel",
            merging_data.get("added_deleted_mask", np.zeros((voxel_resolution,) * 3)),
        )
        if edit_dilation_steps > 0:
            added_deleted = binary_dilation(
                added_deleted > 0.5, iterations=edit_dilation_steps
            ).astype(np.float32)

        # Convert to coordinates
        added_deleted_voxel_coords = np.argwhere(added_deleted > 0.5)  # (N, 3)

        if len(added_deleted_voxel_coords) > 0:
            # Add batch dimension (0) as first column
            added_deleted_coords_mask = torch.cat(
                [
                    torch.zeros(len(added_deleted_voxel_coords), 1).int(),
                    torch.from_numpy(added_deleted_voxel_coords).int(),
                ],
                dim=1,
            ).cuda()

            print(f"  Added/deleted mask: {len(added_deleted_coords_mask)} coordinates")
        else:
            print("  No added/deleted coordinates found")

    # Create sampler and run appearance editing
    print("Running appearance editing...")
    sampler = AppearanceEditingSampler(sigma_min, steps=num_steps)
    slat, output_slat_latent, _ = sampler.sample_appearance_editing(
        flow_model,
        noise,
        cond,
        cfg_strength,
        slat_latent=slat_latent,
        preserve_coords_mask=preserve_coords_mask,
        injection_steps=appearance_injection_steps,
        edit_injection_mappings=edit_injection_mappings,
        added_deleted_coords_mask=added_deleted_coords_mask,
        abstraction_injection_steps=abstraction_injection_steps,
    )

    # normalize the slat
    std = torch.tensor(pipeline.slat_normalization["std"], device=pipeline.device)[None]
    mean = torch.tensor(pipeline.slat_normalization["mean"], device=pipeline.device)[
        None
    ]
    slat = slat * std + mean

    # Save SLAT latents of the appearance edited output
    output_slat_latents_path = output_dir / "output_slat_latents.pt"
    torch.save(
        {
            "slat_latent": output_slat_latent,
            "coords": slat.coords.cpu() if hasattr(slat.coords, "cpu") else slat.coords,
        },
        output_slat_latents_path,
    )
    print(f"Saved appearance edited SLAT latents: {output_slat_latents_path}")

    # Decode to mesh + gaussians
    assets_tgt = pipeline.decode_slat(slat, ["gaussian", "mesh"])
    torch.set_grad_enabled(True)
    glb_tgt = postprocessing_utils.to_glb(
        assets_tgt["gaussian"][0],
        assets_tgt["mesh"][0],
        simplify=0.95,
        texture_size=1024,
    )

    # Export to GLB
    glb_output_path = output_dir / "output.glb"
    glb_tgt.export(glb_output_path)

    # Apply denormalization to restore original scale and position
    center = normalization["center"]
    scale = normalization["scale"]
    apply_denormalization_to_glb(str(glb_output_path), center=center, scale=scale)

    # Render the generated mesh using the same method as SLAT inversion verification
    # Use render_obj_with_blender with rotation to match SLAT coordinate system
    render_output_path = output_dir / "output.png"
    print("\nRendering generated mesh...")
    try:
        from prox_e.utils import render_obj_with_blender

        trellis_rotation = (-90, 0, 0)  # Aligns Trellis output with standard view
        render_obj_with_blender(
            str(glb_output_path),
            str(render_output_path),
            rotation=trellis_rotation,
            shade_smooth=shade_smooth,
        )
        print(f"  - Render: {render_output_path}")
    except Exception as e:
        print(f"  Warning: Failed to render: {e}")

    print("Appearance editing complete!")
    return glb_tgt
