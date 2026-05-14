#!/usr/bin/env python3
"""
Latent merging utilities for injecting inverted latents into SuperGen sampling.

This module provides functions to blend inverted latents with sampled latents
using masks and spatial transformations, all at latent resolution (16^3).

Key data structures:
- changed_edited_masks: List of dicts, each with:
    - 'mask': 3D binary numpy array (16, 16, 16) for this superquadric
    - 'M_latent': 4x4 transformation matrix for this superquadric (in latent index space)
- changed_original_mask: Single 3D binary mask at latent resolution (16^3)
- added_deleted_mask: Single 3D binary mask at latent resolution (16^3)
"""

import torch
import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates
from typing import Dict, Optional, Tuple, List


def dilate_mask_3d(mask: np.ndarray, iterations: int) -> np.ndarray:
    """
    Dilate a 3D binary mask.
    
    Args:
        mask: 3D binary numpy array
        iterations: Number of dilation iterations
    
    Returns:
        Dilated mask
    """
    if iterations <= 0:
        return mask.copy()
    
    # Use a 3x3x3 structuring element for 3D dilation
    struct = np.ones((3, 3, 3), dtype=bool)
    return binary_dilation(mask, structure=struct, iterations=iterations).astype(mask.dtype)


def interpolate_latent_at_coords(
    latent_coords: np.ndarray,
    latent_grid: np.ndarray,
    transformation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Interpolate latent values at coordinates using a transformation.
    
    All coordinates and the transformation matrix are in latent index space (0-15).
    
    Args:
        latent_coords: Array of shape (N, 3) with latent coordinates (0-15 range)
        latent_grid: 4D numpy array of shape (C, H, W, D) containing latent values
        transformation_matrix: 4x4 transformation matrix (already in latent index space)
    
    Returns:
        Array of shape (N, C) with interpolated latent values
    """
    N = latent_coords.shape[0]
    if N == 0:
        return np.zeros((0, latent_grid.shape[0]), dtype=np.float32)
    
    # Convert to homogeneous coordinates [x, y, z, 1]
    ones = np.ones((N, 1), dtype=np.float32)
    coords_homogeneous = np.hstack((latent_coords.astype(np.float32), ones))  # (N, 4)
    
    # Apply transformation directly (already in latent space)
    transformed_coords = (transformation_matrix @ coords_homogeneous.T).T  # (N, 4)
    transformed_coords = transformed_coords[:, :3]  # (N, 3)
    
    # Interpolate from latent grid (C, H, W, D)
    num_channels = latent_grid.shape[0]
    interpolated = np.zeros((N, num_channels), dtype=np.float32)
    
    for c in range(num_channels):
        interpolated[:, c] = map_coordinates(
            latent_grid[c],
            transformed_coords.T,
            order=1,  # Trilinear interpolation
            mode='constant',
            cval=0.0
        )
    
    return interpolated


def fix_latent_axes(latent: torch.Tensor) -> torch.Tensor:
    """
    Fix axis swap between VoxHammer and SuperGen coordinate systems.
    Permute from (B, C, X, Y, Z) to (B, C, X, Z, Y).
    """
    return latent.permute(0, 1, 2, 4, 3).contiguous()


def merge_latents_with_masks(
    original_sample: torch.Tensor,
    inverted_latents: Dict[str, torch.Tensor],
    timestep: float,
    step_idx: int,
    changed_edited_masks: List[Dict],
    changed_original_mask: np.ndarray,
    added_deleted_mask: np.ndarray,
    dilation_dict: Dict[str, int],
    inversion_injection_steps: int,
) -> Tuple[torch.Tensor, bool]:
    """
    Merge inverted latents with the original sample using masks at latent resolution.
    
    Each edited superquadric has its own mask and transformation matrix.
    All masks are already at latent resolution (16^3).
    
    Args:
        original_sample: Current sample tensor of shape (B, C, H, W, D), typically (1, 8, 16, 16, 16)
        inverted_latents: Dict mapping timestep strings to latent tensors
        timestep: Current timestep value
        step_idx: Current step index
        changed_edited_masks: List of dicts, each with:
            - 'mask': 3D binary numpy array (16^3) for this superquadric
            - 'M_latent': 4x4 transformation matrix in latent index space
        changed_original_mask: 3D binary numpy array (16^3) - union of all changed original SQs
        added_deleted_mask: 3D binary numpy array (16^3) - union of all added/deleted SQs
        dilation_dict: Dict with dilation iterations for each mask type
                       e.g., {'changed_edited': 2, 'changed_original': 1, 'added_deleted': 1}
        inversion_injection_steps: Number of steps to apply injection
    
    Returns:
        Tuple of (merged_sample, was_injected)
    """
    # Check if we should inject at this step
    if step_idx >= inversion_injection_steps:
        return original_sample, False
    
    # Find the inverted latent for this timestep
    t_key = f"{timestep}"
    if t_key not in inverted_latents:
        print(f"[LatentMerging] Step {step_idx}: No inverted latent for t={timestep:.6f}")
        return original_sample, False
    
    # Load and fix axes of inverted latent
    inverted_latent = fix_latent_axes(inverted_latents[t_key].cuda())
    
    # Get latent resolution from the sample shape
    _, C, H, W, D = original_sample.shape
    latent_resolution = H  # Assuming H = W = D
    
    # Start with a clone of the inverted latent as the result
    result = inverted_latent.clone()
    
    device = original_sample.device
    dtype = original_sample.dtype
    
    # Get dilation values
    dil_changed_edited = dilation_dict.get('changed_edited', 0)
    dil_changed_original = dilation_dict.get('changed_original', 0)
    dil_added_deleted = dilation_dict.get('added_deleted', 0)
    
    # Process changed_original and added_deleted (simple union masks, already at latent resolution)
    mask_changed_original_dilated = dilate_mask_3d(changed_original_mask, dil_changed_original)
    mask_added_deleted_dilated = dilate_mask_3d(added_deleted_mask, dil_added_deleted)
    
    # Create union mask for regions that should use original sample
    # (added_deleted OR changed_original)
    use_original_mask = torch.from_numpy(
        np.logical_or(mask_added_deleted_dilated, mask_changed_original_dilated).astype(np.float32)
    ).to(device=device, dtype=dtype)
    
    # Expand mask to match latent shape (1, C, H, W, D)
    use_original_mask = use_original_mask.unsqueeze(0).unsqueeze(0).expand_as(result)
    
    # In the union of (added_deleted OR changed_original), use original sample
    result = result * (1 - use_original_mask) + original_sample * use_original_mask
    
    # Process each edited superquadric - TEMPORARY: just copy from original_sample like changed_original
    # TODO: Re-enable interpolation once debugging is complete
    if changed_edited_masks and len(changed_edited_masks) > 0:
        # Create union mask for all changed_edited regions
        changed_edited_union = np.zeros_like(changed_original_mask)
        
        for sq_data in changed_edited_masks:
            sq_mask = sq_data.get('mask')
            if sq_mask is None:
                continue
            
            # Dilate this superquadric's mask (already at latent resolution)
            sq_mask_dilated = dilate_mask_3d(sq_mask, dil_changed_edited)
            changed_edited_union = np.logical_or(changed_edited_union, sq_mask_dilated)
        
        # Convert to tensor and apply - use original_sample in these regions
        use_edited_mask = torch.from_numpy(changed_edited_union.astype(np.float32)).to(device=device, dtype=dtype)
        use_edited_mask = use_edited_mask.unsqueeze(0).unsqueeze(0).expand_as(result)
        
        # In changed_edited regions, use original_sample (temporary - should be interpolated)
        result = result * (1 - use_edited_mask) + original_sample * use_edited_mask
    
    print(f"[LatentMerging] Step {step_idx}: Merged latents at t={timestep:.6f}")
    return result, True
