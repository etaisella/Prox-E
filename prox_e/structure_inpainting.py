"""Structure inpainting using VoxHammer's latent space."""

import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Union, Dict

# Add VoxHammer submodule to path
VOXHAMMER_PATH = Path(__file__).parent / "submodules" / "voxhammer"
sys.path.insert(0, str(VOXHAMMER_PATH))

from voxhammer.edit_pipeline import InversionFlowEulerGuidanceIntervalSampler

from scipy.ndimage import binary_dilation

from prox_e.inversion import setup_pipeline_for_inversion, save_voxels_as_ply
from prox_e.inpaint_data_preparation import interpolate_grid_values
from scipy.ndimage import label


def remove_floaters(voxels: torch.Tensor, dilation_steps: int = 2, threshold: float = 0.5) -> torch.Tensor:
    """
    Remove floater voxels that are not connected to the main component.
    
    Uses dilation to determine connectivity - voxels that would be connected
    after `dilation_steps` of dilation are kept.
    
    Args:
        voxels: Voxel tensor of shape (1, 1, D, H, W)
        dilation_steps: Number of dilation steps for connectivity check
        threshold: Threshold for binarizing voxels
    
    Returns:
        Filtered voxel tensor with floaters removed
    """
    # Convert to numpy for connected component analysis
    voxels_np = voxels.squeeze().cpu().detach().numpy()
    binary_voxels = (voxels_np > threshold).astype(np.float32)
    
    # Dilate to connect nearby components
    dilated = binary_dilation(binary_voxels, iterations=dilation_steps)
    
    # Find connected components in dilated volume
    labeled, num_features = label(dilated)
    
    if num_features <= 1:
        print(f"  Floater removal: {num_features} component(s), no filtering needed")
        return voxels
    
    # Find the largest component
    component_sizes = []
    for i in range(1, num_features + 1):
        component_sizes.append((i, (labeled == i).sum()))
    component_sizes.sort(key=lambda x: x[1], reverse=True)
    main_component_label = component_sizes[0][0]
    
    # Create mask for main component (in dilated space)
    main_component_mask = (labeled == main_component_label)
    
    # Apply mask to original (undilated) voxels
    # Keep voxels that are in the main component's dilated region
    filtered_voxels = voxels_np * main_component_mask
    
    # Count removed voxels
    original_count = (binary_voxels > 0).sum()
    filtered_count = (filtered_voxels > threshold).sum()
    removed_count = original_count - filtered_count
    
    print(f"  Floater removal: {num_features} components found, kept main ({component_sizes[0][1]} voxels in dilated), removed {removed_count} floater voxels")
    
    # Convert back to tensor
    filtered_tensor = torch.from_numpy(filtered_voxels).unsqueeze(0).unsqueeze(0).to(voxels.device)
    return filtered_tensor


class InpaintingSampler(InversionFlowEulerGuidanceIntervalSampler):
    """Sampler with inpainting support."""
    
    def __init__(self, sigma_min: float, steps: int = 25):
        super().__init__(sigma_min)
        self.steps = steps
    
    @torch.no_grad()
    def sample_inpainting(self, 
                          model, 
                          noise, 
                          cond, 
                          cfg_strength, 
                          latent, 
                          abstraction_latent, 
                          merging_data, 
                          num_inpainting_steps, 
                          interpolation_steps, 
                          edit_dilation_steps=1, 
                          abstraction_injection_steps=18, 
                          transformed_mesh_latents=None, 
                          inject_original=False,
                          no_preserve_injection=False):
        """Run inpainting sampling.
        
        Args:
            transformed_mesh_latents: List of dicts with 'ss_latent' and 'sq_index' for each transformed mesh.
                                      Used to inject latents into edited regions (like abstraction for added/deleted).
        """
        steps = self.steps
        rescale_t = 3.0
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)

        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        kv = {}
        
        # Prepare edit masks with their corresponding transformed mesh latents
        dilated_edit_masks = []
        for edit_mask_data in merging_data['changed_edited_masks']:
            mask = edit_mask_data['mask'].copy()
            if edit_dilation_steps > 0:
                mask = binary_dilation(mask > 0.5, iterations=edit_dilation_steps).astype(np.float32)
            
            sq_index = edit_mask_data.get('sq_index')
            
            # Find corresponding transformed mesh latent
            transformed_latent = None
            if transformed_mesh_latents is not None:
                for tl in transformed_mesh_latents:
                    if tl.get('sq_index') == sq_index:
                        transformed_latent = tl.get('ss_latent')
                        break
            
            dilated_edit_masks.append({
                'mask': mask,
                'M_latent': edit_mask_data['M_latent'],
                'sq_index': sq_index,
                'transformed_latent': transformed_latent,
            })
        
        # Dilate added_deleted mask
        added_deleted_mask_np = merging_data['added_deleted_mask'].copy()
        if edit_dilation_steps > 0:
            added_deleted_mask_np = binary_dilation(added_deleted_mask_np > 0.5, iterations=edit_dilation_steps).astype(np.float32)
        
        # Dilate changed_original mask by the same amount as edit masks
        changed_original_mask_np = merging_data['changed_original_mask'].copy()
        if edit_dilation_steps > 0:
            changed_original_mask_np = binary_dilation(changed_original_mask_np > 0.5, iterations=edit_dilation_steps).astype(np.float32)
        
        # Get keep mask (using dilated edit masks, dilated added_deleted mask, and dilated changed_original mask)
        keep_mask = (1 - changed_original_mask_np) * (1 - added_deleted_mask_np)
        for edited_mask in dilated_edit_masks:
            keep_mask = keep_mask * (1 - edited_mask['mask'])
        keep_mask = torch.from_numpy(keep_mask).float().cuda()
        changed_original_mask = torch.from_numpy(changed_original_mask_np).float().cuda()
        added_deleted_mask = torch.from_numpy(added_deleted_mask_np).float().cuda()
        
        # Create union mask for all edited regions
        edited_union_mask_np = np.zeros_like(added_deleted_mask_np)
        for edited_mask in dilated_edit_masks:
            edited_union_mask_np = np.maximum(edited_union_mask_np, edited_mask['mask'])
        edited_union_mask = torch.from_numpy(edited_union_mask_np).float().cuda()
        
        # Get unchanged mask (undilated) for inject_original mode
        unchanged_mask = None
        if inject_original:
            # Use the unchanged superquadric mask from merging_data
            unchanged_mask_np = merging_data.get('unchanged_mask')
            if unchanged_mask_np is not None:
                unchanged_mask = torch.from_numpy(unchanged_mask_np).float().cuda()
                print(f"  [inject_original] Unchanged mask: {int(unchanged_mask.sum().item())} voxels")
            else:
                print(f"  [inject_original] Warning: unchanged_mask not found in merging_data")
        
        for i, (t_curr, t_prev) in enumerate(tqdm(t_pairs, desc="Inpainting")):
            sample = self.sample_once(model, sample, t_curr, t_prev, cond, cfg_strength, kv, 
                                       None, None, t_curr, True)
            if i < num_inpainting_steps:
                print(f"Inpainting step {i} of {num_inpainting_steps}")
                print(f"Num voxels kept {keep_mask.sum().item()} out of {sample.shape[-1] * sample.shape[-2] * sample.shape[-3]}")
                
                curr_abs_latent = abstraction_latent[f"{t_curr}"].cuda()
                if no_preserve_injection:
                    print(f"Using abstraction latent for injection")
                    curr_latent = curr_abs_latent
                else:
                    print(f"Using inverted original latent for injection")
                    curr_latent = latent[f"{t_curr}"].cuda()
                new_sample = torch.zeros_like(sample)
                
                # Keep regions: use inverted latent from original shape
                new_sample = keep_mask * curr_latent + sample * (1 - keep_mask)
                
                if i < abstraction_injection_steps:
                    # Replace values in added/deleted mask regions with abstraction latent
                    new_sample = new_sample * (1 - added_deleted_mask) + curr_abs_latent * added_deleted_mask
                    new_sample = new_sample * (1 - changed_original_mask) + curr_abs_latent * changed_original_mask
                
                if i < interpolation_steps:
                    # Inject transformed mesh latents into edited regions (like abstraction for added/deleted)
                    for edit_mask_data in dilated_edit_masks:
                        transformed_latent = edit_mask_data.get('transformed_latent')
                        mask = edit_mask_data['mask']
                        mask_tensor = torch.from_numpy(mask).float().cuda()
                        if transformed_latent is None:
                            print(f"transformed latent is None, using abstraction latent")
                            curr_transformed_latent = curr_abs_latent
                        else:
                            curr_transformed_latent = transformed_latent[f"{t_curr}"].cuda()
                        # Inject transformed mesh latent into this edit mask region
                        new_sample = new_sample * (1 - mask_tensor) + curr_transformed_latent * mask_tensor
                        #new_sample = new_sample * (1 - mask_tensor) + curr_abs_latent * mask_tensor
                        print(f"    Injected transformed latent for SQ {edit_mask_data.get('sq_index')}")
                
                # Inject original latent into unchanged regions that overlap with edit regions,
                # but only where original has higher feature norm
                if inject_original and unchanged_mask is not None and i < abstraction_injection_steps:
                    # Compute union of edited regions and added/deleted regions
                    edit_union = added_deleted_mask.clone()
                    edit_union = torch.maximum(edit_union, edited_union_mask)
                    
                    # Intersection: unchanged AND (edited OR added/deleted)
                    boundary_mask = unchanged_mask * edit_union
                    
                    # Calculate feature norms (along channel dimension, assumed to be dim 1)
                    new_sample_norm = torch.norm(new_sample, dim=1, keepdim=True)
                    curr_latent_norm = torch.norm(curr_latent, dim=1, keepdim=True)
                    
                    # Create condition: original norm > new sample norm AND within boundary mask
                    higher_norm_mask = (curr_latent_norm > new_sample_norm).float()
                    inject_mask = boundary_mask.unsqueeze(0).unsqueeze(0) * higher_norm_mask  # Add batch and channel dims
                    
                    # Inject: replace new_sample with curr_latent where mask is active
                    new_sample = new_sample * (1 - inject_mask) + curr_latent * inject_mask
                    
                    num_injected = int((inject_mask > 0).sum().item())
                    num_boundary = int((boundary_mask > 0).sum().item())
                    print(f"    Injected original latent into {num_injected} voxels (where norm higher, out of {num_boundary} boundary)")

                sample = new_sample
        return sample, latent, kv


def run_inpainting(
    pipeline,
    inversion_data: Union[str, Path, Dict],
    abstraction_inversion_data: Union[str, Path, Dict] = None,
    text_prompt: str = "",
    output_dir: str = "",
    num_steps: int = 25,
    num_inpainting_steps: int = 18,
    interpolation_steps: int = 18,
    edit_dilation_steps: int = 1,
    abstraction_injection_steps: int = 18,
    render_fn=None,
    merging_data: Dict = None,
    transformed_mesh_latents: list = None,
    normalization: Dict = None,
    inject_original: bool = False,
    no_preserve_injection: bool = False,
) -> torch.Tensor:
    """
    Run structure inpainting on inverted latents.
    
    Args:
        pipeline: TrellisTextTo3DPipeline (already setup for inversion)
        inversion_data: Either path to ss_latents.pt or the loaded dict
        abstraction_inversion_data: Either path to abstraction ss_latents.pt or the loaded dict
        text_prompt: Text prompt for conditioning
        output_dir: Directory to save outputs
        num_steps: Number of steps (matching inversion)
        num_inpainting_steps: Number of inpainting sampling steps
        interpolation_steps: Number of steps to apply interpolation
        edit_dilation_steps: Number of dilation iterations for edit masks (default: 1)
        abstraction_injection_steps: Number of steps to inject abstraction latents into added/deleted regions
        render_fn: Optional render function for visualization
        merging_data: Dict containing masks and transformations for latent merging
        normalization: Dict with 'center' and 'scale' for reverting normalization on output mesh
        transformed_mesh_latents: List of dicts with 'ss_latent' and 'sq_index' for each transformed mesh
        inject_original: If True, inject original latent into unchanged regions for all inpainting steps
        no_preserve_injection: If True, use abstraction latent instead of inverted original for preserve regions
    
    Returns:
        Inpainted voxel tensor
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load inversion data if path provided
    if isinstance(inversion_data, (str, Path)):
        inversion_data = torch.load(inversion_data, weights_only=False)
    
    noise = inversion_data['noise']
    #noise = torch.randn_like(inversion_data['noise'])
    ss_latent = inversion_data['ss_latent'] # inverted latent
    
    # Load abstraction inversion data if provided
    abstraction_ss_latent = None
    if abstraction_inversion_data is not None:
        if isinstance(abstraction_inversion_data, (str, Path)):
            abstraction_inversion_data = torch.load(abstraction_inversion_data, weights_only=False)
        abstraction_ss_latent = abstraction_inversion_data['ss_latent']
    
    # Move noise to GPU
    if hasattr(noise, 'cuda'):
        noise = noise.cuda()
    
    print("="*60)
    print("STRUCTURE INPAINTING")
    print("="*60)
    
    # Ensure pipeline is setup for inversion/inpainting
    setup_pipeline_for_inversion(pipeline)
    
    # Get conditioning
    cond = pipeline.get_cond([text_prompt])
    
    # Get models
    flow_model = pipeline.models["sparse_structure_flow_model"]
    decoder = pipeline.models["sparse_structure_decoder"]
    sigma_min = pipeline.sparse_structure_sampler.sigma_min
    cfg_strength = pipeline.sparse_structure_sampler_params["cfg_strength"]
    
    # Create sampler and run inpainting
    sampler = InpaintingSampler(sigma_min, steps=num_steps)
    
    print(f"Running inpainting (steps={num_inpainting_steps}, interp={interpolation_steps}, dilation={edit_dilation_steps}, abs_inject={abstraction_injection_steps}, inject_original={inject_original}, cfg={cfg_strength})...")
    z_s, _, _ = sampler.sample_inpainting(
        flow_model, noise, cond, cfg_strength, ss_latent, abstraction_ss_latent, merging_data, 
        num_inpainting_steps, interpolation_steps, edit_dilation_steps, abstraction_injection_steps,
        transformed_mesh_latents, inject_original, no_preserve_injection,
    )
    
    # save z_s to file
    torch.save(z_s, output_dir / "inpainted_z_s.pt")
    print(f"Saved: {output_dir / 'inpainted_z_s.pt'}")
    
    # Decode to voxels
    voxels = decoder(z_s)
    
    # Filter out floaters (voxels not connected to main component after 2-step dilation)
    voxels = remove_floaters(voxels, dilation_steps=2)
    
    # Save voxels to file
    torch.save(voxels, output_dir / "inpainted_voxels.pt")
    print(f"Saved: {output_dir / 'inpainted_voxels.pt'}")
    
    # Save output with denormalization
    output_ply = output_dir / "inpainted_voxels.ply"
    save_voxels_as_ply(voxels, str(output_ply), normalization=normalization)
    print(f"Saved: {output_ply}")
    
    # Render if function provided
    # Use rotation to align Trellis-oriented voxels with standard view
    if render_fn is not None:
        print("Rendering inpainted voxels...")
        try:
            trellis_rotation = (-90, 0, 0)
            render_fn(str(output_ply), str(output_dir / "inpainted_voxels.png"), rotation=trellis_rotation)
            print(f"Saved render: {output_dir / 'inpainted_voxels.png'}")
        except Exception as e:
            print(f"Render failed: {e}")
    
    print("Inpainting complete!")
    return voxels


def save_voxel_artifacts(voxels, output_dir, normalization=None, render_fn=None):
    """
    Save voxel artifacts (pt, ply, render) without running inpainting.

    Useful when no structural editing is needed and the original voxels
    are reused as-is.

    Args:
        voxels: Sparse structure voxel tensor
        output_dir: Directory to save outputs
        normalization: Optional dict with 'center' and 'scale' for denormalization
        render_fn: Optional render function for visualization
    """
    output_dir = Path(output_dir)

    torch.save(voxels, output_dir / "inpainted_voxels.pt")
    print(f"Saved: {output_dir / 'inpainted_voxels.pt'}")

    output_ply = output_dir / "inpainted_voxels.ply"
    save_voxels_as_ply(voxels, str(output_ply), normalization=normalization)
    print(f"Saved: {output_ply}")

    if render_fn is not None:
        print("Rendering voxels...")
        try:
            trellis_rotation = (-90, 0, 0)
            render_fn(str(output_ply), str(output_dir / "inpainted_voxels.png"), rotation=trellis_rotation)
            print(f"Saved render: {output_dir / 'inpainted_voxels.png'}")
        except Exception as e:
            print(f"Render failed: {e}")
