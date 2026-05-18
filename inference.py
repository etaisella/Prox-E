#!/usr/bin/env python3
"""
Generate superquadric abstractions from ShapeNet models using SuperDec.
"""
import os
import sys
import random
import time
import torch
import argparse
import numpy as np
from pathlib import Path

from prox_e.utils import (
    render_obj_with_blender,
    generate_summary,
    load_parse_result,
    TRELLIS_DECODE_GLB_BLENDER_ROTATION_DEG,
)
from prox_e.orientation import (
    expected_orientation_metadata,
    load_oriented_normalized_mesh,
    metadata_matches,
)
from prox_e.abstraction import (
    generate_abstraction,
    load_abstraction,
    mesh_from_abstraction,
    render_multiview,
)
from prox_e.structure_editing import prepare_vlm_input, edit_shape_with_vlm
from prox_e.inpaint_data_preparation import prepare_edit_data

# Import inversion module (uses VoxHammer's approach)
from prox_e.inversion import run_full_inversion, load_trellis_text_pipeline, compute_global_normalization
from prox_e.structure_inpainting import run_inpainting, save_voxel_artifacts
from prox_e.appearance_editing import run_appearance_editing, load_trellis_image_pipeline, preprocess_single_image, render_for_conditioning

from prox_e.analyze_instruction import analyze_shapetalk_prompt, save_parse_result
from prox_e.edit_appearance_condition import edit_appearance_image

def _shapenet_root() -> Path:
    """ShapeNet dataset root (required when loading samples without --input_mesh)."""
    root = os.environ.get("SHAPENET_ROOT")
    if not root:
        raise ValueError(
            "SHAPENET_ROOT is not set. Export SHAPENET_ROOT to load ShapeNet samples, "
            "or pass --input_mesh for a custom mesh."
        )
    return Path(root)

# Appearance injection steps when using custom appearance condition
APPEARANCE_INJECTION_STEPS_W_EDIT = 0

# ShapeNet category name to synset ID mapping
SHAPENET_CATEGORIES = {
    'airplane': '02691156',
    'bag': '02773838',
    'basket': '02801938',
    'bathtub': '02808440',
    'bed': '02818832',
    'bench': '02828884',
    'birdhouse': '02843684',
    'bookshelf': '02871439',
    'bottle': '02876657',
    'bowl': '02880940',
    'bus': '02924116',
    'cabinet': '02933112',
    'camera': '02942699',
    'can': '02946921',
    'cap': '02954340',
    'car': '02958343',
    'cellphone': '02992529',
    'chair': '03001627',
    'clock': '03046257',
    'dishwasher': '03207941',
    'display': '03211117',
    'earphone': '03261776',
    'faucet': '03325088',
    'file': '03337140',
    'guitar': '03467517',
    'helmet': '03513137',
    'jar': '03593526',
    'keyboard': '03085013',
    'knife': '03624134',
    'lamp': '03636649',
    'laptop': '03642806',
    'loudspeaker': '03691459',
    'mailbox': '03710193',
    'microphone': '03759954',
    'microwave': '03761084',
    'monitor': '03790512',
    'motorcycle': '03790512',
    'mug': '03797390',
    'piano': '03928116',
    'pillow': '03938244',
    'pistol': '03948459',
    'pot': '03991062',
    'printer': '04004475',
    'remote': '04074963',
    'rifle': '04090263',
    'rocket': '04099429',
    'skateboard': '04225987',
    'sofa': '04256520',
    'stove': '04330267',
    'table': '04379243',
    'telephone': '04401088',
    'tower': '04460130',
    'train': '04468005',
    'watercraft': '04530566',
    'washer': '04554684',
}


def list_available_categories():
    """List all available ShapeNet categories that exist in the dataset."""
    available = {}
    for name, synset in SHAPENET_CATEGORIES.items():
        category_path = _shapenet_root() / synset
        if category_path.exists():
            available[name] = synset
    return available


def get_category_samples(category_name):
    """Get all available samples for a given category."""
    category_name = category_name.lower()
    
    if category_name not in SHAPENET_CATEGORIES:
        available = list(SHAPENET_CATEGORIES.keys())
        raise ValueError(f"Category '{category_name}' not found. Available: {', '.join(sorted(available))}")
    
    synset_id = SHAPENET_CATEGORIES[category_name]
    category_path = _shapenet_root() / synset_id
    
    if not category_path.exists():
        raise ValueError(f"Category path does not exist: {category_path}")
    
    # Use os.scandir for fast directory listing
    return [Path(entry.path) for entry in os.scandir(category_path) if entry.is_dir()]


def get_random_sample(category_name):
    """Get a random sample from a given category (optimized with os.scandir)."""
    category_name = category_name.lower()
    
    if category_name not in SHAPENET_CATEGORIES:
        available = list(SHAPENET_CATEGORIES.keys())
        raise ValueError(f"Category '{category_name}' not found. Available: {', '.join(sorted(available))}")
    
    synset_id = SHAPENET_CATEGORIES[category_name]
    category_path = _shapenet_root() / synset_id
    
    if not category_path.exists():
        raise ValueError(f"Category path does not exist: {category_path}")
    
    # Use os.scandir for fast directory listing
    sample_entries = [entry for entry in os.scandir(category_path) if entry.is_dir()]
    
    if not sample_entries:
        raise ValueError(f"No samples found for category '{category_name}'")
    
    selected_entry = random.choice(sample_entries)
    selected = Path(selected_entry.path)
    
    print(f"Selected random {category_name} sample: {selected.name}")
    print(f"  (1 of {len(sample_entries)} available samples)")
    
    return selected


def get_sample_by_id(category_name, sample_id):
    """Get a specific sample by ID from a given category."""
    category_name = category_name.lower()
    
    if category_name not in SHAPENET_CATEGORIES:
        available = list(SHAPENET_CATEGORIES.keys())
        raise ValueError(f"Category '{category_name}' not found. Available: {', '.join(sorted(available))}")
    
    synset_id = SHAPENET_CATEGORIES[category_name]
    sample_path = _shapenet_root() / synset_id / sample_id
    
    if not sample_path.exists():
        raise ValueError(f"Sample not found: {sample_path}")
    
    print(f"Loading {category_name} sample: {sample_id}")
    return sample_path


def parse_args():
    parser = argparse.ArgumentParser(description='Generate superquadric abstractions from ShapeNet models')
    parser.add_argument('--output_folder', type=str, default='./outputs',
                        help='Output folder for all generated files (default: ./outputs)')
    parser.add_argument('--category', type=str, default='chair',
                        help='ShapeNet category name (default: chair)')
    parser.add_argument('--shapenet_id', type=str, default=None,
                        help='Specific ShapeNet sample ID to load (if not provided, selects random)')
    parser.add_argument('--abstraction_json_path', type=str, default=None,
                        help='Path to an existing abstraction JSON file to render (skips SuperDec inference)')
    parser.add_argument('--input_mesh', type=str, default=None,
                        help='Path to input mesh file (.glb, .obj, etc.) instead of using ShapeNet')
    parser.add_argument('--single_view', type=bool, default=False,
                        help='Render single view of the object (default: False)')
    parser.add_argument('--vlm', type=str, default='gemini', choices=['gemini', 'gpt', 'qwen'],
                        help='VLM to use for editing: gemini, gpt, or qwen (default: gemini)')
    parser.add_argument(
        '--gemini_parse_model',
        type=str,
        default='gemini-2.5-flash',
        help='Gemini model id for edit-instruction parsing (structural/appearance descriptions) (default: gemini-2.5-flash)',
    )
    parser.add_argument(
        '--gemini_model',
        type=str,
        default='gemini-3.1-pro-preview',
        help='Gemini model id for VLM shape editing when --vlm gemini (default: gemini-3.1-pro-preview)',
    )
    parser.add_argument(
        '--gpt_model',
        type=str,
        default='gpt-5.5',
        help='OpenAI model id when --vlm gpt; requires OPENAI_API_KEY (default: gpt-5.5)',
    )
    parser.add_argument(
        '--qwen_model_size',
        type=str,
        default='4B',
        help='Local Qwen3-VL checkpoint size when --vlm qwen, e.g. 4B (default: 4B)',
    )
    parser.add_argument('--edit_instruction', type=str, default='Make the seat twice as thick',
                        help='Edit instruction for the VLM')
    parser.add_argument('--max_iterations', type=int, default=2,
                        help='Maximum VLM feedback iterations (default: 3)')
    parser.add_argument('--use_gemini_cache', action='store_true',
                        help='Enable Gemini context caching for multi-iteration editing (requires special permissions)')
    parser.add_argument('--verify_inversion', action='store_true',
                        help='Verify inversion by reconstruction')
    parser.add_argument('--num_inpainting_steps', type=int, default=20,
                        help='Number of inpainting steps (default: 20)')
    parser.add_argument('--interpolation_steps', type=int, default=16,
                        help='Number of steps to apply interpolation (default: 18)')
    parser.add_argument('--edit_dilation_steps', type=int, default=1,
                        help='Number of dilation iterations for edit masks (default: 1)')
    parser.add_argument('--abstraction_injection_steps', type=int, default=14,
                        help='Number of steps to inject abstraction latents into added/deleted regions (default: 16)')
    parser.add_argument('--inject_original', action='store_true',
                        help='Inject original latent into unchanged regions for all inpainting steps')
    parser.add_argument('--appearance_injection_steps', type=int, default=22,
                        help='Number of steps to inject inverted appearance latents in unchanged regions (default: 12)')
    parser.add_argument('--appearance_abstraction_injection_steps', type=int, default=20,
                        help='Number of steps to inject latents into added/deleted regions (default: 0)')
    parser.add_argument('--num_views', type=int, default=75,
                        help='Number of views to render for VoxHammer inversion (default: 75)')
    parser.add_argument('--appearance_condition_path', type=str, default=None,
                        help='Optional path to a custom image for appearance editing conditioning (overrides default)')
    parser.add_argument('--appearance_edit_model', type=str, default='kontext', choices=['gemini', 'kontext'],
                        help='Model to use for editing appearance condition image (default: gemini)')
    parser.add_argument('--shade_smooth', action='store_true',
                        help='Use smooth shading for inversion multi-view renders and conditioning_render.png '
                             '(original.png and output.png always use smooth shading)')
    parser.add_argument('--edit3dbench', action='store_true',
                        help='Edit3D-Bench mode: do not rotate the mesh on x axis')
    parser.add_argument('--orientation_index', type=int, default=0,
                        help='Orientation index from scripts/orientation_sweep.py for custom input meshes')
    return parser.parse_args()


def editing_pipeline(args, sample_path: Path, sample_id: str, text_pipeline=None, image_pipeline=None):
    """
    Main editing pipeline that processes a ShapeNet model through the full
    abstraction, VLM editing, inversion, inpainting, and appearance editing stages.
    
    Args:
        args: Parsed command line arguments from parse_args()
        sample_path: Path to the ShapeNet sample folder OR a direct mesh file path
        sample_id: Identifier for the output folder (e.g., shapenet_id or assignment_id)
        text_pipeline: Optional pre-loaded TrellisTextTo3DPipeline (avoids reloading)
        image_pipeline: Optional pre-loaded TrellisImageTo3DPipeline (avoids reloading)
    """
    import shutil
    
    RESOLUTION = 30
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize timing dict
    timings = {}
    vlm_iterations = 0
    vlm_token_usage = None
    total_start_time = time.time()
    
    category = args.category
    step_start = time.time()
    
    sample_output_folder = output_folder / category / sample_id
    sample_output_folder.mkdir(parents=True, exist_ok=True)
    
    # Determine if input is a direct mesh file or ShapeNet folder
    if sample_path.is_file():
        # Direct mesh file input (e.g., .glb, .obj)
        # Copy original input mesh to output folder for reference
        input_mesh_folder = sample_output_folder / "input_mesh"
        input_mesh_folder.mkdir(parents=True, exist_ok=True)
        original_mesh_copy = input_mesh_folder / sample_path.name
        if not original_mesh_copy.exists():
            shutil.copyfile(sample_path, original_mesh_copy)
            print(f"Copied original input mesh to: {original_mesh_copy}")
        
        normalized_mesh_path = input_mesh_folder / "normalized.obj"
        orientation_metadata_path = input_mesh_folder / "normalized.orientation.json"
        expected_metadata = expected_orientation_metadata(
            sample_path,
            args.orientation_index,
            edit3dbench=args.edit3dbench,
        )
        can_reuse_normalized = normalized_mesh_path.exists() and (
            metadata_matches(orientation_metadata_path, expected_metadata)
            or (args.orientation_index is None and not orientation_metadata_path.exists())
        )
        if not can_reuse_normalized:
            if args.orientation_index is not None:
                print(f"Loading, orienting, and normalizing mesh: {sample_path}")
                print(f"  Orientation index: {args.orientation_index}")
            else:
                print(f"Loading and normalizing mesh: {sample_path}")
            metadata = load_oriented_normalized_mesh(
                sample_path,
                normalized_mesh_path,
                orientation_index=args.orientation_index,
                edit3dbench=args.edit3dbench,
                metadata_path=orientation_metadata_path,
            )
            norm = metadata["normalization"]
            print(f"Saved normalized mesh to: {normalized_mesh_path}")
            print(f"  Applied orientation: {metadata['orientation_label']}")
            print(f"  Original bbox: min={norm['bbox_min']}, max={norm['bbox_max']}")
            print(f"  Original center: {norm['center']}, max_dim: {norm['max_dim']:.4f}")
            print(f"  Normalized to ShapeNet scale (max_dim={norm['target_max_dim']})")
        else:
            print(f"Using existing normalized mesh: {normalized_mesh_path}")
        
        # Use normalized mesh for the pipeline
        mesh_file = normalized_mesh_path
        
        print(f"\nInput mesh: {sample_path}")
        print(f"Category: {category}")
        print(f"Normalized mesh: {mesh_file}")
        print(f"Output folder: {sample_output_folder.resolve()}")
    else:
        # ShapeNet folder structure
        mesh_file = sample_path / 'models' / 'model_normalized.obj'
        
        # Copy ShapeNet source folder to output for reference
        from_shapenet_folder = sample_output_folder / "from_shapenet"
        if not from_shapenet_folder.exists():
            try:
                shutil.copytree(sample_path, from_shapenet_folder)
            except shutil.Error:
                pass  # Ignore permission errors on NFS
            print(f"Copied ShapeNet source to: {from_shapenet_folder}")
        
        print(f"\nSample path: {sample_path}")
        synset_id = SHAPENET_CATEGORIES.get(category, 'unknown')
        print(f"Category: {category} (synset: {synset_id})")
        print(f"Mesh file: {mesh_file}")
        print(f"Mesh exists: {mesh_file.exists()}")
        print(f"Output folder: {sample_output_folder.resolve()}")

    # Check if outputs already exist
    abstraction_json_exists = (sample_output_folder / "abstraction.json").exists()
    vlm_inputs_exist = (sample_output_folder / "vlm_original.png").exists() and \
                      (sample_output_folder / "vlm_abstraction.png").exists()
    
    if abstraction_json_exists and vlm_inputs_exist:
        print("\n*** Existing outputs found - skipping rendering and abstraction generation ***")
        timings['Setup & Abstraction'] = 0  # Skipped
    else:
        # Render original mesh
        print("Rendering original mesh with Blender...")
        render_obj_with_blender(str(mesh_file), str(sample_output_folder / "original.png"), shade_smooth=True)

        # Generate abstraction (runs SuperDec, saves OBJ and JSONs)
        result = generate_abstraction(
            mesh_file=mesh_file,
            output_folder=sample_output_folder,
            category=category,
            resolution=RESOLUTION,
        )

        # Render SuperDec output
        print("Rendering SuperDec mesh...")
        render_obj_with_blender(result['obj_path'], str(sample_output_folder / "original_abstraction.png"))
        render_multiview(result['obj_path'], sample_output_folder, single_view=args.single_view)
        
        # Prepare VLM input images
        print("Preparing VLM input images...")
        prepare_vlm_input(sample_output_folder)
        
        print(f"\n=== Output files (in {sample_output_folder}) ===")
        print(f"Original render:  original.png")
        print(f"SuperDec JSON:    superdec.json")
        print(f"SuperDec OBJ:     superdec.obj")
        print(f"Abstraction JSON: abstraction.json")
        print(f"VLM inputs:       vlm_original.png, vlm_abstraction.png")
    
    timings['Setup & Abstraction'] = time.time() - step_start
    
    # Define output paths (consistent naming)
    edit_instruction_file = sample_output_folder / "edit_instruction.txt"
    edited_json_path = sample_output_folder / "edited_abstraction_final.json"
    edited_obj_path = sample_output_folder / "edited_final.obj"
    original_abstraction_path = sample_output_folder / "abstraction.json"
    
    # Check if edit instruction matches (for cache validity of all edit-dependent steps)
    instruction_matches = False
    if edit_instruction_file.exists():
        with open(edit_instruction_file, 'r') as f:
            saved_instruction = f.read().strip()
        instruction_matches = (saved_instruction == args.edit_instruction)
        if not instruction_matches:
            print(f"\n*** Instruction mismatch detected - will re-run edit-dependent steps ***")
            print(f"  Saved:   {saved_instruction[:60]}{'...' if len(saved_instruction) > 60 else ''}")
            print(f"  Current: {args.edit_instruction[:60]}{'...' if len(args.edit_instruction) > 60 else ''}")
        else:
            print(f"\n*** Instruction matches - will re-run edit-dependent steps ***")
            print(f"  Saved:   {saved_instruction[:60]}{'...' if len(saved_instruction) > 60 else ''}")
            print(f"  Current: {args.edit_instruction[:60]}{'...' if len(args.edit_instruction) > 60 else ''}")

    # === Parse Edit Instruction ===
    structural_description = f"a {category}"
    appearance_description = f"a {category}"
    instruction_parse_data = None
    
    print("\n" + "="*80)
    print("PARSING EDIT INSTRUCTION")
    print("="*80)

    parse_result_path = sample_output_folder / "instruction_parse_result.txt"
    cached = load_parse_result(parse_result_path, category) if instruction_matches else None

    if cached:
        structural_description = cached['structural_description']
        appearance_description = cached['appearance_description']
        print(f"  Loaded from cache: {parse_result_path}")
    else:
        parse_backend = "gemini"
        parse_kwargs = {"gemini_model": args.gemini_parse_model}
        if args.vlm == "gpt":
            parse_backend = "gpt"
            parse_kwargs = {"gpt_model": args.gpt_model}
        elif args.vlm == "qwen":
            parse_backend = "qwen"
            parse_kwargs = {}
        parse_result = analyze_shapetalk_prompt(
            prompt=args.edit_instruction,
            category=category,
            backend=parse_backend,
            **parse_kwargs,
        )
        structural_description = parse_result['structural_description'] or f"a {category}"
        appearance_description = parse_result['appearance_description'] or f"a {category}"
        save_parse_result(parse_result, parse_result_path)

    print(f"  Structural: {structural_description}")
    print(f"  Appearance: {appearance_description}")

    instruction_parse_data = {
        'structural_description': structural_description,
        'appearance_description': appearance_description,
        'token_usage': None if cached else parse_result.get('token_usage'),
    }
    
    # === STEP 1: VLM Editing ===
    vlm_outputs_exist = edited_json_path.exists()
    print(f"VLM outputs exist: {vlm_outputs_exist}")
    skip_vlm_editing = vlm_outputs_exist and instruction_matches
    
    # Generate OBJ from JSON if JSON exists but OBJ doesn't
    if vlm_outputs_exist and not edited_obj_path.exists():
        edited_abstraction_data = load_abstraction(edited_json_path)
        edited_mesh = mesh_from_abstraction(edited_abstraction_data, resolution=RESOLUTION)
        edited_mesh.export(str(edited_obj_path))
        print(f"Generated edited OBJ from JSON: {edited_obj_path}")
    
    no_structural_changes = structural_description == f"a {category}"

    vlm_edit_instruction = (
        structural_description if not no_structural_changes else args.edit_instruction
    )
    
    step_start = time.time()
    if no_structural_changes:
        # No structural editing needed - use original abstraction as edited
        print("\n" + "="*80)
        print("SKIPPING VLM EDITING: No structural changes needed")
        print("="*80)
        print(f"  Structural description: {structural_description}")
        
        # Copy original abstraction to edited paths
        shutil.copyfile(original_abstraction_path, edited_json_path)
        
        # Generate OBJ from original abstraction
        original_abstraction_data = load_abstraction(original_abstraction_path)
        original_mesh = mesh_from_abstraction(original_abstraction_data, resolution=RESOLUTION)
        original_mesh.export(str(edited_obj_path))
        print(f"  Copied original abstraction to: {edited_json_path}")
        print(f"  Generated edited OBJ: {edited_obj_path}")
        
        # Render for consistency
        render_obj_with_blender(str(edited_obj_path), str(sample_output_folder / "edited_final_render.png"))
        
        timings['VLM Editing'] = 0  # Skipped
    elif skip_vlm_editing:
        print("\n" + "="*80)
        print("SKIPPING VLM EDITING: Outputs already exist for this instruction")
        print("="*80)
        print(f"  Edit instruction: {vlm_edit_instruction}")
        print(f"  To re-run, delete: {edited_obj_path}")
        timings['VLM Editing'] = 0  # Skipped
    else:
        # Save the edit instruction for future cache checks
        with open(edit_instruction_file, 'w') as f:
            f.write(args.edit_instruction)
        
        # Edit shape with VLM
        print("\n" + "="*80)
        print(f"Editing shape with VLM ({args.vlm}) - max {args.max_iterations} iterations...")
        print("="*80)
        print(f"  Edit prompt: {vlm_edit_instruction}")
        edit_result = edit_shape_with_vlm(
            output_folder=sample_output_folder,
            edit_instruction=vlm_edit_instruction,
            vlm=args.vlm,
            max_iterations=args.max_iterations,
            use_gemini_cache=args.use_gemini_cache,
            gemini_model=args.gemini_model,
            gpt_model=args.gpt_model,
            model_size=args.qwen_model_size,
        )
        
        print("\n" + "="*80)
        print("VLM EDITING COMPLETE")
        print("="*80)
        print(f"Success: {edit_result['success']}")
        print(f"Iterations: {edit_result['iterations']}")
        vlm_iterations = edit_result['iterations']
        vlm_token_usage = edit_result.get('token_usage', None)
        
        # Update flag after editing
        vlm_outputs_exist = edited_json_path.exists() and edited_obj_path.exists()
        timings['VLM Editing'] = time.time() - step_start
    
    # === STEP 2: Prepare Edit Data (masks and transformations) ===
    # NOTE: This creates transformed meshes which must be included in normalization,
    # so we compute normalization AFTER this step completes.
    merging_data_path = sample_output_folder / "merging_data.pt"
    latent_masks_folder = sample_output_folder / "latent_masks"
    
    if merging_data_path.exists() and latent_masks_folder.exists() and instruction_matches:
        print("\n" + "="*80)
        print("LOADING EDIT DATA")
        print("="*80)
        merging_data = torch.load(merging_data_path, weights_only=False)
        print(f"Loaded from: {merging_data_path}")
    else:
        print("\n" + "="*80)
        print("PREPARING EDIT DATA")
        print("="*80)
        
        if not original_abstraction_path.exists():
            print("Warning: Could not find original abstraction file")
            merging_data = None
        else:
            merging_data = prepare_edit_data(
                original_json=str(original_abstraction_path),
                edited_json=str(edited_json_path),
                output_folder=str(sample_output_folder),
                render_fn=render_obj_with_blender,
                save_visualizations=True,
                run_verification=True,
                create_transformed_meshes=True,
                original_mesh_path=str(mesh_file),
                # NOTE: Not passing external normalization here - masks use superquadric-based
                # normalization. We compute unified normalization after this step to include
                # transformed meshes for inversion.
            )
            if merging_data is not None:
                torch.save(merging_data, merging_data_path)
                print(f"Saved merging data to: {merging_data_path}")

    unified_norm = {
        'center': merging_data['global_center'],
        'scale': merging_data['global_scale']
    }
    print(f"\n[Normalization] Using normalization from prepare_edit_data:")
    print(f"  Center: {unified_norm['center']}")
    print(f"  Scale: {unified_norm['scale']}")

    # === STEP 3: Inversion ===
    inversion_prompt = f"a {category}"
    inpainting_prompt = structural_description
    inversion_latents_path = sample_output_folder / "inversion" / "original_shape_ss_latents.pt"
    abs_inversion_latents_path = sample_output_folder / "inversion" / "abstraction_ss_latents.pt"
    slat_latents_path = sample_output_folder / "inversion" / "original_shape_slat_latents.pt"
    normalization_path = sample_output_folder / "inversion" / "normalization.pt"
    trellis_pipeline = text_pipeline
    original_inversion_data = None
    normalization_data = None
    
    # Check for SLAT latents (created last) to ensure inversion fully completed
    # Also require instruction match since abstraction inversion depends on the edit
    inversion_exists = inversion_latents_path.exists() and abs_inversion_latents_path.exists() and slat_latents_path.exists()
    if inversion_exists and instruction_matches:
        print("\n" + "="*80)
        print("SKIPPING INVERSION: Latents already exist for original shape and abstraction")
        print("="*80)
        print(f"  To re-run, delete: {inversion_latents_path} and {abs_inversion_latents_path}")
        timings['SLAT Inversion'] = 0  # Skipped
        timings['Structure Inversion'] = 0  # Skipped
        
        # Use unified normalization (already computed above)
        normalization_data = unified_norm
        print(f"  Using unified normalization for appearance editing")
    else:
        print("\n" + "="*80)
        print("RUNNING INVERSION")
        print("="*80)
        
        # Check if SLAT latents already exist - skip expensive rendering/features if so
        slat_latents_exist = slat_latents_path.exists()
        # Check if features exist independently - can skip expensive multi-view
        # rendering even if SLAT inversion needs re-doing (e.g., conditioning changed)
        features_exist = (sample_output_folder / "inversion" / "original_shape_renders" / "features.npz").exists()
        
        if slat_latents_exist:
            print(f"  SLAT latents already exist: {slat_latents_path}")
            print(f"  Skipping rendering and feature extraction")
        elif features_exist:
            print(f"  Features already exist, skipping multi-view rendering")
            print(f"  Will re-do SLAT inversion (conditioning may have changed)")
        
        inversion_result = run_full_inversion(
            original_mesh_path=str(mesh_file),
            abstraction_mesh_path=str(edited_obj_path),
            output_dir=str(sample_output_folder),
            external_normalization=unified_norm,  # Use same normalization as masks
            text_prompt=inversion_prompt,
            merging_data=merging_data,
            verify=args.verify_inversion,
            render_fn=render_obj_with_blender,
            num_steps=25,
            num_views=args.num_views,
            run_rendering=not slat_latents_exist and not features_exist,
            run_features=not slat_latents_exist,
            shade_smooth=args.shade_smooth,
            text_pipeline=trellis_pipeline,
            image_pipeline=image_pipeline,
        )
        
        trellis_pipeline = inversion_result['pipeline']
        original_inversion_data = inversion_result['original_inversion_data']
        normalization_data = unified_norm  # Use unified normalization (same as passed in)
        if inversion_result.get('image_pipeline') is not None:
            image_pipeline = inversion_result['image_pipeline']
        
        inv_timings = inversion_result.get('timings', {})
        timings['SLAT Inversion'] = inv_timings.get('slat_inversion', 0)
        timings['Structure Inversion'] = inv_timings.get('structure_inversion', 0)
    
    # === STEP 4: Run inpainting ===
    inpainted_voxels_path = sample_output_folder / "inpainted_voxels.pt"
    
    # Load transformed mesh latents for each edited superquadric
    transformed_mesh_latents = []
    if merging_data is not None and 'changed_edited_masks' in merging_data:
        for mask_data in merging_data['changed_edited_masks']:
            sq_index = mask_data.get('sq_index')
            latent_path = sample_output_folder / "inversion" / f"transformed_sq{sq_index}_ss_latents.pt"
            if latent_path.exists():
                latent_data = torch.load(latent_path, weights_only=False)
                transformed_mesh_latents.append({
                    'sq_index': sq_index,
                    'ss_latent': latent_data.get('ss_latent'),
                })
                print(f"Loaded transformed mesh latent for SQ {sq_index}")
            else:
                print(f"Warning: Transformed mesh latent not found for SQ {sq_index}: {latent_path}")
    
    step_start = time.time()
    if inpainted_voxels_path.exists() and instruction_matches:
        print("\n" + "="*80)
        print("LOADING INPAINTED VOXELS")
        print("="*80)
        voxels = torch.load(inpainted_voxels_path, weights_only=False)
        print(f"Loaded inpainted voxels from: {inpainted_voxels_path}")
        timings['Structure Inpainting'] = 0  # Skipped
    elif no_structural_changes:
        # No structural editing — reuse original shape voxels directly
        print("\n" + "="*80)
        print("SKIPPING INPAINTING: No structural changes, using original voxels")
        print("="*80)
        original_voxels_path = sample_output_folder / "inversion" / "original_shape_voxels.pt"
        if original_voxels_path.exists():
            voxels = torch.load(original_voxels_path, weights_only=False)
            print(f"Loaded original voxels from: {original_voxels_path}")
        else:
            raise FileNotFoundError(f"Original voxels not found: {original_voxels_path}")
        save_voxel_artifacts(voxels, sample_output_folder, normalization=normalization_data, render_fn=render_obj_with_blender)
        timings['Structure Inpainting'] = 0  # Skipped
    else:
        if trellis_pipeline is None:
            trellis_pipeline = load_trellis_text_pipeline()
        
        voxels = run_inpainting(
            pipeline=trellis_pipeline,
            inversion_data=inversion_latents_path,
            abstraction_inversion_data=abs_inversion_latents_path,
            text_prompt=inpainting_prompt,
            output_dir=str(sample_output_folder),
            num_steps=25,
            num_inpainting_steps=args.num_inpainting_steps,
            interpolation_steps=args.interpolation_steps,
            edit_dilation_steps=args.edit_dilation_steps,
            abstraction_injection_steps=args.abstraction_injection_steps,
            render_fn=render_obj_with_blender,
            merging_data=merging_data,
            transformed_mesh_latents=transformed_mesh_latents if transformed_mesh_latents else None,
            normalization=normalization_data,
            inject_original=args.inject_original,
            no_preserve_injection=False,
        )
        torch.save(voxels, inpainted_voxels_path)
        print(f"Saved inpainted voxels to: {inpainted_voxels_path}")
        timings['Structure Inpainting'] = time.time() - step_start
    
    # === STEP 5: Appearance Editing (Image-Guided) ===
    step_start = time.time()
    
    # Ensure conditioning_render.png exists (render if missing)
    conditioning_render_path = sample_output_folder / "conditioning_render.png"
    if not conditioning_render_path.exists():
        slat_glb = sample_output_folder / "original_slat.glb"
        if slat_glb.exists():
            cond_model = str(slat_glb)
            cond_rotation = TRELLIS_DECODE_GLB_BLENDER_ROTATION_DEG
            print(
                "conditioning_render.png not found; rendering from original_slat.glb "
                "(same Blender path as original_slat.png, conditioning camera only)..."
            )
        else:
            cond_model = str(mesh_file)
            cond_rotation = None
            print(
                "conditioning_render.png not found; original_slat.glb missing, "
                "rendering from normalized mesh (no Trellis GLB rotation)..."
            )
        render_for_conditioning(
            cond_model,
            str(conditioning_render_path),
            shade_smooth=args.shade_smooth,
            rotation=cond_rotation,
        )
        print(f"Saved: {conditioning_render_path}")
    
    # Load image pipeline for appearance editing if not already loaded
    if image_pipeline is None:
        image_pipeline = load_trellis_image_pipeline()
    
    # Check if we need to edit appearance based on parsed instruction
    print(f"Appearance description: {appearance_description}")
    use_edited_appearance = (
        appearance_description != f"a {category}"
        and appearance_description != f"an {category}"
        and not args.appearance_condition_path
    )
    
    # Get image conditioning - REQUIRED for appearance editing
    if args.appearance_condition_path:
        # User-provided custom appearance condition
        custom_condition_path = Path(args.appearance_condition_path)
        if not custom_condition_path.exists():
            raise FileNotFoundError(f"Appearance condition path not found: {custom_condition_path}")
        print(f"Using custom appearance condition image: {custom_condition_path}")
        preprocessed_image = preprocess_single_image(str(custom_condition_path))
        preprocessed_image.save(sample_output_folder / "conditioning_render_preprocessed.png")
        image_cond = image_pipeline.get_cond([preprocessed_image])
    elif use_edited_appearance:
        # Edit conditioning_render.png based on appearance_description
        print("\n--- Editing Appearance Based on Parsed Instruction ---")
        conditioning_render_path = sample_output_folder / "conditioning_render.png"
        edited_condition_path = sample_output_folder / "conditioning_render_edited.png"
        
        edited_image = edit_appearance_image(
            image_path=str(conditioning_render_path),
            category=category,
            appearance_description=appearance_description,
            output_path=str(edited_condition_path),
            model=args.appearance_edit_model,
        )
        
        # Preprocess edited image (skip rembg: u2net often fails on edited RGB, leaving ~0 alpha → black after premultiply)
        preprocessed_image = preprocess_single_image(str(edited_condition_path), skip_rembg=True)
        preprocessed_image.save(sample_output_folder / "conditioning_render_preprocessed.png")
        image_cond = image_pipeline.get_cond([preprocessed_image])
    elif original_inversion_data is not None and 'image_cond' in original_inversion_data:
        print("Using image conditioning from inversion data")
        image_cond = original_inversion_data['image_cond']
    else:
        # Preprocess conditioning_render.png from the output folder
        conditioning_render_path = sample_output_folder / "conditioning_render.png"
        if not conditioning_render_path.exists():
            raise FileNotFoundError(f"No image conditioning available. conditioning_render.png not found: {conditioning_render_path}")
        print(f"Preprocessing conditioning_render.png: {conditioning_render_path}")
        preprocessed_image = preprocess_single_image(str(conditioning_render_path))
        preprocessed_image.save(sample_output_folder / "conditioning_render_preprocessed.png")
        image_cond = image_pipeline.get_cond([preprocessed_image])
    
    # Get SLAT latents from inversion data
    slat_latent = None
    if original_inversion_data is not None:
        slat_latent = original_inversion_data.get('slat_latent')
    
    # If slat_latent not in memory, try loading from file
    slat_latents_path = sample_output_folder / "inversion" / "original_shape_slat_latents.pt"
    if slat_latent is None and slat_latents_path.exists():
        print(f"Loading SLAT latents from: {slat_latents_path}")
        slat_data = torch.load(slat_latents_path, weights_only=False)
        slat_latent = slat_data.get('slat_latent')
    
    # Use reduced appearance injection steps when using edited/custom appearance condition
    use_custom_appearance = args.appearance_condition_path or use_edited_appearance
    appearance_steps = APPEARANCE_INJECTION_STEPS_W_EDIT if use_custom_appearance else args.appearance_injection_steps
    
    run_appearance_editing(
        pipeline=image_pipeline,
        inversion_data=inversion_latents_path,
        output_dir=str(sample_output_folder),
        image_cond=image_cond,
        num_steps=25,
        sparse_structure_latent=voxels,
        render_fn=render_obj_with_blender,
        slat_latent=slat_latent,
        merging_data=merging_data,
        appearance_injection_steps=appearance_steps,
        edit_dilation_steps=args.edit_dilation_steps,
        normalization=normalization_data,
        do_edit_injection=False,
        abstraction_injection_steps=args.appearance_abstraction_injection_steps,
        shade_smooth=True,
    )
    timings['Appearance Editing'] = time.time() - step_start
    
    # === FINAL STEP: Generate Summary Image ===
    print("\n" + "="*80)
    print("GENERATING SUMMARY")
    print("="*80)
    # Record total time
    timings['Total'] = time.time() - total_start_time
    
    generate_summary(
        sample_output_folder,
        args=args,
        timings=timings,
        vlm_iterations=vlm_iterations,
        vlm_token_usage=vlm_token_usage,
        instruction_parse_data=instruction_parse_data,
    )
    
    return sample_output_folder


if __name__ == "__main__":
    args = parse_args()
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.input_mesh:
        # Direct mesh file input mode, or a local ShapeNet-style sample folder.
        input_mesh_path = Path(args.input_mesh)
        if not input_mesh_path.exists():
            raise ValueError(f"Input mesh not found: {input_mesh_path}")
        sample_path = input_mesh_path
        if input_mesh_path.is_file():
            sample_id = input_mesh_path.stem  # Use filename without extension as sample_id
            print(f"Using input mesh: {input_mesh_path}")
        elif (input_mesh_path / "models" / "model_normalized.obj").exists():
            sample_id = input_mesh_path.name
            print(f"Using local ShapeNet-style sample folder: {input_mesh_path}")
        else:
            raise ValueError(
                f"Input path must be a mesh file or contain models/model_normalized.obj: {input_mesh_path}"
            )
    else:
        category = args.category
        # Load sample - either by ID or random
        if args.shapenet_id:
            sample_path = get_sample_by_id(category, args.shapenet_id)
        else:
            print("Loading random sample...")
            sample_path = get_random_sample(category)
        sample_id = sample_path.name
    
    # Check if output_summary.png already exists (skip if so)
    sample_output_folder = output_folder / args.category / sample_id
    if (sample_output_folder / 'output_summary.png').exists():
        print(f"Skipping: output_summary.png already exists at {sample_output_folder}")
        sys.exit(0)
    
    # Run the editing pipeline
    editing_pipeline(args, sample_path, sample_id)
