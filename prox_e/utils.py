"""
Blender utility functions for rendering and boolean operations.
"""
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch

# Default Blender executable path. Prefer user configuration or PATH; keep the
# lab path as a final fallback for existing internal runs.
DEFAULT_BLENDER_PATH = (
    os.environ.get("BLENDER_PATH")
    or shutil.which("blender")
    or "/nfs/usr/esella/blender-3.4.1-linux-x64/blender"
)

# Applied in Blender when rendering Trellis decode exports (e.g. original_slat.png, output.png).
TRELLIS_DECODE_GLB_BLENDER_ROTATION_DEG = (-90, 0, 0)


def format_time(seconds: float) -> str:
    """Format seconds into human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.1f}s"


def load_parse_result(file_path: Path, category: str) -> dict:
    """
    Load structural/appearance descriptions from instruction_parse_result.txt.
    
    Returns dict with 'structural_description' and 'appearance_description',
    or None if file cannot be parsed.
    """
    default = f"a {category}"
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        structural = default
        appearance = default
        for line in lines:
            line = line.strip()
            if line.startswith("Structural Description:"):
                structural = line.replace("Structural Description:", "").strip() or default
            elif line.startswith("Appearance Description:"):
                appearance = line.replace("Appearance Description:", "").strip() or default
        return {'structural_description': structural, 'appearance_description': appearance}
    except Exception:
        return None


def transform_voxels_to_trellis(voxels: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Transform voxels to Trellis coordinate orientation.
    
    The Trellis model expects coordinates in [x, z, y_inverted] format where:
    - y and z axes are swapped
    - the new y axis is inverted
    
    For a 3D grid, this means:
    - Swap axes 1 and 2 (y and z)
    - Flip along axis 1 (the new y axis)
    
    Args:
        voxels: Either numpy array of shape (res, res, res) or (1, 1, res, res, res),
                or torch tensor of shape (1, 1, res, res, res)
    
    Returns:
        Transformed voxels in the same format as input
    """
    if isinstance(voxels, torch.Tensor):
        # Shape: (1, 1, x, y, z) -> (1, 1, x, z, y_inv)
        if voxels.dim() == 5:
            # Swap y and z (axes 3 and 4)
            transformed = voxels.permute(0, 1, 2, 4, 3)
            # Flip the new y axis (axis 3)
            transformed = torch.flip(transformed, dims=[3])
            return transformed.contiguous()
        elif voxels.dim() == 3:
            # Shape: (x, y, z) -> (x, z, y_inv)
            transformed = voxels.permute(0, 2, 1)
            transformed = torch.flip(transformed, dims=[1])
            return transformed.contiguous()
        else:
            raise ValueError(f"Unsupported tensor shape: {voxels.shape}")
    else:
        # NumPy array
        if voxels.ndim == 3:
            # Shape: (x, y, z) -> (x, z, y_inv)
            transformed = np.swapaxes(voxels, 1, 2)
            transformed = np.flip(transformed, axis=1)
            return np.ascontiguousarray(transformed)
        elif voxels.ndim == 5:
            # Shape: (1, 1, x, y, z) -> (1, 1, x, z, y_inv)
            transformed = np.swapaxes(voxels, 3, 4)
            transformed = np.flip(transformed, axis=3)
            return np.ascontiguousarray(transformed)
        else:
            raise ValueError(f"Unsupported array shape: {voxels.shape}")


def render_obj_with_blender(
        obj_path,
        output_path,
        res_x=512,
        res_y=512,
        dist=1.5,
        azim=70.0,
        elev=20.0,
        fov=45.0,
        light_energy=3,
        transparent=True,
        blender_path=None,
        rotation=None,
        rotation_matrix=None,
        force_glb_vertex_color=False,
        invisible_ground=False,
        use_sun_light=True,
        shade_smooth=True,
        cycles_gpu: bool = True,
    ):
    """Render a single PNG; delegates to :func:`render_obj_with_blender_sequence`."""
    return render_obj_with_blender_sequence(
        obj_path,
        [float(azim)],
        [os.path.abspath(str(output_path))],
        res_x=res_x,
        res_y=res_y,
        dist=dist,
        elev=elev,
        fov=fov,
        light_energy=light_energy,
        transparent=transparent,
        blender_path=blender_path,
        rotation=rotation,
        rotation_matrix=rotation_matrix,
        force_glb_vertex_color=force_glb_vertex_color,
        invisible_ground=invisible_ground,
        use_sun_light=use_sun_light,
        shade_smooth=shade_smooth,
        cycles_gpu=cycles_gpu,
    )


def render_obj_with_blender_sequence(
        obj_path,
        azim_degrees,
        output_paths,
        res_x=512,
        res_y=512,
        dist=1.5,
        elev=20.0,
        fov=45.0,
        light_energy=3,
        transparent=True,
        blender_path=None,
        rotation=None,
        rotation_matrix=None,
        force_glb_vertex_color=False,
        invisible_ground=False,
        use_sun_light=True,
        shade_smooth=True,
        cycles_gpu: bool = True,
    ):
    """
    Render one or more PNGs in a **single** Blender process: import mesh and build the scene
    once, then change camera azimuth and filepath per frame. Pixel output matches calling
    :func:`render_obj_with_blender` once per frame.

    Args:
        obj_path: Path to the OBJ, GLB, or PLY file
        azim_degrees: List of camera azimuth angles in degrees (same length as ``output_paths``)
        output_paths: List of output PNG paths (absolute paths are recommended)
        (remaining args match :func:`render_obj_with_blender`)

    Returns:
        subprocess.CompletedProcess result
    """
    if blender_path is None:
        blender_path = DEFAULT_BLENDER_PATH

    azim_degrees = [float(x) for x in azim_degrees]
    output_paths = [os.path.abspath(str(p)) for p in output_paths]
    if len(azim_degrees) != len(output_paths):
        raise ValueError("azim_degrees and output_paths must have the same length")
    if len(azim_degrees) < 1:
        raise ValueError("need at least one frame")
    for p in output_paths:
        out_dir = os.path.dirname(p)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    batch_json_f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".batch.json", delete=False, encoding="utf-8"
    )
    batch_json_path = batch_json_f.name
    try:
        json.dump({"azims": azim_degrees, "outputs": output_paths}, batch_json_f)
    finally:
        batch_json_f.close()

    batch_json_path_repr = repr(batch_json_path)

    if invisible_ground and cycles_gpu:
        _gpu_src = textwrap.dedent(
            """\
            def _tredit_pick_cycles_device():
                try:
                    cycles_prefs = bpy.context.preferences.addons.get("cycles")
                    if not cycles_prefs:
                        return "CPU"
                    prefs = cycles_prefs.preferences
                    for compute_type in ("CUDA", "OPTIX", "HIP", "METAL"):
                        try:
                            prefs.compute_device_type = compute_type
                            prefs.get_devices()
                            for device in prefs.devices:
                                if device.type != "CPU":
                                    device.use = True
                            if any(d.use and d.type != "CPU" for d in prefs.devices):
                                print(f"[tredit] Cycles GPU: {compute_type}")
                                return "GPU"
                        except Exception:
                            continue
                except Exception as e:
                    print(f"[tredit] Cycles GPU setup failed: {e}")
                return "CPU"

            scene.cycles.device = _tredit_pick_cycles_device()
            print(f"[tredit] Cycles device: {scene.cycles.device}")
            """
        ).strip()
        _cycles_device_block = textwrap.indent(_gpu_src, "        ") + "\n"
    elif invisible_ground:
        _cycles_device_block = (
            '        scene.cycles.device = "CPU"\n'
            '        print("[tredit] Cycles device: CPU (forced)")\n'
        )
    else:
        _cycles_device_block = ""

    if rotation_matrix is not None:
        R = np.asarray(rotation_matrix, dtype=np.float64)
        if R.shape == (3, 3):
            R4 = np.eye(4, dtype=np.float64)
            R4[:3, :3] = R
        elif R.shape == (4, 4):
            R4 = R
        else:
            raise ValueError("rotation_matrix must be 3x3 or 4x4")
        row_strs = []
        for i in range(4):
            row_strs.append("(" + ", ".join(f"{x:.16g}" for x in R4[i]) + ")")
        matrix_rows = ",\n        ".join(row_strs)
        apply_vertex_rotation = f"""
    # Apply rotation matrix (mesh may already encode a dataset orientation M)
    import bmesh
    from mathutils import Matrix

    rot_matrix = Matrix((
        {matrix_rows},
    ))
    for obj in mesh_objects:
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.transform(bm, matrix=rot_matrix, verts=bm.verts)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
"""
    elif rotation is not None:
        apply_vertex_rotation = f"""
    # Apply rotation if specified (useful for coordinate system alignment)
    import bmesh
    from mathutils import Matrix

    rx, ry, rz = {rotation}
    rot_matrix = Matrix.Rotation(math.radians(rx), 4, 'X') @ Matrix.Rotation(math.radians(ry), 4, 'Y') @ Matrix.Rotation(math.radians(rz), 4, 'Z')

    for obj in mesh_objects:
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.transform(bm, matrix=rot_matrix, verts=bm.verts)
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()
"""
    else:
        apply_vertex_rotation = ""

    force_glb_vcol = bool(force_glb_vertex_color)

    blender_script = textwrap.dedent(f"""
    import bpy
    import json
    import math
    from mathutils import Vector

    with open({batch_json_path_repr}) as _bf:
        _batch = json.load(_bf)
    _azims = _batch["azims"]
    _outputs = _batch["outputs"]

    # --- helper: vertex-color material (based on your working code) ---
    def setMat_vertex_color(obj, vcol_name=None, roughness: float = 0.3):
        if obj.type != 'MESH':
            return

        mesh_obj = obj  # just to mirror your previous naming

        # create and assign material
        mat = bpy.data.materials.new('MeshMaterial')
        # replace existing materials ONLY in the vcolor case
        mesh_obj.data.materials.clear()
        mesh_obj.data.materials.append(mat)
        mesh_obj.active_material = mat

        mat.use_nodes = True
        tree = mat.node_tree

        principled = tree.nodes.get('Principled BSDF')
        if principled is None:
            return

        # add vertex color node and (optionally) set layer name
        vcolor = tree.nodes.new('ShaderNodeVertexColor')
        if vcol_name is not None:
            try:
                vcolor.layer_name = vcol_name
            except Exception:
                pass  # older Blender may not need this / may fail silently

        tree.links.new(vcolor.outputs['Color'], principled.inputs['Base Color'])

        # BSDF tweaks from your script
        principled.inputs['Roughness'].default_value = roughness
        princ_inputs = principled.inputs
        if 'Sheen Tint' in princ_inputs:
            princ_inputs['Sheen Tint'].default_value = 0.0
        if 'Specular IOR Level' in princ_inputs:
            princ_inputs['Specular IOR Level'].default_value = 0.5
        if 'IOR' in princ_inputs:
            princ_inputs['IOR'].default_value = 1.45
        if 'Transmission Weight' in princ_inputs:
            princ_inputs['Transmission Weight'].default_value = 0.0
        if 'Coat Roughness' in princ_inputs:
            princ_inputs['Coat Roughness'].default_value = 0.0

    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Detect file type and import accordingly
    mesh_path = r\"\"\"{obj_path}\"\"\"
    file_ext = mesh_path.lower().split('.')[-1]
    imported = False
    
    if file_ext in ('glb', 'gltf'):
        # Import GLB/GLTF file
        try:
            bpy.ops.import_scene.gltf(filepath=mesh_path)
            imported = True
        except Exception as e:
            print(f"GLB import failed: {{e}}")
    elif file_ext == 'ply':
        # Import PLY file (supports vertex colors)
        # Use forward_axis and up_axis to match OBJ coordinate system
        try:
            bpy.ops.wm.ply_import(filepath=mesh_path, forward_axis='NEGATIVE_Z', up_axis='Y')
            imported = True
        except TypeError:
            # Try without axis options
            try:
                bpy.ops.wm.ply_import(filepath=mesh_path)
                imported = True
            except Exception:
                pass
        except AttributeError:
            # Older Blender versions
            try:
                bpy.ops.import_mesh.ply(filepath=mesh_path)
                imported = True
            except Exception as e:
                print(f"PLY import failed: {{e}}")
    else:
        # --- Import OBJ, trying to ensure vertex colors come in ---
        # Blender 4.x operator
        if hasattr(bpy.ops.wm, "obj_import"):
            try:
                bpy.ops.wm.obj_import(
                    filepath=mesh_path,
                    import_colors='SRGB'  # ensure vertex colors are imported
                )
                imported = True
            except TypeError:
                # fallback if this Blender build has different params
                bpy.ops.wm.obj_import(filepath=mesh_path)
                imported = True

        # Older Blender (2.8x–3.x) operator
        if not imported and hasattr(bpy.ops.import_scene, "obj"):
            try:
                bpy.ops.import_scene.obj(
                    filepath=mesh_path
                )
                imported = True
            except Exception:
                pass

    # Get all mesh objects (for GLB, selected_objects may be empty after import)
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
{apply_vertex_rotation}
    
    # Center mesh geometry at origin (move vertices, not just object location)
    if mesh_objects:
        # Compute combined bounding box center
        min_coord = Vector((float('inf'), float('inf'), float('inf')))
        max_coord = Vector((float('-inf'), float('-inf'), float('-inf')))
        
        for obj in mesh_objects:
            for vert in obj.data.vertices:
                world_co = obj.matrix_world @ vert.co
                min_coord.x = min(min_coord.x, world_co.x)
                min_coord.y = min(min_coord.y, world_co.y)
                min_coord.z = min(min_coord.z, world_co.z)
                max_coord.x = max(max_coord.x, world_co.x)
                max_coord.y = max(max_coord.y, world_co.y)
                max_coord.z = max(max_coord.z, world_co.z)
        
        center = (min_coord + max_coord) / 2
        
        # Translate vertices to center at origin
        import bmesh
        for obj in mesh_objects:
            mesh = obj.data
            bm = bmesh.new()
            bm.from_mesh(mesh)
            for vert in bm.verts:
                vert.co = vert.co - center
            bm.to_mesh(mesh)
            bm.free()
            mesh.update()
    
    # Apply smooth shading if requested
    if {shade_smooth}:
        for obj in mesh_objects:
            if obj.type == 'MESH':
                obj.select_set(True)
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.shade_smooth()
                obj.select_set(False)
    
    # Set object location and apply vertex colors
    force_glb_vcol = {force_glb_vcol}
    for obj in mesh_objects:
        obj.location = (0, 0, 0)

        # GLB: keep default PBR import unless we need vertex-color shading (e.g. abstraction GLBs)
        if file_ext in ('glb', 'gltf'):
            if not force_glb_vcol:
                continue
            data = obj.data
            vcol_name = None
            if hasattr(data, "color_attributes") and data.color_attributes:
                layer = data.color_attributes.active or data.color_attributes[0]
                vcol_name = layer.name
            if vcol_name is None and hasattr(data, "vertex_colors") and data.vertex_colors:
                layer = data.vertex_colors.active or data.vertex_colors[0]
                vcol_name = layer.name
            if vcol_name is not None:
                setMat_vertex_color(obj, vcol_name=vcol_name)
            continue

        data = obj.data

        # detect vertex color layer name
        vcol_name = None

        # Newer API (color_attributes)
        if hasattr(data, "color_attributes") and data.color_attributes:
            # prefer active; otherwise first
            layer = data.color_attributes.active or data.color_attributes[0]
            vcol_name = layer.name

        # Older API (vertex_colors)
        if vcol_name is None and hasattr(data, "vertex_colors") and data.vertex_colors:
            layer = data.vertex_colors.active or data.vertex_colors[0]
            vcol_name = layer.name

        # Only touch materials if a vcol layer actually exists
        if vcol_name is not None:
            setMat_vertex_color(obj, vcol_name=vcol_name)

    # Camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    el = math.radians({elev})
    dist = {dist}

    def _apply_azimuth_deg(az_deg):
        az = math.radians(az_deg)
        cam_obj.location = (
            dist * math.cos(el) * math.cos(az),
            dist * math.cos(el) * math.sin(az),
            dist * math.sin(el)
        )
        direction = Vector((0, 0, 0)) - cam_obj.location
        cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    # FOV
    cam_data.lens_unit = 'FOV'
    cam_data.angle = math.radians({fov})
    
    # Camera clipping planes (avoid clipping large objects)
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0

    ## Lighting - main sun (brighter for better appearance conditioning)
    #light_data = bpy.data.lights.new("Sun", type='SUN')
    #light_object = bpy.data.objects.new("Sun", object_data=light_data)
    #bpy.context.scene.collection.objects.link(light_object)
    #light_object.location = (5, 5, 12)
    #light_data.energy = {light_energy} * 2.5  # Brighter main light
    ## Make shadows softer
    #light_data.angle = 0.1  # Larger angle = softer shadows (in radians)
    #
    ## Add fill light to reduce dark shadows (for all file types)
    #fill_light_data = bpy.data.lights.new("FillLight", type='SUN')
    #fill_light_object = bpy.data.objects.new("FillLight", object_data=fill_light_data)
    #bpy.context.scene.collection.objects.link(fill_light_object)
    #fill_light_object.location = (-5, -5, 8)
    #fill_light_data.energy = {light_energy} * 1.2  # Strong fill
    #fill_light_data.angle = 0.1  # Softer shadows
    #
    ## Add ambient world lighting
    #bpy.context.scene.world.use_nodes = True
    #world_nodes = bpy.context.scene.world.node_tree.nodes
    #bg_node = world_nodes.get('Background')
    #if bg_node:
    #    bg_node.inputs[0].default_value = (0.15, 0.15, 0.15, 1.0)  # Reduced ambient for richer colors
    #    bg_node.inputs[1].default_value = 1.0  # Strength


    # --- REPLACEMENT LIGHTING BLOCK ---
    
    # 1. Main Light: Switch from 'SUN' to 'AREA'
    # Area lights are physically correct for soft, diffuse "studio" lighting.
    light_data = bpy.data.lights.new("SoftBox", type='AREA')
    light_object = bpy.data.objects.new("SoftBox", object_data=light_data)
    bpy.context.scene.collection.objects.link(light_object)
    
    # 2. Position: Place it directly overhead
    # Moving it to Z=10 ensures the light comes from the top, creating those contact shadows under the object.
    light_object.location = (0, 0, 10)
    
    # 3. Size: The most important setting for softness
    # A larger size spreads the light out, making shadows blurrier. 
    # 5.0 to 10.0 meters is usually good for a "giant softbox" look.
    light_data.size = 8.0
    
    # 4. Energy: Adjust for unit differences
    # Sun lights use W/m² (irradiance), while Area lights use Watts. 
    # You need to multiply your input energy significantly (e.g., * 200 or * 300) to match brightness.
    light_data.energy = {light_energy} * 600

    # 5. Fill Light: (Optional) 
    # The large Area light usually provides enough wrap-around fill on its own.
    # We disable the hard 'Sun' fill light to prevent a second set of hard shadows.
    # If you still find shadows too dark, increase the World Background strength below instead.
    
    # --- END REPLACEMENT ---

    # Add ambient world lighting (Keep this part from your original script)
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    bg_node = world_nodes.get('Background')
    if bg_node:
        bg_node.inputs[0].default_value = (0.15, 0.15, 0.15, 1.0)
        # Increase strength slightly if the underside of the object is too dark
        bg_node.inputs[1].default_value = 1.0

    # Add invisible ground plane (shadow catcher)
    if {invisible_ground}:
        # Find the minimum Z coordinate of all mesh objects to position ground below them
        min_z = 0
        if mesh_objects:
            for obj in mesh_objects:
                if obj.data.vertices:
                    for v in obj.data.vertices:
                        world_co = obj.matrix_world @ v.co
                        if world_co.z < min_z:
                            min_z = world_co.z
        
        # Position ground plane slightly below the mesh (with some margin)
        ground_z = min_z - 0.01
        bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, ground_z))
        ground = bpy.context.active_object
        ground.name = "GroundPlane"
        
        # Set up shadow catcher material
        ground_mat = bpy.data.materials.new(name="ShadowCatcher")
        ground_mat.use_nodes = True
        ground.data.materials.append(ground_mat)
        
        # Enable shadow catcher in Cycles
        ground.is_shadow_catcher = True
        
        # Adjust material for better shadows
        ground_nodes = ground_mat.node_tree.nodes
        ground_links = ground_mat.node_tree.links
        
        # Clear default nodes
        for node in ground_nodes:
            ground_nodes.remove(node)
        
        # Add shader nodes for shadow catcher
        output = ground_nodes.new('ShaderNodeOutputMaterial')
        output.location = (300, 0)
        
        # For transparent background with shadows
        if {'True' if transparent else 'False'}:
            # Mix shader to blend transparent and diffuse for lighter shadows
            mix = ground_nodes.new('ShaderNodeMixShader')
            mix.location = (100, 0)
            mix.inputs['Fac'].default_value = 0.65  # Reduce shadow opacity (0=full shadow, 1=no shadow)
            
            transparent_shader = ground_nodes.new('ShaderNodeBsdfTransparent')
            transparent_shader.location = (-100, 100)
            
            diffuse = ground_nodes.new('ShaderNodeBsdfDiffuse')
            diffuse.location = (-100, -100)
            diffuse.inputs['Color'].default_value = (1, 1, 1, 1)
            
            ground_links.new(transparent_shader.outputs['BSDF'], mix.inputs[1])
            ground_links.new(diffuse.outputs['BSDF'], mix.inputs[2])
            ground_links.new(mix.outputs['Shader'], output.inputs['Surface'])
        else:
            # Simple diffuse for non-transparent
            diffuse = ground_nodes.new('ShaderNodeBsdfDiffuse')
            diffuse.location = (0, 0)
            diffuse.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1)
            ground_links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])

    # Render settings
    scene = bpy.context.scene
    
    # Use Cycles only if invisible ground is enabled (for better shadow catcher support)
    if {invisible_ground}:
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 128  # Good quality/speed balance
{_cycles_device_block}
    scene.render.resolution_x = {res_x}
    scene.render.resolution_y = {res_y}
    scene.render.resolution_percentage = 100

    # Output settings (filepath is set each iteration; matches single-frame render_obj_with_blender)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # Transparent background
    scene.render.film_transparent = {'True' if transparent else 'False'}

    for _fi in range(len(_azims)):
        _apply_azimuth_deg(_azims[_fi])
        scene.render.filepath = _outputs[_fi]
        print(f"[tredit] Render frame {{_fi + 1}}/{{len(_azims)}} -> {{_outputs[_fi]}}")
        bpy.ops.render.render(write_still=True)
    """)

    # Write temporary Blender script
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(blender_script)
        temp_script_path = tf.name

    # Execute Blender in background
    cmd = [
        blender_path,
        "--background",
        "--python", temp_script_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Cleanup
    try:
        os.remove(temp_script_path)
    except OSError:
        pass
    try:
        os.remove(batch_json_path)
    except OSError:
        pass

    return result


def run_boolean_cut(
    obj_a: str,
    obj_b: str,
    out_obj: str,
    blender_path: str = None,
):
    """
    Call Blender in background to compute A - B and export result to out_obj.
    
    Args:
        obj_a: Path to shape A OBJ file
        obj_b: Path to shape B OBJ file
        out_obj: Output path for the result OBJ
        blender_path: Path to Blender executable (default: uses DEFAULT_BLENDER_PATH)
    
    Returns:
        subprocess.CompletedProcess result
    """
    if blender_path is None:
        blender_path = DEFAULT_BLENDER_PATH

    out_obj = os.path.abspath(out_obj)
    out_dir = os.path.dirname(out_obj)
    os.makedirs(out_dir, exist_ok=True)

    blender_script = textwrap.dedent(f"""
    import bpy
    import os

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def clear_scene():
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def import_obj(path):
        path = os.path.abspath(path)
        before = set(bpy.data.objects)

        imported = False

        # Blender 4.x+ importer
        if hasattr(bpy.ops.wm, "obj_import"):
            try:
                bpy.ops.wm.obj_import(
                    filepath=path,
                    import_colors='SRGB'  # ensure vertex colors when available
                )
                imported = True
            except TypeError:
                bpy.ops.wm.obj_import(filepath=path)
                imported = True

        # Older importer
        if not imported and hasattr(bpy.ops.import_scene, "obj"):
            bpy.ops.import_scene.obj(filepath=path)
            imported = True

        after = set(bpy.data.objects)
        new_objs = [o for o in (after - before) if o.type == 'MESH']

        if not new_objs:
            raise RuntimeError(f"No mesh objects imported from {{path!r}}")

        # If multiple objects, join them into one
        if len(new_objs) == 1:
            obj = new_objs[0]
        else:
            for o in new_objs:
                o.select_set(True)
            bpy.context.view_layer.objects.active = new_objs[0]
            bpy.ops.object.join()
            obj = new_objs[0]

        # Apply transforms to be safe
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        return obj

    def object_has_vertex_colors(obj):
        if obj.type != 'MESH':
            return False
        data = obj.data
        # Newer API
        if hasattr(data, "color_attributes") and data.color_attributes:
            return True
        # Older API
        if hasattr(data, "vertex_colors") and data.vertex_colors:
            return True
        return False

    def strip_vertex_colors(obj):
        \"\"\"Remove any color attributes / vertex colors from the mesh.\"\"\"
        if obj.type != 'MESH':
            return
        data = obj.data

        # Newer API
        if hasattr(data, "color_attributes"):
            for layer in list(data.color_attributes):
                data.color_attributes.remove(layer)

        # Older API
        if hasattr(data, "vertex_colors"):
            for layer in list(data.vertex_colors):
                data.vertex_colors.remove(layer)

    def boolean_difference(obj_a, obj_b):
        # Make sure only A is active
        bpy.ops.object.select_all(action='DESELECT')
        obj_a.select_set(True)
        bpy.context.view_layer.objects.active = obj_a

        # Add boolean modifier to A
        bool_mod = obj_a.modifiers.new(name="BooleanDiff", type='BOOLEAN')
        bool_mod.operation = 'DIFFERENCE'
        bool_mod.object = obj_b
        bool_mod.solver = 'EXACT'

        # Apply modifier
        bpy.ops.object.modifier_apply(modifier=bool_mod.name)

        # Delete B
        bpy.ops.object.select_all(action='DESELECT')
        obj_b.select_set(True)
        bpy.ops.object.delete()

        return obj_a

    def export_obj(obj, filepath, export_vertex_colors):
        filepath = os.path.abspath(filepath)
        out_dir = os.path.dirname(filepath)
        os.makedirs(out_dir, exist_ok=True)

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # If we don't want vertex colors, strip any color layers first
        if not export_vertex_colors:
            strip_vertex_colors(obj)

        exported = False

        # Blender 4.x exporter
        if hasattr(bpy.ops.wm, "obj_export"):
            try:
                bpy.ops.wm.obj_export(
                    filepath=filepath,
                    export_selected_objects=True,
                    export_materials=True,
                    export_colors=export_vertex_colors,
                )
                exported = True
            except TypeError:
                # older signature without export_colors
                bpy.ops.wm.obj_export(
                    filepath=filepath,
                    export_selected_objects=True,
                    export_materials=True,
                )
                exported = True

        # Older exporter
        if not exported and hasattr(bpy.ops.export_scene, "obj"):
            bpy.ops.export_scene.obj(
                filepath=filepath,
                use_selection=True,
                use_materials=True,
            )
            exported = True

        if not exported:
            raise RuntimeError("Could not find a suitable OBJ exporter in this Blender build.")


    # -------------------------------------------------
    # Main
    # -------------------------------------------------
    clear_scene()

    obj_a_path = r\"\"\"{os.path.abspath(obj_a)}\"\"\"
    obj_b_path = r\"\"\"{os.path.abspath(obj_b)}\"\"\"
    out_obj_path = r\"\"\"{out_obj}\"\"\"

    # Import shapes
    A = import_obj(obj_a_path)
    B = import_obj(obj_b_path)

    # Check if A actually has vertex colors
    A_has_vcols = object_has_vertex_colors(A)

    # Perform boolean A - B
    result = boolean_difference(A, B)

    # Export result; only export vertex colors if A had them
    export_obj(result, out_obj_path, export_vertex_colors=A_has_vcols)
    """)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(blender_script)
        temp_script_path = tf.name

    cmd = [
        blender_path,
        "--background",
        "--python", temp_script_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    os.remove(temp_script_path)

    if result.returncode != 0:
        raise RuntimeError(f"Blender boolean A-B failed (code {result.returncode})")

    return result


def run_add_shape_c(
    cut_obj: str,
    obj_c: str,
    out_obj: str,
    blender_path: str = None,
):
    """
    Call Blender in background to take CUT = (A - B),
    add shape C (bright green) via boolean UNION, and export result to out_obj.
    
    Args:
        cut_obj: Path to the cut result OBJ file (A - B)
        obj_c: Path to shape C OBJ file
        out_obj: Output path for the result OBJ
        blender_path: Path to Blender executable (default: uses DEFAULT_BLENDER_PATH)
    
    Returns:
        subprocess.CompletedProcess result
    """
    if blender_path is None:
        blender_path = DEFAULT_BLENDER_PATH

    out_obj = os.path.abspath(out_obj)
    out_dir = os.path.dirname(out_obj)
    os.makedirs(out_dir, exist_ok=True)

    blender_script = textwrap.dedent(f"""
    import bpy
    import os

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------
    def clear_scene():
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete()

    def import_obj(path):
        path = os.path.abspath(path)
        before = set(bpy.data.objects)

        imported = False

        # Blender 4.x+ importer
        if hasattr(bpy.ops.wm, "obj_import"):
            try:
                bpy.ops.wm.obj_import(
                    filepath=path,
                    import_colors='SRGB'
                )
                imported = True
            except TypeError:
                bpy.ops.wm.obj_import(filepath=path)
                imported = True

        # Older importer
        if not imported and hasattr(bpy.ops.import_scene, "obj"):
            bpy.ops.import_scene.obj(filepath=path)
            imported = True

        after = set(bpy.data.objects)
        new_objs = [o for o in (after - before) if o.type == 'MESH']

        if not new_objs:
            raise RuntimeError(f"No mesh objects imported from {{path!r}}")

        if len(new_objs) == 1:
            obj = new_objs[0]
        else:
            for o in new_objs:
                o.select_set(True)
            bpy.context.view_layer.objects.active = new_objs[0]
            bpy.ops.object.join()
            obj = new_objs[0]

        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        return obj

    def object_has_vertex_colors(obj):
        if obj.type != 'MESH':
            return False
        data = obj.data
        if hasattr(data, "color_attributes") and data.color_attributes:
            return True
        if hasattr(data, "vertex_colors") and data.vertex_colors:
            return True
        return False

    def strip_vertex_colors(obj):
        if obj.type != 'MESH':
            return
        data = obj.data
        if hasattr(data, "color_attributes"):
            for layer in list(data.color_attributes):
                data.color_attributes.remove(layer)
        if hasattr(data, "vertex_colors"):
            for layer in list(data.vertex_colors):
                data.vertex_colors.remove(layer)

    def fill_vertex_colors(obj, vcol_name, color):
        \"\"\"Ensure obj has a vertex color layer named vcol_name, fill with color (r,g,b,a).\"\"\"
        if obj.type != 'MESH':
            return
        data = obj.data

        # Newer API
        if hasattr(data, "color_attributes"):
            if vcol_name in data.color_attributes:
                layer = data.color_attributes[vcol_name]
            else:
                # BYTE_COLOR / domain='CORNER' is typical for colors
                layer = data.color_attributes.new(
                    name=vcol_name,
                    type='BYTE_COLOR',
                    domain='CORNER'
                )
            for i in range(len(layer.data)):
                layer.data[i].color = color
            return

        # Older API
        if hasattr(data, "vertex_colors"):
            if vcol_name in data.vertex_colors:
                layer = data.vertex_colors[vcol_name]
            else:
                layer = data.vertex_colors.new(name=vcol_name)
            for loop_col in layer.data:
                loop_col.color = color

    def set_constant_green_material(obj):
        \"\"\"Assign a constant bright green material (no vertex colors).\"\"\"
        if obj.type != 'MESH':
            return

        mesh = obj.data
        mat = bpy.data.materials.new(name="ConstGreen")
        mesh.materials.clear()
        mesh.materials.append(mat)

        mat.use_nodes = True
        tree = mat.node_tree
        principled = tree.nodes.get("Principled BSDF")
        if principled is None:
            return

        principled.inputs["Base Color"].default_value = (0.0, 1.0, 0.0, 1.0)
        principled.inputs["Roughness"].default_value = 0.3

    def boolean_union(obj_base, obj_c):
        bpy.ops.object.select_all(action='DESELECT')
        obj_base.select_set(True)
        bpy.context.view_layer.objects.active = obj_base

        bool_mod = obj_base.modifiers.new(name="BooleanUnion", type='BOOLEAN')
        bool_mod.operation = 'UNION'
        bool_mod.object = obj_c
        bool_mod.solver = 'EXACT'

        bpy.ops.object.modifier_apply(modifier=bool_mod.name)

        bpy.ops.object.select_all(action='DESELECT')
        obj_c.select_set(True)
        bpy.ops.object.delete()

        return obj_base

    def export_obj(obj, filepath, export_vertex_colors):
        filepath = os.path.abspath(filepath)
        out_dir = os.path.dirname(filepath)
        os.makedirs(out_dir, exist_ok=True)

        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        if not export_vertex_colors:
            strip_vertex_colors(obj)

        exported = False

        if hasattr(bpy.ops.wm, "obj_export"):
            try:
                bpy.ops.wm.obj_export(
                    filepath=filepath,
                    export_selected_objects=True,
                    export_materials=True,
                    export_colors=export_vertex_colors,
                )
                exported = True
            except TypeError:
                bpy.ops.wm.obj_export(
                    filepath=filepath,
                    export_selected_objects=True,
                    export_materials=True,
                )
                exported = True

        if not exported and hasattr(bpy.ops.export_scene, "obj"):
            bpy.ops.export_scene.obj(
                filepath=filepath,
                use_selection=True,
                use_materials=True,
            )
            exported = True

        if not exported:
            raise RuntimeError("Could not find a suitable OBJ exporter in this Blender build.")


    # -------------------------------------------------
    # Main
    # -------------------------------------------------
    clear_scene()

    cut_path = r\"\"\"{os.path.abspath(cut_obj)}\"\"\"
    obj_c_path = r\"\"\"{os.path.abspath(obj_c)}\"\"\"
    out_obj_path = r\"\"\"{out_obj}\"\"\"

    base = import_obj(cut_path)
    C = import_obj(obj_c_path)

    base_has_vcols = object_has_vertex_colors(base)

    if base_has_vcols:
        # find base's vertex color layer name
        data = base.data
        vcol_name = None
        if hasattr(data, "color_attributes") and data.color_attributes:
            layer = data.color_attributes.active or data.color_attributes[0]
            vcol_name = layer.name
        elif hasattr(data, "vertex_colors") and data.vertex_colors:
            layer = data.vertex_colors.active or data.vertex_colors[0]
            vcol_name = layer.name

        if vcol_name is None:
            base_has_vcols = False
        else:
            # give C a matching vcol layer filled bright green
            fill_vertex_colors(C, vcol_name, (0.0, 1.0, 0.0, 1.0))
    else:
        # base has no vcols -> keep materials/MTL for base,
        # and use a green material for C
        strip_vertex_colors(C)
        set_constant_green_material(C)

    result = boolean_union(base, C)

    export_obj(result, out_obj_path, export_vertex_colors=base_has_vcols)
    """)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(blender_script)
        temp_script_path = tf.name

    cmd = [
        blender_path,
        "--background",
        "--python", temp_script_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    os.remove(temp_script_path)

    if result.returncode != 0:
        raise RuntimeError(f"Blender union with C failed (code {result.returncode})")

    return result


def generate_summary(output_folder: Path, voxel_dist: float = 1.8,
                     args=None, timings: dict = None, vlm_iterations: int = 0,
                     vlm_token_usage: dict = None, instruction_parse_data: dict = None):
    """
    Generate a summary image and text file showing the full pipeline results.
    
    Args:
        output_folder: Path to the sample output folder
        voxel_dist: Camera distance for voxel renders (default: 1.8)
        args: Parsed command line arguments
        timings: Dict of step names to execution times in seconds
        vlm_iterations: Number of VLM editing iterations
        vlm_token_usage: Dict with token usage info from VLM calls:
            - total_input_tokens: Total input tokens
            - total_output_tokens: Total output tokens
            - total_tokens: Total tokens (input + output)
            - log: List of per-call token usage
        instruction_parse_data: Dict with instruction parsing info:
            - structural_description: Parsed structural description
            - appearance_description: Parsed appearance description
            - token_usage: Dict with input_tokens and output_tokens
    """
    output_folder = Path(output_folder)

    # Import matplotlib lazily to avoid hard dependency for rendering-only scripts.
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.image as mpimg  # type: ignore
    except Exception as e:
        raise ImportError(
            "matplotlib is required for generate_summary(), but it failed to import. "
            "If you only need Blender renders / VLM slide renders, avoid calling generate_summary()."
        ) from e
    
    # Use original edit instruction from args (not the parsed/structural version)
    if args is not None and hasattr(args, 'edit_instruction'):
        edit_instruction = args.edit_instruction
    elif (output_folder / 'edit_instruction.txt').exists():
        with open(output_folder / 'edit_instruction.txt', 'r') as f:
            edit_instruction = f.read().strip()
    else:
        print("Warning: No edit instruction found, skipping summary generation")
        return
    
    # Re-render assets with consistent settings
    inversion_folder = output_folder / 'inversion'
    
    # Rotation to align Trellis-oriented outputs with standard view
    trellis_rotation = (-90, 0, 0)
    
    # Re-render original_slat and final output from GLB for consistent rendering
    original_slat_glb = output_folder / 'original_slat.glb'
    original_slat_png = output_folder / 'original_slat.png'
    if original_slat_glb.exists():
        render_obj_with_blender(str(original_slat_glb), str(original_slat_png), rotation=trellis_rotation)
    
    output_glb = output_folder / 'output.glb'
    output_png = output_folder / 'output.png'
    if output_glb.exists():
        render_obj_with_blender(str(output_glb), str(output_png), rotation=trellis_rotation)
    
    # Re-render abstraction images for consistent camera/lighting with SLAT outputs
    # Abstractions are in original mesh space (not Trellis space), so no rotation needed
    superdec_obj = output_folder / 'superdec.obj'
    original_abstraction_png = output_folder / 'original_abstraction.png'
    if superdec_obj.exists():
        render_obj_with_blender(str(superdec_obj), str(original_abstraction_png))
    
    edited_final_obj = output_folder / 'edited_final.obj'
    edited_final_render_png = output_folder / 'edited_final_render.png'
    if edited_final_obj.exists():
        render_obj_with_blender(str(edited_final_obj), str(edited_final_render_png))
    
    # Render original voxels if PLY exists and PNG doesn't
    original_voxels_ply = inversion_folder / 'original_shape_original_voxels.ply'
    original_voxels_png = output_folder / 'original_voxels.png'
    if original_voxels_ply.exists() and not original_voxels_png.exists():
        render_obj_with_blender(str(original_voxels_ply), str(original_voxels_png), rotation=trellis_rotation)
    
    # Render inpainted voxels if PLY exists and PNG doesn't
    inpainted_voxels_ply = output_folder / 'inpainted_voxels.ply'
    inpainted_voxels_png = output_folder / 'inpainted_voxels.png'
    if inpainted_voxels_ply.exists() and not inpainted_voxels_png.exists():
        render_obj_with_blender(str(inpainted_voxels_ply), str(inpainted_voxels_png), rotation=trellis_rotation)
    
    # Build list of images and titles based on what exists
    # Use original_slat.png for original and output.png for final output
    images = []
    titles = []
    
    candidates = [
        ('original_slat.png', 'Original'),
        ('original_abstraction.png', 'Abstraction'),
        ('edited_final_render.png', 'Edited Abstraction'),
        ('original_voxels.png', 'Original Voxels'),
        ('inpainted_voxels.png', 'Generated Voxels'),
        ('output.png', 'Output'),
    ]
    
    for img_name, title in candidates:
        if (output_folder / img_name).exists():
            images.append(img_name)
            titles.append(title)
    
    if len(images) < 2:
        print("Warning: Not enough images found for summary")
        return
    
    # Extract ShapeNet ID from folder name (e.g., "6a5be179ac61ab24b07017f7091028ed_shapetalk" -> "6a5be179ac61ab24b07017f7091028ed")
    shapenet_id = output_folder.name.split('_')[0]
    title = f"{shapenet_id}: {edit_instruction}"
    
    # Create figure
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=(3 * n_images, 3.5))
    if n_images == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    
    for ax, img_name, title in zip(axes, images, titles):
        img = mpimg.imread(output_folder / img_name)
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis('off')
    
    plt.subplots_adjust(wspace=0.02, left=0.01, right=0.99)
    summary_path = output_folder / 'output_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved image summary: {summary_path}")
    
    # Generate text summary
    text_summary_path = output_folder / 'run_summary.txt'
    with open(text_summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("TREDIT RUN SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Edit Instruction: {edit_instruction}\n\n")
        
        # Instruction Parse Results
        if instruction_parse_data is not None:
            f.write("--- Instruction Parsing ---\n")
            f.write(f"  Structural Description: {instruction_parse_data.get('structural_description', 'N/A')}\n")
            f.write(f"  Appearance Description: {instruction_parse_data.get('appearance_description', 'N/A')}\n")
            token_usage = instruction_parse_data.get('token_usage')
            if token_usage:
                f.write(f"  Token Usage: {token_usage.get('input_tokens', 0):,} input, {token_usage.get('output_tokens', 0):,} output\n")
            f.write("\n")
        
        # Arguments
        if args is not None:
            f.write("--- Arguments ---\n")
            for key, value in vars(args).items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")
        
        # VLM iterations
        if vlm_iterations > 0:
            f.write(f"VLM Editing Iterations: {vlm_iterations}\n\n")
        
        # VLM Token Usage
        if vlm_token_usage is not None:
            f.write("--- VLM Token Usage ---\n")
            f.write(f"  Total Input Tokens:  {vlm_token_usage.get('total_input_tokens', 0):,}\n")
            f.write(f"  Total Output Tokens: {vlm_token_usage.get('total_output_tokens', 0):,}\n")
            cached_tokens = vlm_token_usage.get('total_cached_tokens', 0)
            if cached_tokens > 0:
                f.write(f"  Cached Tokens:       {cached_tokens:,} (reduced cost)\n")
            f.write(f"  Total Tokens:        {vlm_token_usage.get('total_tokens', 0):,}\n")
            
            # Per-call breakdown
            log = vlm_token_usage.get('log', [])
            if log:
                f.write("\n  Per-call breakdown:\n")
                for entry in log:
                    cached_str = f", {entry.get('cached_tokens', 0):,} cached" if entry.get('cached_tokens', 0) > 0 else ""
                    f.write(f"    Iteration {entry['iteration']} ({entry['type']}): "
                            f"{entry['input_tokens']:,} input, {entry['output_tokens']:,} output{cached_str}\n")
            f.write("\n")
        
        # Timings
        if timings is not None:
            f.write("--- Timing ---\n")
            for step_name, step_time in timings.items():
                if step_time is not None:
                    f.write(f"  {step_name}: {format_time(step_time)}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"Saved text summary: {text_summary_path}")


# -----------------------------------------------------------------------------
# Progress, locking, and S3/bucket utilities (used by inference)
# -----------------------------------------------------------------------------


def copy_results_to_bucket(sample_output_folder: Path, s3bucket_path: str, category: str, sample_id: str):
    """
    Copy results to S3 bucket and cleanup output folder.

    Copies all non-.pt files to the S3 bucket subfolder, then deletes everything
    except output_summary.png from the output folder.

    Args:
        sample_output_folder: Path to the sample output folder
        s3bucket_path: Path to the S3 bucket base folder
        category: Category name (e.g., 'chair')
        sample_id: Sample identifier (e.g., assignment_id)
    """
    print("\n" + "="*80)
    print("COPYING TO S3 BUCKET AND CLEANUP")
    print("="*80)

    s3_subfolder = Path(s3bucket_path) / category / sample_id
    s3_subfolder.mkdir(parents=True, exist_ok=True)

    # Copy all files to s3bucket subfolder
    # For inversion subfolder, only copy .ply and .pt files
    copied_count = 0
    inversion_folder = sample_output_folder / "inversion"
    for item in sample_output_folder.rglob('*'):
        if not item.is_file():
            continue
        # For files in inversion subfolder, only copy .ply and .pt files
        if inversion_folder.exists() and inversion_folder in item.parents:
            if item.suffix not in ('.ply', '.pt'):
                continue
        # Compute relative path within sample_output_folder
        rel_path = item.relative_to(sample_output_folder)
        dest_path = s3_subfolder / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item, dest_path)
        copied_count += 1
    print(f"Copied {copied_count} files to: {s3_subfolder}")

    # Delete everything except output_summary.png from output folder
    deleted_count = 0
    for item in list(sample_output_folder.rglob('*')):
        if item.is_file() and item.name != 'output_summary.png':
            item.unlink()
            deleted_count += 1

    # Remove empty directories
    for item in sorted(sample_output_folder.rglob('*'), reverse=True):
        if item.is_dir():
            try:
                item.rmdir()  # Only removes if empty
            except OSError:
                pass  # Directory not empty, skip

    print(f"Deleted {deleted_count} files from output folder (kept output_summary.png)")


def try_lock(lock_path: Path):
    """Try to acquire a non-blocking lock. Returns lock file handle or None."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = open(lock_path, 'w')
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return f
    except (IOError, OSError):
        return None


def unlock(lock_file):
    """Release a lock."""
    if lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
        except Exception:
            pass


def get_progress_count(progress_file: Path) -> int:
    """Read the current progress count from the progress file."""
    if not progress_file.exists():
        return 0
    try:
        content = progress_file.read_text().strip()
        return int(content) if content else 0
    except (ValueError, IOError, OSError) as e:
        print(f"Warning: Could not read progress file: {e}")
        return 0


def increment_progress_count(progress_file: Path) -> int:
    """Increment and return the progress count (with locking)."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    lock_path = progress_file.parent / f".{progress_file.name}.lock"

    lock = try_lock(lock_path)
    if not lock:
        print("Warning: Could not acquire progress file lock, retrying...")
        time.sleep(0.5)
        lock = try_lock(lock_path)

    try:
        # Read current count
        count = get_progress_count(progress_file)
        count += 1

        # Write new count
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write(str(count))
            tmp_path = tmp.name
        with open(tmp_path, 'rb') as src:
            data = src.read()
        with open(progress_file, 'wb') as dst:
            dst.write(data)
        os.remove(tmp_path)
        return count
    except Exception as e:
        print(f"Warning: Could not update progress file: {e}")
        return -1
    finally:
        unlock(lock)

def count_completed_samples(output_folder: Path) -> int:
    """
    Count the number of completed samples by finding output_summary.png files.

    Searches recursively in all category subfolders.
    """
    count = 0
    if not output_folder.exists():
        return 0

    # Count all output_summary.png files in the output folder
    for summary_file in output_folder.rglob('output_summary.png'):
        count += 1

    return count


def initialize_progress_file(
    progress_file: Path, output_folder: Path, assignment_ids: list = None
) -> int:
    """
    Initialize the progress file from existing completed samples if it doesn't exist.

    If assignment_ids is provided, counts only those specific subfolder names
    that have an output_summary.png (useful for CSV-driven ablation loops).
    Otherwise falls back to a recursive glob count.

    Returns the initial count.
    """
    progress_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists and has content
    if progress_file.exists():
        try:
            content = progress_file.read_text().strip()
            if content:
                return int(content)
        except (ValueError, IOError):
            pass  # File exists but is invalid, will reinitialize

    # Count existing completed samples
    if assignment_ids is not None:
        count = 0
        for aid in assignment_ids:
            if (output_folder / aid / "output_summary.png").exists():
                count += 1
    else:
        count = count_completed_samples(output_folder)

    # Write the count
    progress_file.write_text(str(count))
    print(f"Initialized progress file with {count} existing completed samples")

    return count


def copy_slat_latents(source_folder: Path, dest_folder: Path):
    """
    Copy SLAT latents and related inversion files from source to destination folder.

    Copies the inversion subfolder contents needed to skip re-running inversion.
    """
    source_inversion = source_folder / "inversion"
    dest_inversion = dest_folder / "inversion"

    if not source_inversion.exists():
        print(f"  Warning: No inversion folder found in {source_folder}")
        return False

    dest_inversion.mkdir(parents=True, exist_ok=True)

    # Files to copy (all original_shape_* files needed for skipping inversion)
    files_to_copy = [
        "original_shape_slat_latents.pt",
        "original_shape_ss_latents.pt",
        "original_shape_voxels.pt",
    ]

    copied = 0
    for filename in files_to_copy:
        source_file = source_inversion / filename
        if source_file.exists():
            dest_file = dest_inversion / filename
            shutil.copyfile(source_file, dest_file)
            copied += 1
            print(f"  Copied: {filename}")

    # Also copy the renders folder if it exists (for feature extraction)
    source_renders = source_inversion / "original_shape_renders"
    if source_renders.exists():
        dest_renders = dest_inversion / "original_shape_renders"
        if not dest_renders.exists():
            try:
                shutil.copytree(source_renders, dest_renders)
            except shutil.Error:
                pass  # Ignore permission errors on NFS
            print(f"  Copied: original_shape_renders/")

    # Copy from_shapenet if it exists
    source_shapenet = source_folder / "from_shapenet"
    dest_shapenet = dest_folder / "from_shapenet"
    if source_shapenet.exists() and not dest_shapenet.exists():
        try:
            shutil.copytree(source_shapenet, dest_shapenet)
        except shutil.Error:
            pass  # Ignore permission errors on NFS
        print(f"  Copied: from_shapenet/")

    return copied > 0
