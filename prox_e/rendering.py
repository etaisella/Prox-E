"""
Rendering utilities for tredit.

This module contains a modified copy of VoxHammer's BpyRenderer with
flat shading fix applied for proper ShapeNet model rendering.
"""
import os
import bpy
import json
import math
import numpy as np

from mathutils import Vector
from typing import Dict, Tuple


def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    def radical_inverse(base, n):
        val = 0
        inv_base = 1.0 / base
        inv_base_n = inv_base
        while n > 0:
            digit = n % base
            val += digit * inv_base_n
            n //= base
            inv_base_n *= inv_base
        return val

    def halton_sequence(dim, n):
        return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

    def hammersley_sequence(dim, n, num_samples):
        return [n / num_samples] + halton_sequence(dim - 1, n)

    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]


def apply_flat_shading():
    """
    Apply flat shading to all mesh objects in the scene.
    
    This fixes shading artifacts that can occur with ShapeNet models.
    """
    print("[FLAT SHADING] Starting flat shading application...")
    
    # Count mesh objects
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    print(f"[FLAT SHADING] Found {len(mesh_objects)} mesh objects in scene")
    
    if len(mesh_objects) == 0:
        print("[FLAT SHADING] WARNING: No mesh objects found!")
        return
    
    # 1. Ensure we start in Object Mode
    if bpy.context.object and bpy.context.object.mode != 'OBJECT':
        print("[FLAT SHADING] Switching to Object Mode...")
        bpy.ops.object.mode_set(mode='OBJECT')
        
    # 2. Deselect all objects initially
    bpy.ops.object.select_all(action='DESELECT')

    for obj in mesh_objects:
        print(f"[FLAT SHADING] Processing mesh: {obj.name}")
        # Set the current object as the "Active" object
        bpy.context.view_layer.objects.active = obj
        # Select it
        obj.select_set(True)
        
        # Apply Flat Shading (Works in Object Mode)
        try:
            bpy.ops.object.shade_flat()
            print(f"[FLAT SHADING]   Applied flat shading to: {obj.name}")
        except Exception as e:
            print(f"[FLAT SHADING]   ERROR applying flat shading to {obj.name}: {e}")
        
        # Deselect it for the next iteration
        obj.select_set(False)
    
    print("[FLAT SHADING] Flat shading application complete.")


class BpyRenderer:
    """
    Blender renderer for multi-view rendering.
    
    Modified from VoxHammer's BpyRenderer with flat shading fix.
    """
    
    def __init__(self, resolution: int = 512, engine: str = "BLENDER_EEVEE", geo_mode: bool = False, split_normal: bool = False, shade_smooth: bool = False):
        self.resolution = resolution
        self.engine = engine
        self.geo_mode = geo_mode
        self.split_normal = split_normal
        self.shade_smooth = shade_smooth
        self.import_functions = self._setup_import_functions()

    def _apply_smooth_shading(self):
        """Apply smooth shading to all mesh objects in the scene."""
        mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
        if not mesh_objects:
            return
        if bpy.context.object and bpy.context.object.mode != 'OBJECT':
            bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        for obj in mesh_objects:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            try:
                bpy.ops.object.shade_smooth()
            except Exception as e:
                print(f"[SMOOTH SHADING] ERROR on {obj.name}: {e}")
            obj.select_set(False)
        print(f"[SMOOTH SHADING] Applied smooth shading to {len(mesh_objects)} mesh objects")

    def _setup_import_functions(self):
        import_functions = {
            "obj": bpy.ops.wm.obj_import,
            "glb": bpy.ops.import_scene.gltf,
            "gltf": bpy.ops.import_scene.gltf,
            "usd": bpy.ops.import_scene.usd,
            "fbx": bpy.ops.import_scene.fbx,
            "stl": bpy.ops.import_mesh.stl,
            "usda": bpy.ops.import_scene.usda,
            "dae": bpy.ops.wm.collada_import,
            "ply": bpy.ops.wm.ply_import,
            "abc": bpy.ops.wm.alembic_import,
            "blend": bpy.ops.wm.append,
        }
        return import_functions

    def init_render_settings(self):
        bpy.context.scene.render.engine = self.engine
        bpy.context.scene.render.resolution_x = self.resolution
        bpy.context.scene.render.resolution_y = self.resolution
        bpy.context.scene.render.resolution_percentage = 100
        bpy.context.scene.render.image_settings.file_format = "PNG"
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.film_transparent = True
        
        # Enable GPU for both CYCLES and EEVEE
        self._setup_gpu()
        
        if self.engine == "CYCLES":
            bpy.context.scene.render.engine = "CYCLES"
            bpy.context.scene.cycles.samples = 32 if not self.geo_mode else 1
            bpy.context.scene.cycles.filter_type = "BOX"
            bpy.context.scene.cycles.filter_width = 1
            bpy.context.scene.cycles.diffuse_bounces = 1
            bpy.context.scene.cycles.glossy_bounces = 1
            bpy.context.scene.cycles.transparent_max_bounces = (3 if not self.geo_mode else 0)
            bpy.context.scene.cycles.transmission_bounces = (3 if not self.geo_mode else 1)
            bpy.context.scene.cycles.use_denoising = True
            bpy.context.scene.cycles.device = "GPU"
            try:
                bpy.context.scene.cycles.device = "GPU"
                bpy.context.preferences.addons["cycles"].preferences.get_devices()
                bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
            except:
                pass
    
    def _setup_gpu(self):
        """Configure GPU rendering for Blender (works for both CYCLES and EEVEE)."""
        try:
            # Get cycles preferences (this also affects GPU detection for EEVEE in some cases)
            cycles_prefs = bpy.context.preferences.addons.get("cycles")
            if cycles_prefs:
                prefs = cycles_prefs.preferences
                # Try CUDA first, then OPTIX, then HIP (AMD), then METAL (Apple)
                for compute_type in ["CUDA", "OPTIX", "HIP", "METAL"]:
                    try:
                        prefs.compute_device_type = compute_type
                        prefs.get_devices()
                        # Enable all available GPU devices
                        for device in prefs.devices:
                            if device.type != "CPU":
                                device.use = True
                                print(f"[GPU] Enabled {device.type} device: {device.name}")
                        if any(d.use and d.type != "CPU" for d in prefs.devices):
                            print(f"[GPU] Using compute type: {compute_type}")
                            break
                    except Exception:
                        continue
        except Exception as e:
            print(f"[GPU] Could not configure GPU: {e}")

    def init_scene(self):
        for obj in bpy.data.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        for material in bpy.data.materials:
            bpy.data.materials.remove(material, do_unlink=True)
        for texture in bpy.data.textures:
            bpy.data.textures.remove(texture, do_unlink=True)
        for image in bpy.data.images:
            bpy.data.images.remove(image, do_unlink=True)

    def init_camera(self):
        cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
        bpy.context.collection.objects.link(cam)
        bpy.context.scene.camera = cam
        cam.data.sensor_height = cam.data.sensor_width = 32
        cam_constraint = cam.constraints.new(type="TRACK_TO")
        cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
        cam_constraint.up_axis = "UP_Y"
        cam_empty = bpy.data.objects.new("Empty", None)
        cam_empty.location = (0, 0, 0)
        bpy.context.scene.collection.objects.link(cam_empty)
        cam_constraint.target = cam_empty
        return cam

    def init_lighting(self):
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.object.select_by_type(type="LIGHT")
        bpy.ops.object.delete()

        default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
        bpy.context.collection.objects.link(default_light)
        default_light.data.energy = 1000
        default_light.location = (4, 1, 6)
        default_light.rotation_euler = (0, 0, 0)

        top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
        bpy.context.collection.objects.link(top_light)
        top_light.data.energy = 10000
        top_light.location = (0, 0, 10)
        top_light.scale = (100, 100, 100)

        bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
        bpy.context.collection.objects.link(bottom_light)
        bottom_light.data.energy = 1000
        bottom_light.location = (0, 0, -10)
        bottom_light.rotation_euler = (0, 0, 0)
        return {"default_light": default_light, "top_light": top_light, "bottom_light": bottom_light}

    def load_object(self, object_path: str):
        file_extension = object_path.split(".")[-1].lower()
        if file_extension not in self.import_functions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        import_function = self.import_functions[file_extension]
        print(f"Loading object from {object_path}")
        if file_extension == "blend":
            import_function(directory=object_path, link=False)
        elif file_extension in {"glb", "gltf"}:
            import_function(filepath=object_path, merge_vertices=True, import_shading="NORMALS")
        else:
            import_function(filepath=object_path)

    def delete_invisible_objects(self):
        bpy.ops.object.select_all(action="DESELECT")
        for obj in bpy.context.scene.objects:
            if obj.hide_viewport or obj.hide_render:
                obj.hide_viewport = False
                obj.hide_render = False
                obj.hide_select = False
                obj.select_set(True)
        bpy.ops.object.delete()
        invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
        for col in invisible_collections:
            bpy.data.collections.remove(col)

    def unhide_all_objects(self):
        for obj in bpy.context.scene.objects:
            obj.hide_set(False)

    def convert_to_meshes(self):
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
        for obj in bpy.context.scene.objects:
            obj.select_set(True)
        bpy.ops.object.convert(target="MESH")

    def triangulate_meshes(self):
        bpy.ops.object.select_all(action="DESELECT")
        objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        bpy.context.view_layer.objects.active = objs[0]
        for obj in objs:
            obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.reveal()
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

    def split_mesh_normal(self):
        bpy.ops.object.select_all(action="DESELECT")
        objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        bpy.context.view_layer.objects.active = objs[0]
        for obj in objs:
            obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.split_normals()
        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")

    def delete_custom_normals(self):
        for this_obj in bpy.data.objects:
            if this_obj.type == "MESH":
                bpy.context.view_layer.objects.active = this_obj
                bpy.ops.mesh.customdata_custom_splitnormals_clear()

    def override_material(self):
        new_mat = bpy.data.materials.new(name="Override0123456789")
        new_mat.use_nodes = True
        new_mat.node_tree.nodes.clear()
        bsdf = new_mat.node_tree.nodes.new("ShaderNodeBsdfDiffuse")
        bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
        bsdf.inputs[1].default_value = 1
        output = new_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        new_mat.node_tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        bpy.context.scene.view_layers["View Layer"].material_override = new_mat

    def scene_bbox(self) -> Tuple[Vector, Vector]:
        bbox_min = (math.inf,) * 3
        bbox_max = (-math.inf,) * 3
        found = False
        scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
        for obj in scene_meshes:
            found = True
            for coord in obj.bound_box:
                coord = Vector(coord)
                coord = obj.matrix_world @ coord
                bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
                bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
        if not found:
            raise RuntimeError("no objects in scene to compute bounding box for")
        return Vector(bbox_min), Vector(bbox_max)

    def normalize_scene(self) -> Tuple[float, Vector]:
        scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
        if len(scene_root_objects) > 1:
            scene = bpy.data.objects.new("ParentEmpty", None)
            bpy.context.scene.collection.objects.link(scene)
            for obj in scene_root_objects:
                obj.parent = scene
        else:
            scene = scene_root_objects[0]

        bbox_min, bbox_max = self.scene_bbox()
        print(f"[INFO] Bounding box: {bbox_min}, {bbox_max}")
        scale = 1 / max(bbox_max - bbox_min)
        scene.scale = scene.scale * scale
        bpy.context.view_layer.update()
        bbox_min, bbox_max = self.scene_bbox()
        offset = -(bbox_min + bbox_max) / 2
        scene.matrix_world.translation += offset
        bpy.ops.object.select_all(action="DESELECT")
        return scale, offset

    def get_transform_matrix(self, obj: bpy.types.Object) -> list:
        pos, rt, _ = obj.matrix_world.decompose()
        rt = rt.to_matrix()
        matrix = []
        for ii in range(3):
            a = []
            for jj in range(3):
                a.append(rt[ii][jj])
            a.append(pos[ii])
            matrix.append(a)
        matrix.append([0, 0, 0, 1])
        return matrix

    def render_object(self, file_path: str, output_dir: str, num_views: int = 150, scale: float = 1.0, offset = None, save_mesh: bool = True) -> Dict:
        print(f"[RENDER] Starting render_object for: {file_path}")
        print(f"[RENDER] Output dir: {output_dir}, num_views: {num_views}")
        os.makedirs(output_dir, exist_ok=True)
        self.init_render_settings()
        if file_path.endswith(".blend"):
            self.delete_invisible_objects()
        else:
            self.init_scene()
            print(f"[RENDER] Loading object: {file_path}")
            self.load_object(file_path)
            
            # List all objects after loading
            print(f"[RENDER] Objects after loading: {[obj.name for obj in bpy.data.objects]}")
            print(f"[RENDER] Mesh objects: {[obj.name for obj in bpy.data.objects if obj.type == 'MESH']}")
            
            # NOTE: Flat shading fix is disabled by default since switching to BLENDER_EEVEE
            # resolved the shading issues. Uncomment below if needed for CYCLES rendering.
            # apply_flat_shading()
            
            if self.shade_smooth:
                self._apply_smooth_shading()
            
            if self.split_normal:
                self.split_mesh_normal()
            # delete_custom_normals()
        print("[INFO] Scene initialized.")

        if offset is None:
            scale, offset = self.normalize_scene()
            print(f"[INFO] Scene normalized with auto scale: {scale}, offset: {offset}")
        else:
            # Convert tuple/list offset to Vector if needed
            if not isinstance(offset, Vector):
                offset = Vector(offset)
            
            scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
            if len(scene_root_objects) > 1:
                scene = bpy.data.objects.new("ParentEmpty", None)
                bpy.context.scene.collection.objects.link(scene)
                for obj in scene_root_objects:
                    obj.parent = scene
            else:
                scene = scene_root_objects[0]
            scene.scale = scene.scale * scale
            bpy.context.view_layer.update()
            scene.matrix_world.translation += offset
            bpy.ops.object.select_all(action="DESELECT")
            print(f"[INFO] Scene scaled with specified scale: {scale}, offset: {offset}")

        cam = self.init_camera()
        self.init_lighting()
        print("[INFO] Camera and lighting initialized.")
        if self.geo_mode:
            self.override_material()

        yaws = []
        pitchs = []
        offset_random = (np.random.rand(), np.random.rand())
        for i in range(num_views):
            y, p = sphere_hammersley_sequence(i, num_views, offset_random)
            yaws.append(y)
            pitchs.append(p)
        radius = [2] * num_views
        fov = [40 / 180 * np.pi] * num_views
        views = [{"yaw": y, "pitch": p, "radius": r, "fov": f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
        to_export = {
            "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            "scale": scale,
            "offset": [offset.x, offset.y, offset.z],
            "frames": [],
        }

        for i, view in enumerate(views):
            cam.location = (
                view["radius"] * np.cos(view["yaw"]) * np.cos(view["pitch"]),
                view["radius"] * np.sin(view["yaw"]) * np.cos(view["pitch"]),
                view["radius"] * np.sin(view["pitch"]),
            )
            cam.data.lens = 16 / np.tan(view["fov"] / 2)
            bpy.context.scene.render.filepath = os.path.join(output_dir, f"{i:03d}.png")
            bpy.ops.render.render(write_still=True)
            bpy.context.view_layer.update()
            metadata = {
                "file_path": f"{i:03d}.png",
                "camera_angle_x": view["fov"],
                "transform_matrix": self.get_transform_matrix(cam),
            }
            to_export["frames"].append(metadata)
        with open(os.path.join(output_dir, "transforms.json"), "w") as f:
            json.dump(to_export, f, indent=4)

        mesh_file_path = None
        if save_mesh:
            try:
                self.unhide_all_objects()
                self.convert_to_meshes()
                self.triangulate_meshes()
                print("[INFO] Meshes triangulated.")
                ply_path = os.path.join(output_dir, "mesh.ply")
                try:
                    bpy.ops.wm.ply_export(filepath=ply_path)
                    mesh_file_path = ply_path
                    print("[INFO] Mesh file saved.")
                except AttributeError:
                    try:
                        bpy.ops.export_mesh.ply(filepath=ply_path)
                        mesh_file_path = ply_path
                        print("[INFO] Mesh file saved.")
                    except AttributeError:
                        print("[WARNING] PLY export not available, skipping mesh export")
            except Exception as e:
                print(f"[WARNING] Mesh export failed: {e}")
        return {
            "rendered": True,
            "num_views": num_views,
            "output_dir": output_dir,
            "transforms_file": os.path.join(output_dir, "transforms.json"),
            "mesh_file": mesh_file_path,
        }


def render_3d_model(
    file_path: str, 
    output_dir: str, 
    num_views: int = 150, 
    scale: float = 1.0, 
    offset = None, 
    resolution: int = 512, 
    engine: str = "BLENDER_EEVEE", 
    geo_mode: bool = False, 
    split_normal: bool = False, 
    save_mesh: bool = True,
    shade_smooth: bool = False,
) -> Dict:
    """
    Render a 3D model from multiple views.
    
    This is a modified version of VoxHammer's render_3d_model with flat shading fix.
    
    Args:
        file_path: Path to the input 3D model
        output_dir: Directory to save rendered images
        num_views: Number of views to render (default 150)
        scale: Scale factor (default 1.0)
        offset: Optional offset vector
        resolution: Render resolution (default 512)
        engine: Blender render engine (default "BLENDER_EEVEE")
        geo_mode: If True, use geometry-only rendering
        split_normal: If True, split normals
        save_mesh: If True, save mesh as PLY
    
    Returns:
        Dict with rendering results
    """
    renderer = BpyRenderer(resolution=resolution, engine=engine, geo_mode=geo_mode, split_normal=split_normal, shade_smooth=shade_smooth)
    return renderer.render_object(file_path, output_dir, num_views, scale, offset, save_mesh)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Render a 3D model with flat shading fix")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input 3D model")
    parser.add_argument("--output", "-o", type=str, default="./render_test", help="Output directory")
    parser.add_argument("--num_views", "-n", type=int, default=5, help="Number of views to render (default: 5)")
    parser.add_argument("--resolution", "-r", type=int, default=512, help="Render resolution (default: 512)")
    parser.add_argument("--engine", "-e", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"], 
                        help="Render engine (default: BLENDER_EEVEE)")
    
    args = parser.parse_args()
    
    print("="*60)
    print("RENDERING TEST WITH FLAT SHADING FIX")
    print("="*60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Num views: {args.num_views}")
    print(f"Resolution: {args.resolution}")
    print(f"Engine: {args.engine}")
    print("="*60)
    
    result = render_3d_model(
        file_path=args.input,
        output_dir=args.output,
        num_views=args.num_views,
        resolution=args.resolution,
        engine=args.engine,
    )
    
    print("="*60)
    print("RENDER COMPLETE")
    print("="*60)
    print(f"Result: {result}")

