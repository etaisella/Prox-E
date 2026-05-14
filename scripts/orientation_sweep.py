#!/usr/bin/env python3
"""Render a custom mesh under every supported input orientation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from prox_e.orientation import ORIENTATION_TRANSFORMS, load_oriented_normalized_mesh
from prox_e.utils import render_obj_with_blender


def make_overview(image_paths: list[Path], output_path: Path, *, thumb_size: int = 220) -> None:
    if not image_paths:
        return

    cols = 4
    label_h = 28
    rows = (len(image_paths) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + label_h)), "white")
    draw = ImageDraw.Draw(canvas)

    for i, path in enumerate(image_paths):
        img = Image.open(path).convert("RGBA")
        img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        row, col = divmod(i, cols)
        x = col * thumb_size + (thumb_size - img.width) // 2
        y = row * (thumb_size + label_h)
        tile_bg = Image.new("RGBA", img.size, "white")
        tile_bg.alpha_composite(img)
        canvas.paste(tile_bg.convert("RGB"), (x, y))

        idx = int(path.name.split("_")[1])
        name = ORIENTATION_TRANSFORMS[idx][0]
        draw.text((col * thumb_size + 8, y + thumb_size + 6), f"{idx}: {name}", fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize and render a mesh under every Prox-E orientation index."
    )
    parser.add_argument("--input_mesh", type=Path, required=True, help="Path to a custom mesh file")
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=None,
        help="Output directory (default: <input parent>/orientation_sweep/<input stem>)",
    )
    parser.add_argument("--skip_existing", action="store_true", help="Reuse existing PNGs")
    args = parser.parse_args()

    input_mesh = args.input_mesh.resolve()
    if not input_mesh.is_file():
        raise SystemExit(f"Input mesh not found: {input_mesh}")

    out_dir = args.output_folder or input_mesh.parent / "orientation_sweep" / input_mesh.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {input_mesh}")
    print(f"Output: {out_dir}")
    print(f"Rendering {len(ORIENTATION_TRANSFORMS)} orientations with the original.png camera/style")

    rendered: list[Path] = []
    for idx in sorted(ORIENTATION_TRANSFORMS):
        name = ORIENTATION_TRANSFORMS[idx][0]
        stem = f"orientation_{idx:02d}_{name}"
        normalized_obj = out_dir / f"{stem}.obj"
        render_png = out_dir / f"{stem}.png"

        if args.skip_existing and render_png.exists():
            rendered.append(render_png)
            print(f"  [{idx:02d}] {name}: reused {render_png.name}")
            continue

        load_oriented_normalized_mesh(
            input_mesh,
            normalized_obj,
            orientation_index=idx,
            edit3dbench=False,
        )
        render_obj_with_blender(str(normalized_obj), str(render_png), shade_smooth=True)
        rendered.append(render_png)
        print(f"  [{idx:02d}] {name}: {render_png.name}")

    overview = out_dir / "orientation_sweep_overview.png"
    make_overview(rendered, overview)
    print(f"\nOverview: {overview}")
    print("Use the preferred index with inference.py --orientation_index <index>.")


if __name__ == "__main__":
    main()
