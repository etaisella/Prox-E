# Prox-E Evaluation

Unified evaluator for Prox-E mesh outputs. Computes seven metrics in three groups:

| Group | Metric | Notes |
| --- | --- | --- |
| Identity Preservation | l-GD | masked Chamfer on the non-edited region |
| | LPIPS | perceptual distance on renders of input and output shapes (VGG) |
| | DINO-I | cosine similarity of DINOv2 features on renders of input and output shapes |
| 3D Quality | PFD | Fréchet distance over PointNet-classifier features |
| | FID | Fréchet distance over InceptionV3 features on renders |
| Edit Fidelity | CLIP (sim + dir) | CLIP scores on mesh renders (ViT-B/32) |
| | VQA | VQAScore given a pair of input and output renders, and an edit description |

## Download ShapeTalk Benchmark
```
hf download haopt/prox-e-shapetalk-benchmark --repo-type=dataset
```

## Usage

Use our provided script:
```bash
bash evals/scripts/run.sh
```
or
```bash
python -m evals.main \
  --pred_dir <flat folder of pred .glb/.obj/.ply> \
  --input_dir prox-e-shapetalk-benchmark/input_shapes \
  --instructions_json prox-e-shapetalk-benchmark/instructions.json \
  --input_render_dir prox-e-shapetalk-benchmark/rendered_images \
  --input_pcd_dir prox-e-shapetalk-benchmark/point_cloud \
  --output_dir <eval_run_dir> \
  --device cuda:0 \
  --metrics identity quality fidelity
```

Notes:
- `--input_render_dir` / `--input_pcd_dir` are optional; if omitted, renders and point clouds are generated and cached under `--output_dir/{renders,pcd}/` and reused on subsequent runs.
- Prox-E `.glb` outputs come out of the TRELLIS coordinate system. To render them upright like the GT meshes, pass `--pred_rotation -90 0 0`.


<details>
<summary><b>Details of instructions.json</b></summary>

```json
{
  "some_id": {
    "instruction": "make the chair legs longer",
    "obj_class":   "chair",
    "part_keyword": "leg"          
  },
  ...
}
```

`obj_class` enables `CLIP-Dir`; `obj_class` + `part_keyword` (or a recognisable part word in `instruction`) enables `l-GD`.

</details>

### Output

`<eval_run_dir>/results.json`:

```json
{
  "n_samples": 600,
  "identity_preservation": {"LPIPS": 0.10, "DINO-I": 0.92, "l-GD": 0.02,
                            "l-GD_per_class": {"chair": 0.13, "table": 0.20, "lamp": 0.21}},
  "3d_quality":            {"PFD": 11.34, "FID": 32.60},
  "edit_fidelity":         {"CLIP-Sim": 0.27, "CLIP-Dir": 0.91, "VQA": 0.71}
}
```

`<eval_run_dir>/per_sample.csv` is keyed by `sample_id` with one row per sample and one column per metric, except PFD / FID are distribution-level metrics (don't have).

## Checkpoints

All weights download lazily on first use — no setup step required. Auto-fetched paths:

- `evals/checkpoints/pointnet.pt` — Point-E PointNet classifier (PFD), pulled by `evals.libs.pointnet_cls`.
- `evals/checkpoints/lgd/{class}_100.onnx` — per-class PointNet segmenters (l-GD), pulled from `ailia-models/pointnet_pytorch/` by `evals.libs.lgd`.
- CLIP ViT-B/32 (`~/.cache/clip/`), DINOv2-base (HuggingFace cache), LPIPS-VGG (`~/.cache/torch/hub/checkpoints/`) — fetched by their respective Python wrappers.
- VQA backbone (select via `--vqa_model`) — default `qwen2.5-vl-7b` (`Qwen/Qwen2.5-VL-7B-Instruct`); also supports the rest of the Qwen2-VL / Qwen2.5-VL / Qwen3-VL families, `sail-vl-8b` / `sail-vl-8b-thinking`, and `internvl3_5-8b`.