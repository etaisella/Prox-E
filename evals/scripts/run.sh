#!/usr/bin/env bash
# Launch evals.run_eval on the full 600-sample ShapeTalk set.
#
# Reads inputs from the prox-e-shapetalk-benchmark/ folder
# (instructions.json, input_shapes/, rendered_images/, point_cloud/).
set -euo pipefail

PRED_MESH_DIR=${PRED_MESH_DIR:-proxe_test_600}
INPUT_MESH_DIR=${INPUT_MESH_DIR:-prox-e-shapetalk-benchmark/input_shapes}
INSTRUCTIONS_JSON=${INSTRUCTIONS_JSON:-prox-e-shapetalk-benchmark/instructions.json}
INPUT_RENDER_DIR=${INPUT_RENDER_DIR:-prox-e-shapetalk-benchmark/rendered_images}
INPUT_PCD_DIR=${INPUT_PCD_DIR:-prox-e-shapetalk-benchmark/point_cloud}              

OUTPUT_DIR=${OUTPUT_DIR:-proxe_eval_outputs/results}
DEVICE=${DEVICE:-cuda:0}


input_render_flag=()
if [[ -n "$INPUT_RENDER_DIR" ]]; then
  input_render_flag=(--input_render_dir "$INPUT_RENDER_DIR")
fi

input_pcd_flag=()
if [[ -n "$INPUT_PCD_DIR" ]]; then
  input_pcd_flag=(--input_pcd_dir "$INPUT_PCD_DIR")
fi

python -m evals.main \
  --pred_dir "$PRED_MESH_DIR" \
  --input_dir "$INPUT_MESH_DIR" \
  --instructions_json "$INSTRUCTIONS_JSON" \
  "${input_render_flag[@]}" \
  "${input_pcd_flag[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE" \
  --metrics identity quality fidelity \
  --enable_vqa