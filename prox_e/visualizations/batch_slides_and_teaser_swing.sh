#!/usr/bin/env bash
# For each direct result subfolder under the given roots, run:
#   1) visualizations_for_slides.py --override (recreates teaser_renders/ among other outputs)
#   2) render_teaser_swing_parallel.sh on <result>/teaser_renders
#
# Usage:
#   ./batch_slides_and_teaser_swing.sh
#   DRY_RUN=1 ./batch_slides_and_teaser_swing.sh   # print commands only

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIZ_PY="${SCRIPT_DIR}/visualizations_for_slides.py"
RENDER_SWING="${SCRIPT_DIR}/render_teaser_swing_parallel.sh"

EDIT3D_ROOT="../../edit3dbench_results"
SHAPETALK_ROOT="../../shapetalk_results"

if [[ ! -f "$VIZ_PY" ]]; then
  echo "Missing: $VIZ_PY" >&2
  exit 1
fi
if [[ ! -f "$RENDER_SWING" ]]; then
  echo "Missing: $RENDER_SWING" >&2
  exit 1
fi

run_one() {
  local result_dir="$1"
  local teaser_dir="${result_dir}/teaser_renders"

  if [[ -n "${DRY_RUN:-}" ]]; then
    echo "python3 \"${VIZ_PY}\" --result-folder \"${result_dir}\" --override"
    echo "\"${RENDER_SWING}\" \"${teaser_dir}\""
    return 0
  fi

  echo "=== ${result_dir} ==="
  if ! python3 "${VIZ_PY}" --result-folder "${result_dir}" --override; then
    echo "visualizations_for_slides.py failed: ${result_dir}" >&2
    return 1
  fi

  if [[ ! -d "$teaser_dir" ]]; then
    echo "No teaser_renders directory after viz (skip swing): ${teaser_dir}" >&2
    return 1
  fi

  if ! "${RENDER_SWING}" "$teaser_dir"; then
    echo "render_teaser_swing_parallel.sh failed: ${teaser_dir}" >&2
    return 1
  fi
  return 0
}

process_tree_root() {
  local tree_root="$1"
  if [[ ! -d "$tree_root" ]]; then
    echo "Skipping missing directory: $tree_root" >&2
    return 0
  fi

  shopt -s nullglob
  local candidates=("$tree_root"/*)
  shopt -u nullglob
  if [[ ${#candidates[@]} -eq 0 ]]; then
    echo "No entries under: $tree_root" >&2
    return 0
  fi

  for path in "${candidates[@]}"; do
    [[ -d "$path" ]] || continue
    local base
    base="$(basename "$path")"
    [[ "$base" == .* ]] && continue
    run_one "$path" || true
  done
}

if [[ -n "${DRY_RUN:-}" ]]; then
  process_tree_root "$EDIT3D_ROOT"
  process_tree_root "$SHAPETALK_ROOT"
  exit 0
fi

process_tree_root "$EDIT3D_ROOT"
process_tree_root "$SHAPETALK_ROOT"
