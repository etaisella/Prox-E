#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${PROXE_ENV_NAME:-prox-e}"
PYTHON_VERSION="${PROXE_PYTHON_VERSION:-3.11}"
CUDA_TAG="${PROXE_CUDA:-cu128}"
TORCH_VERSION="${PROXE_TORCH_VERSION:-2.7.1}"
TORCHVISION_VERSION="${PROXE_TORCHVISION_VERSION:-0.22.1}"
TORCHAUDIO_VERSION="${PROXE_TORCHAUDIO_VERSION:-2.7.1}"

case "${CUDA_TAG}" in
  cu128|cu126|cu118|cpu)
    ;;
  *)
    echo "Unsupported PROXE_CUDA='${CUDA_TAG}'. Use one of: cu128, cu126, cu118, cpu." >&2
    exit 2
    ;;
esac

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH. Install Miniconda/Mambaforge or source conda.sh first." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Updating existing conda environment: ${ENV_NAME}"
  conda install -y -n "${ENV_NAME}" -c conda-forge \
    "python=${PYTHON_VERSION}" pip setuptools wheel packaging ninja
else
  echo "Creating conda environment: ${ENV_NAME}"
  conda create -y -n "${ENV_NAME}" -c conda-forge \
    "python=${PYTHON_VERSION}" pip setuptools wheel packaging ninja
fi

conda activate "${ENV_NAME}"
cd "${REPO_ROOT}"

python -m pip install --upgrade pip setuptools wheel packaging ninja

if [[ "${CUDA_TAG}" == "cpu" ]]; then
  python -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install \
    "torch==${TORCH_VERSION}+${CUDA_TAG}" \
    "torchvision==${TORCHVISION_VERSION}+${CUDA_TAG}" \
    "torchaudio==${TORCHAUDIO_VERSION}+${CUDA_TAG}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

# Prefer PyPI as the primary index so a misconfigured corporate PIP_INDEX_URL
# (401 / incomplete mirrors) does not shadow packages like bpy from PyPI + Blender extras.
PIP_INDEX_URL="${PROXE_PIP_INDEX_URL:-https://pypi.org/simple}" \
PIP_EXTRA_INDEX_URL="${PROXE_PIP_EXTRA_INDEX_URL:-https://download.blender.org/pypi/}" \
  python -m pip install -r requirements.txt

# nvdiffrast must be built for the GPU you run on; otherwise Trellis GLB export hits CUDA error 209.
if [[ "${CUDA_TAG}" != "cpu" ]]; then
  if [[ -n "${PROXE_TORCH_CUDA_ARCH_LIST:-}" ]]; then
    export TORCH_CUDA_ARCH_LIST="${PROXE_TORCH_CUDA_ARCH_LIST}"
    echo "[nvdiffrast] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (from PROXE_TORCH_CUDA_ARCH_LIST)"
  elif [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      _cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '[:space:]')"
      if [[ -n "${_cc}" ]]; then
        export TORCH_CUDA_ARCH_LIST="${_cc}+PTX"
        echo "[nvdiffrast] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (from nvidia-smi first GPU)"
      else
        export TORCH_CUDA_ARCH_LIST="8.0+PTX"
        echo "[nvdiffrast] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (fallback: could not read compute_cap)"
      fi
    else
      export TORCH_CUDA_ARCH_LIST="8.0+PTX"
      echo "[nvdiffrast] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} (fallback: nvidia-smi not found)"
    fi
  else
    echo "[nvdiffrast] Using existing TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
  fi
  # --no-deps: nvdiffrast otherwise upgrades NumPy and breaks numba / opencv-headless (see requirements.txt pin).
  python -m pip install --no-build-isolation --force-reinstall --no-deps git+https://github.com/NVlabs/nvdiffrast.git

  # FlashAttention for Trellis (dense + sparse). Uses TORCH_CUDA_ARCH_LIST from above; build can take many minutes without a matching wheel.
  if [[ "${PROXE_SKIP_FLASH_ATTN:-0}" != "1" ]]; then
    _fa_spec="${PROXE_FLASH_ATTN_SPEC:-flash-attn}"
    echo "[flash-attn] Installing ${_fa_spec} (CUDA extension; set PROXE_SKIP_FLASH_ATTN=1 to skip)..."
    _mj="${PROXE_FLASH_ATTN_MAX_JOBS:-}"
    if [[ -z "${_mj}" ]] && command -v nproc >/dev/null 2>&1; then
      _mj="$(nproc)"
    fi
    if [[ -n "${_mj}" ]]; then
      MAX_JOBS="${_mj}" python -m pip install --no-build-isolation "${_fa_spec}"
    else
      python -m pip install --no-build-isolation "${_fa_spec}"
    fi
  else
    echo "[flash-attn] Skipped (PROXE_SKIP_FLASH_ATTN=1)."
  fi
else
  echo "[nvdiffrast] Skipping (CPU PyTorch build)."
fi
python -m pip install --no-build-isolation \
  "git+https://github.com/autonomousvision/mip-splatting.git#subdirectory=submodules/diff-gaussian-rasterization"

# Re-pin NumPy after native extension installs (nvdiffrast / diff-gaussian-rasterization can pull a newer NumPy).
python -m pip install --no-cache-dir "numpy==2.2.6"

if [[ ! -d "prox_e/submodules/superdec/checkpoints/normalized" ]]; then
  echo "SuperDec checkpoints were not found; downloading them now."
  (cd prox_e/submodules/superdec && bash scripts/download_checkpoints.sh)
fi

cat <<EOF

Prox-E environment setup complete.

Activate it with:
  conda activate ${ENV_NAME}

Optional overrides for future setup runs:
  PROXE_ENV_NAME=<name> PROXE_CUDA=<cu128|cu126|cu118|cpu> bash scripts/setup_environment.sh

pip index overrides (requirements install only; defaults avoid broken corporate PIP_INDEX_URL):
  PROXE_PIP_INDEX_URL=... PROXE_PIP_EXTRA_INDEX_URL=... bash scripts/setup_environment.sh

nvdiffrast / GPU arch (CUDA builds only; avoids runtime CUDA error 209 on Trellis mesh export):
  PROXE_TORCH_CUDA_ARCH_LIST=8.0+PTX bash scripts/setup_environment.sh

flash-attn (CUDA builds only; skip if you have no working CUDA toolkit / compiler for source builds):
  PROXE_SKIP_FLASH_ATTN=1 bash scripts/setup_environment.sh
  PROXE_FLASH_ATTN_SPEC='flash-attn==2.8.3' bash scripts/setup_environment.sh
EOF
