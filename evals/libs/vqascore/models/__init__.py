"""Trimmed VQAScore model registry: only the 3 wrapper families used by Prox-E.

Wrapper modules are imported lazily (qwen2vl/sailvl/internvl pull in
``qwen_vl_utils``, ``decord``, and other heavy deps that we don't want to
require just to import the package).
"""

import importlib

from ..constants import HF_CACHE_DIR

# (module_path, models_attr, class_attr) tuples — resolved on first use.
_WRAPPERS = [
    (".qwen2vl_model", "QWEN2_VL_MODELS", "Qwen2VLModel"),
    (".internvl_model", "INTERNVL2_MODELS", "InternVL2Model"),
    (".sailvl_model", "SAIL_VL_MODELS", "SAILVLModel"),
]

# Hardcoded model-name lists so list_all_vqascore_models() and the dispatch in
# get_vqascore_model() work without importing the heavy wrappers.
_MODEL_NAMES = {
    "qwen2vl": [
        "qwen2-vl-2b",
        "qwen2-vl-7b",
        "qwen2-vl-72b",
        "qwen2.5-vl-3b",
        "qwen2.5-vl-7b",
        "qwen2.5-vl-32b",
        "qwen2.5-vl-72b",
        "qwen3-vl-8b",
        "qwen3-vl-8b-thinking",
    ],
    "internvl": [
        "internvl2-1b",
        "internvl2-2b",
        "internvl2-4b",
        "internvl2-8b",
        "internvl2-26b",
        "internvl2-40b",
        "internvl2-llama3-76b",
        "internvl2.5-1b",
        "internvl2.5-2b",
        "internvl2.5-4b",
        "internvl2.5-8b",
        "internvl2.5-26b",
        "internvl2.5-38b",
        "internvl2.5-78b",
        "internvl3-8b",
        "internvl3-14b",
        "internvl3-78b",
        "internvl3_5-8b",
        "sensenova-si-internvl3-8b",
    ],
    "sailvl": ["sail-vl-8b", "sail-vl-8b-thinking"],
}


def list_all_vqascore_models():
    return [m for names in _MODEL_NAMES.values() for m in names]


def get_vqascore_model(model_name, device="cuda", cache_dir=HF_CACHE_DIR, **kwargs):
    assert model_name in list_all_vqascore_models(), (
        f"Unknown VQAScore model {model_name!r}; available: {list_all_vqascore_models()}"
    )
    if model_name in _MODEL_NAMES["qwen2vl"]:
        mod = importlib.import_module(".qwen2vl_model", __name__)
        return mod.Qwen2VLModel(
            model_name, device=device, cache_dir=cache_dir, **kwargs
        )
    if model_name in _MODEL_NAMES["internvl"]:
        mod = importlib.import_module(".internvl_model", __name__)
        return mod.InternVL2Model(
            model_name, device=device, cache_dir=cache_dir, **kwargs
        )
    if model_name in _MODEL_NAMES["sailvl"]:
        mod = importlib.import_module(".sailvl_model", __name__)
        return mod.SAILVLModel(model_name, device=device, cache_dir=cache_dir, **kwargs)
    raise NotImplementedError(model_name)
