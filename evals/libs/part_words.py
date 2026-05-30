"""Per-class part-label tables and instruction-text → part-keyword resolution.

Kept in its own module so light-weight tools (e.g. dataset prep scripts) can
use ``auto_resolve_part`` and ``CATEGORIES`` without pulling in
``onnxruntime`` / ``torch`` via ``evals.libs.lgd``.
"""

from __future__ import annotations

from typing import Dict, List, Optional


# Per-class part name -> ONNX seg label indices. Verbatim from
# changeit3d ``in_out/pointcloud_ours.py``.
CATEGORIES: Dict[str, Dict[str, List[int]]] = {
    "chair": {"back": [0], "seat": [1], "leg": [2], "arm": [3]},
    "table": {"top": [0], "leg": [1], "support": [2]},
    "lamp": {"base": [0], "shade": [1], "bulb": [2], "tube": [3]},
    "airplane": {"body": [0], "wing": [1], "tail": [2], "engine": [3]},
    "sofa": {"back": [0], "seat": [1], "leg": [2], "arm": [3]},
    "knife": {"blade": [0], "handle": [1]},
    "cap": {"brim": [0], "crown": [1]},
    "skateboard": {"wheel": [0], "deck": [1], "truck": [2]},
    "mug": {"handle": [0], "body": [1]},
    "pistol": {"barrel": [0], "handle": [1], "trigger": [2]},
    "bag": {"strap": [0], "body": [1]},
    "guitar": {"head": [0], "neck": [1], "body": [2]},
}


# Per-class synonyms used to auto-resolve a part keyword from raw instruction
# text. Adapted from changeit3d ``in_out/datasets/shape_net_parts.py:canonical_part_names_to_relevant_words``.
PART_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "chair": {
        "leg": [
            "leg",
            "legs",
            "legged",
            "wheel",
            "wheels",
            "swivel",
            "foot",
            "feet",
            "base",
            "footrest",
            "roller",
            "caster",
        ],
        "back": [
            "back",
            "backed",
            "backrest",
            "backboard",
            "backside",
            "slat",
            "slats",
            "head",
            "headrest",
        ],
        "arm": ["arm", "arms", "armrest", "armrests"],
        "seat": ["seat", "sit", "seating", "sitting", "seater"],
    },
    "table": {
        "leg": [
            "leg",
            "legs",
            "feet",
            "foot",
            "footrest",
            "stretcher",
            "base",
            "pedestal",
        ],
        "top": ["top", "apron", "tabletop"],
        "support": ["support", "stretcher", "brace"],
    },
    "lamp": {
        "base": ["base"],
        "shade": ["shade", "lampshade"],
        "bulb": ["bulb"],
        "tube": ["tube", "body", "pole"],
    },
}


def auto_resolve_part(instruction: str, obj_class: str) -> Optional[str]:
    """Return the first canonical part name found in the instruction text.
    Falls back to ``None`` when no keyword matches (caller may treat as ``unknown``).
    """
    table = PART_KEYWORDS.get(obj_class)
    if table is None:
        return None
    tokens = {t.strip(",.!?\"'()[]").lower() for t in instruction.split()}
    for canonical, words in table.items():
        if tokens & set(words):
            return canonical
    return None
