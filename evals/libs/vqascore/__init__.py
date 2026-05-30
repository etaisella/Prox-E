"""VQAScore library, vendored from https://github.com/linzhiqiu/t2v_metrics.

Public surface used by ``evals/metrics_fidelity.py``:

- ``VQAScore``: the scorer class.
- ``QUESTION_TEMPLATE``: the paper's question (idx 4 from
  ``evaluate_shapetalk_pair_images.py`` in the upstream repo).
- ``IMAGE_DESCRIPTIONS``: interleaved-prompt labels for the (source, edited)
  image pair, matching ``run_eval_shapetalk_mesh_pairs.sh``.
- ``SYSTEM_PROMPT_COT_PATH`` / ``SYSTEM_PROMPT_COT_THINKING_PATH``: chain-of-thought
  system prompts. ``system_prompt_for(model)`` picks the right one for a model.

  The CoT prompts make the model reason and then emit ``Final Answer:\\nYes/No``
  (or ``\\boxed{Yes/No}`` for thinking models), so they only score correctly with
  ``forward_multi_images(..., parse_answer=True)``, which reads the softmax over
  the final Yes/No token rather than the first generated token.
"""

from pathlib import Path

from .vqascore import VQAScore
from .models import get_vqascore_model, list_all_vqascore_models

QUESTION_TEMPLATE = (
    "Does the change from Image 1 to Image 2 correctly reflect the text '{}'? "
    "Answer Yes or No."
)

IMAGE_DESCRIPTIONS = [
    "Image 1 is, the original.",
    "Next is Image 2, the edited version.",
]

_PROMPTS = Path(__file__).parent / "prompts"
SYSTEM_PROMPT_COT_PATH = _PROMPTS / "system_prompt_cot.txt"
SYSTEM_PROMPT_COT_THINKING_PATH = _PROMPTS / "system_prompt_cot_thinking_models.txt"


def system_prompt_for(model_name: str) -> Path:
    """CoT system prompt matching the model; thinking models use the boxed-answer variant."""
    if "thinking" in model_name:
        return SYSTEM_PROMPT_COT_THINKING_PATH
    return SYSTEM_PROMPT_COT_PATH


__all__ = [
    "VQAScore",
    "get_vqascore_model",
    "list_all_vqascore_models",
    "QUESTION_TEMPLATE",
    "IMAGE_DESCRIPTIONS",
    "SYSTEM_PROMPT_COT_PATH",
    "SYSTEM_PROMPT_COT_THINKING_PATH",
    "system_prompt_for",
]
