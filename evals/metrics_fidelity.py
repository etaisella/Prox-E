"""Edit Fidelity metrics: CLIP-Sim, CLIP-Dir, VQA (placeholder).

CLIP-Sim: cosine(CLIP-image(pred_render), CLIP-text(instruction)), averaged over views.
CLIP-Dir: directional CLIP distance:
              text_dir = CLIP-text(instruction) - CLIP-text(source_caption)
              im_dir   = CLIP-image(pred)       - CLIP-image(gt)
              CLIP-Dir = 1 - cos(text_dir, im_dir)            (lower is better)
          Requires a per-sample source caption (object class). If
          ``instructions`` values are dicts containing ``obj_class``, that
          drives the source caption ``"A {obj_class}"``; otherwise CLIP-Dir is
          omitted.
VQA:      P("Yes") from a VLM judging whether the change from GT render
          (Image 1) to pred render (Image 2) reflects the instruction. Uses
          question template 4 plus the chain-of-thought system prompt from the
          upstream t2v_metrics paper: the model reasons, emits a final Yes/No,
          and the score is the softmax over that final token (``parse_answer``).
          The thinking model uses the boxed-answer prompt variant. Disabled by
          default; gate via ``enable_vqa``.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from evals.libs import clip as clip_wrapper

EPS = 1e-8


def _resolve_instruction(value) -> Tuple[str, Optional[str]]:
    """Accept either ``"text"`` or ``{"instruction": "...", "obj_class": "..."}``."""
    if isinstance(value, str):
        return value, None
    instruction = value["instruction"]
    obj_class = value.get("obj_class")
    return instruction, obj_class


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a / (a.norm(dim=-1, keepdim=True) + EPS)
    b = b / (b.norm(dim=-1, keepdim=True) + EPS)
    return float((a * b).sum(dim=-1).mean())


def _avg_clip_dir(text_dir: torch.Tensor, im_dirs: torch.Tensor) -> float:
    """1 - cos(text_dir, per-view image dir), averaged over views. Matches legacy."""
    text_dir = text_dir / (text_dir.norm(dim=-1, keepdim=True) + EPS)
    im_dirs = im_dirs / (im_dirs.norm(dim=-1, keepdim=True) + EPS)
    cos = (im_dirs * text_dir).sum(dim=-1)  # (V,)
    return float((1.0 - cos).mean())


def compute(
    *,
    sample_ids: List[str],
    render_dir: Path,
    instructions: Dict,
    device: str = "cuda:0",
    enable_vqa: bool = False,
    vqa_model: str = "sail-vl-8b",
    vqa_cache_dir: Optional[Path] = None,
    vqa_checkpoint: Optional[str] = None,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]]]:
    """Returns (aggregate, per_sample)."""
    pred_root = render_dir / "pred"
    gt_root = render_dir / "gt"
    model, preprocess = clip_wrapper.load(model_name="ViT-B/32", device=device)

    per_sample: Dict[str, Dict[str, float]] = {}
    sim_values: List[float] = []
    dir_values: List[float] = []

    for sid in tqdm(sample_ids, desc="CLIP"):
        instruction, obj_class = _resolve_instruction(instructions[sid])

        pred_paths = clip_wrapper.render_paths_for(pred_root, sid)
        if not pred_paths:
            continue
        pred_feats = clip_wrapper.encode_image_paths(
            pred_paths, model, preprocess, device
        )
        instr_feat = clip_wrapper.encode_text(instruction, model, device)

        sim = _cosine(pred_feats, instr_feat.expand_as(pred_feats))
        per_sample.setdefault(sid, {})["CLIP-Sim"] = sim
        sim_values.append(sim)

        if obj_class is not None:
            gt_paths = clip_wrapper.render_paths_for(gt_root, sid)
            if gt_paths and len(gt_paths) == len(pred_paths):
                gt_feats = clip_wrapper.encode_image_paths(
                    gt_paths, model, preprocess, device
                )
                src_feat = clip_wrapper.encode_text(f"A {obj_class}", model, device)
                text_dir = instr_feat - src_feat
                im_dirs = pred_feats - gt_feats
                d = _avg_clip_dir(text_dir, im_dirs)
                per_sample[sid]["CLIP-Dir"] = d
                dir_values.append(d)

    aggregate: Dict[str, object] = {
        "CLIP-Sim": float(np.mean(sim_values)) if sim_values else float("nan"),
        "VQA": None,
    }
    if dir_values:
        aggregate["CLIP-Dir"] = float(np.mean(dir_values))

    if enable_vqa:
        from evals.libs.vqascore import (
            VQAScore,
            QUESTION_TEMPLATE,
            IMAGE_DESCRIPTIONS,
            system_prompt_for,
        )

        vqa_init_kwargs = {
            "model": vqa_model,
            "device": device,
            "system_prompt": system_prompt_for(vqa_model).read_text(),
        }
        if vqa_cache_dir is not None:
            vqa_init_kwargs["cache_dir"] = str(vqa_cache_dir)
        if vqa_checkpoint is not None:
            vqa_init_kwargs["checkpoint"] = vqa_checkpoint
        scorer = VQAScore(**vqa_init_kwargs)
        vqa_values: List[float] = []
        for sid in tqdm(sample_ids, desc="VQA"):
            instruction, _ = _resolve_instruction(instructions[sid])
            pred_paths = clip_wrapper.render_paths_for(pred_root, sid)
            gt_paths = clip_wrapper.render_paths_for(gt_root, sid)
            if not pred_paths or not gt_paths:
                continue
            scores, _ = scorer.forward_multi_images(
                images=[[str(gt_paths[0]), str(pred_paths[0])]],
                texts=[instruction],
                question_template=QUESTION_TEMPLATE,
                answer_template="Yes",
                image_descriptions=IMAGE_DESCRIPTIONS,
                parse_answer=True,
            )
            score = scores[0].item()
            per_sample.setdefault(sid, {})["VQA"] = score
            vqa_values.append(score)
        aggregate["VQA"] = float(np.mean(vqa_values)) if vqa_values else float("nan")

    return aggregate, per_sample
