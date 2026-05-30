import torch
import numpy as np
from PIL import Image
from typing import List, Union
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from qwen_vl_utils import process_vision_info
from .vqa_model import VQAScoreModel

QWEN2_VL_MODELS = {
    # Qwen2_VL
    "qwen2-vl-2b": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-2B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2-VL-2B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        "fps": 8.0,
    },
    "qwen2-vl-7b": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2-VL-7B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        "fps": 8.0,
    },
    "qwen2-vl-72b": {
        "tokenizer": {
            "path": "Qwen/Qwen2-VL-72B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2-VL-72B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        "fps": 8.0,
    },
    # Qwen2.5_VL:
    "qwen2.5-vl-3b": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-3B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2.5-VL-3B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        "fps": 8.0,
    },
    "qwen2.5-vl-7b": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2.5-VL-7B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        "fps": 8.0,
    },
    "qwen2.5-vl-32b": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-32B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2.5-VL-32B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        "fps": 8.0,
    },
    "qwen2.5-vl-72b": {
        "tokenizer": {
            "path": "Qwen/Qwen2.5-VL-72B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen2.5-VL-72B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        "fps": 8.0,
    },
    # qwen3
    "qwen3-vl-8b": {
        "tokenizer": {
            "path": "Qwen/Qwen3-VL-8B-Instruct",
        },
        "model": {
            "path": "Qwen/Qwen3-VL-8B-Instruct",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        "fps": 8.0,  # Qwen3 might support dynamic fps better, but 8.0 is a safe default
    },
    "qwen3-vl-8b-thinking": {
        "tokenizer": {
            "path": "Qwen/Qwen3-VL-8B-Thinking",
        },
        "model": {
            "path": "Qwen/Qwen3-VL-8B-Thinking",
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        "fps": 8.0,  # Qwen3 might support dynamic fps better, but 8.0 is a safe default
    },
}


class Qwen2VLModel(VQAScoreModel):
    video_mode = "direct"
    allows_image = True

    def __init__(
        self,
        model_name="qwen2.5-vl-7b",
        device="cuda",
        cache_dir=None,
        checkpoint=None,
        **kwargs,
    ):
        assert model_name in QWEN2_VL_MODELS, (
            f"Model {model_name} not found in QWEN2_VL_MODELS"
        )
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = QWEN2_VL_MODELS[model_name]
        self.checkpoint = checkpoint if checkpoint else self.model_info["model"]["path"]
        self.system_prompt = kwargs.get("system_prompt", None)
        self.load_model()

    def load_model(self):
        # Switch from model dictionary to checkpoint argument
        # model_path = self.model_info['model']['path']
        print(
            'When loading a qwen model, ensure that your model_name or checkpoint contains "qwen2.5". Otherwise, it will be loaded using the "qwen2" config and architecture.'
        )
        model_path = self.checkpoint
        if "qwen2.5" in model_path or "qwen2.5" in self.model_name:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.model_info["model"]["torch_dtype"],
                attn_implementation=self.model_info["model"]["attn_implementation"],
                device_map="auto",
                cache_dir=self.cache_dir,
            )
        elif "qwen3" in model_path or "qwen3" in self.model_name:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.model_info["model"]["torch_dtype"],
                attn_implementation=self.model_info["model"]["attn_implementation"],
                device_map="auto",
                trust_remote_code=True,  # Often needed for newer Qwen models
                cache_dir=self.cache_dir,
            )
        else:
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=self.model_info["model"]["torch_dtype"],
                attn_implementation=self.model_info["model"]["attn_implementation"],
                device_map="auto",
                cache_dir=self.cache_dir,
            )
        self.processor = AutoProcessor.from_pretrained(
            self.model_info["tokenizer"]["path"],
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.model.eval()

        self.device = next(
            self.model.parameters()
        ).device  # If there are multiple GPUs put the model on the first parameters GPU

    def load_images(
        self,
        paths: List[str],
        fps: float = None,
        image_descriptions: list = [],
    ) -> List[Union[torch.Tensor, List[torch.Tensor]]]:
        processed_data = []
        fps = fps if fps is not None else self.model_info.get("fps", 8.0)
        for i, path in enumerate(paths):
            if image_descriptions:
                assert len(image_descriptions) == len(paths), (
                    f"Number of image descriptions ({len(image_descriptions)}) does not match number of paths ({len(paths)})"
                )
                processed_data.append({"type": "text", "text": image_descriptions[i]})

            if path.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv")
            ):  # Video file path
                # video_frames = self.load_video(path, num_frames)
                if fps == "dynamic":
                    processed_data.append(
                        {"type": "video", "video": path, "max_pixels": 360 * 420}
                    )
                else:
                    processed_data.append(
                        {
                            "type": "video",
                            "video": path,
                            "max_pixels": 360 * 420,
                            "fps": fps,
                        }
                    )
            elif path.lower().endswith(".npy"):  # NumPy file
                np_array = np.load(path)
                if np_array.ndim == 3:  # Single image
                    image = Image.fromarray(np_array.astype("uint8"), "RGB")
                    processed_data.append({"type": "image", "image": image})
                elif np_array.ndim == 4:  # Multiple frames
                    frames = [
                        Image.fromarray(frame.astype("uint8"), "RGB")
                        for frame in np_array
                    ]
                    processed_data.append({"type": "video", "video": frames})
                else:
                    raise ValueError(f"Unexpected shape for NumPy array in {path}")
            else:  # Regular image file
                image = Image.open(path).convert("RGB")
                processed_data.append({"type": "image", "image": image})
        return processed_data

    def forward(
        self,
        paths: List[str],
        texts: List[str],
        fps=None,
        question_template: str = 'Does this image show "{}"?',  # "Does this image show \"{}\"? Answer the question with Yes or No",
        answer_template: str = "Yes",
    ) -> torch.Tensor:
        assert len(paths) == len(texts), "Number of paths and texts must match"

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]
        processed_data = self.load_images(paths, fps)

        lm_probs = []
        for data, question, answer in zip(processed_data, questions, answers):
            if self.system_prompt is not None:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [data, {"type": "text", "text": question}],
                    },
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [data, {"type": "text", "text": question}],
                    }
                ]
            if self._is_thinking():
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            else:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,  # Odd that greedy decoding seems necessary for some reason to get the logprobs
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            scores = outputs.scores[0]

            probs = torch.nn.functional.softmax(scores, dim=-1)
            ans_token_id = self.processor.tokenizer.encode(answer)[0]
            lm_prob = probs[0, ans_token_id].item()
            lm_probs.append(lm_prob)
        return torch.tensor(lm_probs)

    def forward_multi_images(
        self,
        paths: List[Union[str, List[str]]],  # Updated type hint
        texts: List[str],
        fps=None,
        question_template: str = 'Does two images show "{}"?',
        answer_template: str = "Yes",
        image_descriptions: list = [],
        parse_answer: bool = False,
    ) -> torch.Tensor:

        assert len(paths) == len(texts), "Number of path groups and texts must match"

        questions = [question_template.format(text) for text in texts]
        answers = [answer_template.format(text) for text in texts]

        lm_probs = []
        output_texts = []

        # We iterate over the inputs explicitly to handle grouping
        for path_input, question, answer in zip(paths, questions, answers):
            # 1. Handle Input Grouping: Ensure we have a list of paths for this specific sample
            if isinstance(path_input, str):
                current_paths = [path_input]  # Single image case
            else:
                current_paths = path_input  # Multi-image case (List[str])

            # 2. Load images for THIS specific prompt only
            # We reuse your existing load_images logic, but only for the current group
            media_items = self.load_images(current_paths, fps, image_descriptions)

            # 3. Construct the content list: [Image1, Image2, ..., Text]
            content = []
            content.extend(media_items)
            content.append({"type": "text", "text": question})

            # 4. Process inputs
            if self.system_prompt is not None:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                ]
            else:
                messages = [{"role": "user", "content": content}]

            if self._is_thinking():
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            else:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # 5. Run Inference (Score Calculation)
            max_new_tokens = 1 if not parse_answer else 2048
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            if not parse_answer:
                scores = outputs.scores[0]
                probs = torch.nn.functional.softmax(scores, dim=-1)
                ans_token_id = self.processor.tokenizer.encode(
                    answer, add_special_tokens=False
                )[0]
                lm_prob = probs[0, ans_token_id].item()
            else:
                # 1. Isolate only the newly generated token IDs
                prompt_length = inputs.input_ids.shape[1]
                gen_ids = outputs.sequences[0][prompt_length:]
                out_text = self.processor.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                output_texts.append(out_text)

                yes_id = self.processor.tokenizer.encode(
                    "Yes", add_special_tokens=False
                )[0]
                no_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[
                    0
                ]
                yes_id_space = self.processor.tokenizer.encode(
                    " Yes", add_special_tokens=False
                )[0]
                no_id_space = self.processor.tokenizer.encode(
                    " No", add_special_tokens=False
                )[0]

                target_ids = torch.tensor(
                    [yes_id, no_id, yes_id_space, no_id_space]
                ).to(self.device)
                matches = torch.isin(gen_ids, target_ids).nonzero(as_tuple=True)[0]

                target_index = -1
                final_decision = None

                if len(matches) > 0:
                    target_index = matches[-1].item()
                    token_id = gen_ids[target_index].item()
                    token_str = self.processor.tokenizer.decode(
                        token_id, skip_special_tokens=True
                    ).strip()
                    final_decision = (
                        "Yes" if token_id in (yes_id, yes_id_space) else "No"
                    )

                # 3. Calculate the score from that specific index
                if target_index != -1:
                    # Get the logits for that specific generation step
                    step_logits = outputs.scores[target_index][0]

                    # # --- Option A: Your Normalization Method ---
                    # step_probs = torch.nn.functional.softmax(step_logits, dim=-1)
                    # lm_prob = (step_probs[yes_token_id] / (step_probs[yes_token_id] + step_probs[no_token_id])).item()

                    # --- Option B: Direct Logit Softmax (Slightly cleaner) ---
                    # You can actually apply softmax directly to just the two isolated logits!
                    yes_no_logits = step_logits[[yes_id, no_id]]
                    normalized_probs = torch.nn.functional.softmax(
                        yes_no_logits, dim=-1
                    )
                    lm_prob = normalized_probs[
                        0
                    ].item()  # Index 0 corresponds to the 'Yes' logit

                    print(f"\nDecision: {final_decision}")
                    print(f"Token '{token_str}' found at step {target_index}.")
                    print(f"Confidence (Probability): {lm_prob:.4f}")
                    print(f"Full generated text:\n{out_text}")
                else:
                    print("Could not locate the isolated Yes/No token.")
                    lm_prob = 0.0
                    final_decision = "Parse Error"

            lm_probs.append(lm_prob)

        if parse_answer:
            return torch.tensor(lm_probs), output_texts

        return torch.tensor(lm_probs)

    def generate(
        self, images: List[str], texts: List[str], fps=None, max_new_tokens: int = 256
    ) -> List[str]:
        assert len(images) == len(texts), "Number of paths and texts must match"

        processed_data = self.load_images(images, fps)

        generated_texts = []
        for data, text in zip(processed_data, texts):
            messages = [
                {"role": "user", "content": [data, {"type": "text", "text": text}]}
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self._is_thinking(),
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                text = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                generated_texts.append(text)

        return generated_texts

    def _is_thinking(self):
        return "thinking" in self.checkpoint.lower()
