import torch
from typing import List
from transformers import AutoModel, AutoProcessor

from .qwen2vl_model import Qwen2VLModel


SAIL_VL_MODELS = {
    "sail-vl-8b": {
        "tokenizer": {
            "path": "BytedanceDouyinContent/SAIL-VL2-8B",
            "trust_remote_code": True,
        },
        "model": {
            "path": "BytedanceDouyinContent/SAIL-VL2-8B",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        },
    },
    "sail-vl-8b-thinking": {
        "tokenizer": {
            "path": "BytedanceDouyinContent/SAIL-VL2-8B-Thinking",
            "trust_remote_code": True,
        },
        "model": {
            "path": "BytedanceDouyinContent/SAIL-VL2-8B-Thinking",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        },
    },
}


class SAILVLModel(Qwen2VLModel):
    """SAIL-VL reuses Qwen2VL's message format but uses AutoModel + direct image passing."""

    def __init__(
        self,
        model_name="qwen2.5-vl-7b",
        device="cuda",
        cache_dir=None,
        checkpoint=None,
        **kwargs,
    ):
        assert model_name in SAIL_VL_MODELS, (
            f"Model {model_name} not found in SAIL_VL_MODELS"
        )
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model_info = SAIL_VL_MODELS[model_name]
        self.checkpoint = checkpoint if checkpoint else self.model_info["model"]["path"]
        self.system_prompt = kwargs.get("system_prompt", None)
        self.load_model()

    def load_model(self):
        model_path = self.checkpoint
        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=self.cache_dir,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=self.model_info["tokenizer"]["trust_remote_code"],
            cache_dir=self.cache_dir,
        )
        self.device = next(self.model.parameters()).device

    def _build_inputs(self, messages, image_list):
        """
        image_list: list of PIL Images (one per image in the messages), or empty.
        Returns tokenized inputs on the correct device/dtype.
        """
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        images = image_list if image_list else None
        inputs = (
            self.processor(
                images=images,
                text=text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            .to(self.device)
            .to(torch.bfloat16)
        )
        return inputs

    def _extract_pil_images(self, messages):
        """Pull PIL Image objects out of a message content list."""
        pil_images = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            for item in msg.get("content", []):
                if item.get("type") == "image":
                    pil_images.append(item["image"])
        return pil_images

    def forward(
        self,
        paths,
        texts,
        fps=None,
        question_template='Does this image show "{}"?',
        answer_template="Yes",
    ):
        assert len(paths) == len(texts)
        questions = [question_template.format(t) for t in texts]
        answers = [answer_template.format(t) for t in texts]
        # load_images from Qwen2VLModel returns content dicts with PIL images embedded
        processed_data = self.load_images(paths, fps)

        lm_probs = []
        for data, question, answer in zip(processed_data, questions, answers):
            # if self._is_thinking():
            #     cot_prompt = r"\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
            # else:
            #     cot_prompt = ""

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

            pil_images = self._extract_pil_images(messages)
            inputs = self._build_inputs(messages, pil_images)

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            scores = outputs.scores[0]
            probs = torch.nn.functional.softmax(scores, dim=-1)
            ans_token_id = self.processor.tokenizer.encode(
                answer, add_special_tokens=False
            )[0]
            lm_probs.append(probs[0, ans_token_id].item())

        return torch.tensor(lm_probs)

    def forward_multi_images(
        self,
        paths,
        texts,
        fps=None,
        question_template='Does two images show "{}"?',
        answer_template="Yes",
        image_descriptions=[],
        parse_answer=False,
    ):

        assert len(paths) == len(texts), "Number of path groups and texts must match"

        questions = [question_template.format(t) for t in texts]
        answers = [answer_template.format(t) for t in texts]

        lm_probs = []
        output_texts = []

        for path_input, question, answer in zip(paths, questions, answers):
            current_paths = [path_input] if isinstance(path_input, str) else path_input
            media_items = self.load_images(current_paths, fps, image_descriptions)

            # if self._is_thinking():
            #     # cot_prompt = r"\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
            #     r"\nYou FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags and must strictly follow the structured steps outlined above, starting explicitly with '1. Visual Context Analysis:', followed by '2. Reasoning Plan', and '3. Step-by-Step Execution'. After closing the </think> tag, provide your '4. Final Conclusion' where the final Yes or No answer MUST BE put in \boxed{}."
            # else:
            #     cot_prompt = ""

            content = list(media_items) + [{"type": "text", "text": question}]
            if self.system_prompt is not None:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                ]
            else:
                messages = [{"role": "user", "content": content}]

            pil_images = self._extract_pil_images(messages)
            max_new_tokens = 2048 if parse_answer else 1
            inputs = self._build_inputs(messages, pil_images)

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
                gen_ids = outputs.sequences[0]
                out_text = self.processor.decode(
                    gen_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                ).strip()
                out_text = out_text.split("<|im_end|>")[0].strip()
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

                if target_index != -1:
                    step_logits = outputs.scores[target_index][0]
                    yes_no_logits = step_logits[[yes_id, no_id]]
                    normalized_probs = torch.nn.functional.softmax(
                        yes_no_logits, dim=-1
                    )
                    lm_prob = normalized_probs[0].item()

                    print(f"\nDecision: {final_decision}")
                    print(f"Token '{token_str}' found at step {target_index}.")
                    print(f"Confidence: {lm_prob:.4f}")
                    print(f"Full generated text:\n{out_text}")
                else:
                    print("Could not locate the isolated Yes/No token.")
                    lm_prob = 0.0

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
                messages, tokenize=False, add_generation_prompt=True
            )
            pil_images = self._extract_pil_images(messages)
            inputs = self._build_inputs(messages, pil_images)

            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens
                )
                text = self.processor.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                text = text.split("<|im_end|>")[0].strip()
                generated_texts.append(text)

        return generated_texts
