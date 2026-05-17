#!/usr/bin/env python3
"""
Analyze ShapeTalk prompts to separate appearance vs structural differences
and convert them to actionable editing instructions.
"""
import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import pandas as pd


# Paths
REPO_ROOT = Path(__file__).parent.resolve()
ANALYZE_INSTRUCTION_PATH = REPO_ROOT / "instruction_prompts" / "analyze_edit_instruction.txt"
IN_CONTEXT_EXAMPLES_PATH = REPO_ROOT / "instruction_prompts" / "in_context_examples.json"
def _shapetalk_csv_path() -> str:
    path = os.environ.get("SHAPETALK_CSV")
    if not path:
        raise ValueError(
            "SHAPETALK_CSV is not set. Export SHAPETALK_CSV to the ShapeTalk CSV path "
            "when using ShapeTalk example helpers."
        )
    return path

# Module-level cache for loaded model
_local_model = None
_local_tokenizer = None
_gemini_client = None


def _load_analyze_instruction() -> str:
    """Load the analysis instruction from the text file."""
    if not ANALYZE_INSTRUCTION_PATH.exists():
        raise FileNotFoundError(f"Instruction file not found: {ANALYZE_INSTRUCTION_PATH}")
    with open(ANALYZE_INSTRUCTION_PATH, 'r') as f:
        return f.read()


def _load_in_context_examples() -> List[Dict]:
    """Load in-context examples from JSON file if it exists."""
    if not IN_CONTEXT_EXAMPLES_PATH.exists():
        return []
    with open(IN_CONTEXT_EXAMPLES_PATH, 'r') as f:
        return json.load(f)


def _format_in_context_examples(examples: List[Dict]) -> str:
    """Format in-context examples as part of the prompt."""
    if not examples:
        return ""
    
    formatted = "\n\n## Additional Examples from User Annotations\n"
    for i, ex in enumerate(examples, 1):
        formatted += f"\nCategory: {ex['category']}\n"
        formatted += f"Prompt: \"{ex['prompt']}\"\n"
        formatted += f"APPEARANCE_DESCRIPTION: {ex.get('appearance_description', '')}\n"
        formatted += f"STRUCTURAL_DESCRIPTION: {ex.get('structural_description', '')}\n"
    
    formatted += "\nNow analyze the following:\n"
    return formatted


def _get_gemini_client():
    """Get Gemini client with API-key auth, falling back to internal GCP auth."""
    global _gemini_client
    
    if _gemini_client is not None:
        return _gemini_client
    
    import os
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        _gemini_client = genai.Client(api_key=api_key)
        return _gemini_client
    
    raise ValueError("No API key provided. Set GOOGLE_API_KEY environment variable.")

def load_local_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> Tuple:
    """
    Load a local text LLM for instruction analysis.
    
    Args:
        model_name: HuggingFace model name (default: Qwen2.5-1.5B-Instruct)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    global _local_model, _local_tokenizer
    
    if _local_model is not None and _local_tokenizer is not None:
        print(f"Using cached model")
        return _local_model, _local_tokenizer
    
    print(f"Loading {model_name}...")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    _local_tokenizer = AutoTokenizer.from_pretrained(model_name)
    _local_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    print(f"Model loaded on device: {_local_model.device}")
    return _local_model, _local_tokenizer


def _analyze_with_gemini(
    prompt: str,
    category: str,
    system_instruction: str,
    gemini_model: str = "gemini-2.5-flash",
) -> dict:
    """
    Query Gemini for prompt analysis.
    
    Args:
        prompt: The ShapeTalk prompt
        category: Shape category
        system_instruction: The full system instruction with examples
        gemini_model: Gemini model to use
    
    Returns:
        Dict with 'text' (response) and 'token_usage' (input/output counts)
    """
    from google import genai
    
    client = _get_gemini_client()
    
    user_query = f"Category: {category}\nPrompt: \"{prompt}\""
    
    config = genai.types.GenerateContentConfig(
        temperature=0.7,
        top_p=0.9,
        max_output_tokens=1024,
        response_modalities=["TEXT"],
        system_instruction=system_instruction,
    )
    
    print(f"Querying Gemini {gemini_model}...")
    
    result = client.models.generate_content(
        model=gemini_model,
        contents=[user_query],
        config=config,
    )
    
    # Extract token usage
    input_tokens = result.usage_metadata.prompt_token_count or 0
    output_tokens = result.usage_metadata.candidates_token_count or 0
    print(f"  Token usage: {input_tokens:,} input, {output_tokens:,} output")
    
    return {
        'text': result.candidates[0].content.parts[0].text,
        'token_usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens},
    }


def _analyze_with_gpt(
    prompt: str,
    category: str,
    system_instruction: str,
    gpt_model: str = "gpt-5.5",
    api_key: Optional[str] = None,
) -> dict:
    """Query OpenAI GPT for prompt analysis (text-only)."""
    import os
    from openai import OpenAI

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    client = OpenAI(api_key=api_key)
    user_query = f"Category: {category}\nPrompt: \"{prompt}\""

    print(f"Querying OpenAI {gpt_model}...")
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_query},
        ],
    )

    token_usage = {'input_tokens': 0, 'output_tokens': 0}
    if hasattr(response, 'usage') and response.usage:
        token_usage['input_tokens'] = getattr(response.usage, 'prompt_tokens', 0) or 0
        token_usage['output_tokens'] = getattr(response.usage, 'completion_tokens', 0) or 0
        print(
            f"  Token usage: {token_usage['input_tokens']:,} input, "
            f"{token_usage['output_tokens']:,} output"
        )

    return {
        'text': response.choices[0].message.content,
        'token_usage': token_usage,
    }


def analyze_shapetalk_prompt(
    prompt: str,
    category: str,
    backend: str = "qwen",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    gemini_model: str = "gemini-2.5-flash",
    gpt_model: str = "gpt-5.5",
    max_new_tokens: int = 1024,
    use_in_context_examples: bool = True,
) -> dict:
    """
    Analyze a ShapeTalk prompt and separate appearance vs structural differences.
    
    Args:
        prompt: The ShapeTalk utterance describing target vs source differences
        category: The shape category (e.g., 'chair', 'table')
        backend: "qwen", "gemini", or "gpt"
        model_name: HuggingFace model to use (for qwen backend)
        gemini_model: Gemini model to use (for gemini backend)
        gpt_model: OpenAI model to use (for gpt backend)
        max_new_tokens: Maximum tokens to generate
        use_in_context_examples: Whether to include user-annotated examples
    
    Returns:
        Dict with appearance/structural differences and instructions
    """
    # Build the system instruction
    system_instruction = _load_analyze_instruction()
    
    # Add in-context examples if available and requested
    if use_in_context_examples:
        examples = _load_in_context_examples()
        if examples:
            # Insert examples before "Now analyze the following:"
            system_instruction = system_instruction.rstrip()
            if system_instruction.endswith("Now analyze the following:"):
                system_instruction = system_instruction[:-len("Now analyze the following:")].rstrip()
            system_instruction += _format_in_context_examples(examples)
    
    # Query the appropriate backend
    token_usage = None
    backend_lower = backend.lower()
    if backend_lower == "gemini":
        gemini_result = _analyze_with_gemini(
            prompt=prompt,
            category=category,
            system_instruction=system_instruction,
            gemini_model=gemini_model,
        )
        response = gemini_result['text']
        token_usage = gemini_result['token_usage']
    elif backend_lower == "gpt":
        gpt_result = _analyze_with_gpt(
            prompt=prompt,
            category=category,
            system_instruction=system_instruction,
            gpt_model=gpt_model,
        )
        response = gpt_result['text']
        token_usage = gpt_result['token_usage']
    else:
        # Local Qwen model
        model, tokenizer = load_local_model(model_name)
        
        user_query = f"Category: {category}\nPrompt: \"{prompt}\""
        
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_query},
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        print(f"Analyzing prompt...")
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        
        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        
        # Calculate token usage for local model
        token_usage = {
            'input_tokens': inputs.input_ids.shape[1],
            'output_tokens': outputs[0].shape[0] - inputs.input_ids.shape[1],
        }
    
    # Parse the response
    result = {
        'raw_response': response,
        'category': category,
        'prompt': prompt,
        'appearance_description': None,
        'structural_description': None,
        'token_usage': token_usage,
    }
    
    # Extract each field - parse line by line for robustness
    for line in response.split('\n'):
        line = line.strip()
        for field in ['APPEARANCE_DESCRIPTION', 'STRUCTURAL_DESCRIPTION']:
            if line.startswith(f'{field}:'):
                value = line[len(f'{field}:'):].strip()
                key = field.lower()
                # Treat 'none', empty quotes, or whitespace as None
                if value.lower() in ('none', '""', "''", ''):
                    result[key] = None
                else:
                    # Remove surrounding quotes if present
                    if (value.startswith('"') and value.endswith('"')) or \
                       (value.startswith("'") and value.endswith("'")):
                        value = value[1:-1]
                    result[key] = value if value else None
                break

    return result


def save_parse_result(parse_result: dict, output_path: Path):
    """Save parse result to a text file."""
    with open(output_path, 'w') as f:
        f.write(f"Edit Instruction: {parse_result['prompt']}\n")
        f.write(f"Category: {parse_result['category']}\n\n")
        f.write(f"Structural Description: {parse_result.get('structural_description', '')}\n")
        f.write(f"Appearance Description: {parse_result.get('appearance_description', '')}\n\n")
        f.write(f"--- Raw Response ---\n{parse_result['raw_response']}\n")
    print(f"  Saved to: {output_path}")


# Keep the old function for backwards compatibility
def analyze_instruction(
    edit_instruction: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens: int = 1024,
    category: str = "shape",
    backend: str = "qwen",
) -> dict:
    """
    Analyze an editing instruction (backwards compatible wrapper).
    
    Args:
        edit_instruction: The raw edit instruction/prompt to analyze
        model_name: HuggingFace model to use
        max_new_tokens: Maximum tokens to generate
        category: Shape category (default: "shape")
        backend: "qwen" for local model or "gemini" for Gemini API
    
    Returns:
        Dict with analysis results
    """
    return analyze_shapetalk_prompt(
        prompt=edit_instruction,
        category=category,
        backend=backend,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )

def get_single_shapetalk_example(category: str = "chair") -> Dict:
    """
    Get a single random ShapeTalk example from the CSV file.
    
    Args:
        category: Shape category to filter by (e.g., 'chair', 'table')
    
    Returns:
        A pandas Series (row) with the sample data
    """
    df = pd.read_csv(_shapetalk_csv_path())
    
    filtered = df[
        (df['source_dataset'] == 'ShapeNet') &
        (df['target_dataset'] == 'ShapeNet') &
        (df['source_object_class'].str.lower() == category.lower()) &
        (df['target_object_class'].str.lower() == category.lower())
    ]
    
    if len(filtered) == 0:
        raise ValueError(f"No ShapeTalk samples found for category: {category}")
    
    # Sample a random row and return it
    return filtered.sample(1).iloc[0]

def test_on_shapetalk(
    num_tests: int = 5,
    category: Optional[str] = None,
    backend: str = "qwen",
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    gemini_model: str = "gemini-2.5-flash",
    max_new_tokens: int = 1024,
    use_in_context_examples: bool = True,
) -> List[Dict]:
    """
    Test the LLM on random ShapeTalk examples.
    
    Args:
        num_tests: Number of examples to test
        category: Filter to specific category (None = all categories)
        backend: "qwen" for local model or "gemini" for Gemini API
        model_name: HuggingFace model to use (for qwen backend)
        gemini_model: Gemini model to use (for gemini backend)
        max_new_tokens: Maximum tokens to generate
        use_in_context_examples: Whether to use annotated examples
    
    Returns:
        List of results
    """
    print(f"Loading ShapeTalk CSV...")
    df = pd.read_csv(_shapetalk_csv_path())
    
    # Filter for ShapeNet dataset
    filtered = df[
        (df['source_dataset'] == 'ShapeNet') &
        (df['target_dataset'] == 'ShapeNet')
    ]
    
    if category:
        filtered = filtered[
            (filtered['source_object_class'].str.lower() == category.lower()) &
            (filtered['target_object_class'].str.lower() == category.lower())
        ]
        print(f"Filtered to category: {category}")
    
    print(f"Available samples: {len(filtered)}")
    
    # Sample random rows
    if len(filtered) < num_tests:
        print(f"Warning: Only {len(filtered)} samples available")
        samples = filtered
    else:
        samples = filtered.sample(n=num_tests)
    
    results = []
    
    for idx, (_, row) in enumerate(samples.iterrows()):
        category_name = row['source_object_class'].lower()
        utterance = row['utterance_0']
        
        print("\n" + "=" * 70)
        print(f"TEST {idx + 1}/{len(samples)}")
        print("=" * 70)
        print(f"Category: {category_name}")
        print(f"Prompt: \"{utterance}\"")
        print("-" * 70)
        
        result = analyze_shapetalk_prompt(
            prompt=utterance,
            category=category_name,
            backend=backend,
            model_name=model_name,
            gemini_model=gemini_model,
            max_new_tokens=max_new_tokens,
            use_in_context_examples=use_in_context_examples,
        )
        
        print(f"\nAPPEARANCE_DESCRIPTION: {result['appearance_description'] or ''}")
        print(f"STRUCTURAL_DESCRIPTION: {result['structural_description'] or ''}")
        
        results.append(result)
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Analyze ShapeTalk prompts to separate appearance vs structural differences'
    )
    parser.add_argument(
        'prompt',
        type=str,
        nargs='?',
        default=None,
        help='The ShapeTalk prompt to analyze (not needed with --test)'
    )
    parser.add_argument(
        '--category',
        type=str,
        default='chair',
        help='Shape category (default: chair)'
    )
    parser.add_argument(
        '--backend',
        type=str,
        default='qwen',
        choices=['qwen', 'gemini'],
        help='Backend to use: qwen (local) or gemini (API) (default: qwen)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2.5-1.5B-Instruct',
        help='HuggingFace model name for qwen backend (default: Qwen/Qwen2.5-1.5B-Instruct)'
    )
    parser.add_argument(
        '--gemini_model',
        type=str,
        default='gemini-2.5-flash',
        help='Gemini model name (default: gemini-2.5-flash)'
    )
    parser.add_argument(
        '--max_tokens',
        type=int,
        default=1024,
        help='Maximum tokens to generate (default: 1024)'
    )
    parser.add_argument(
        '--no_examples',
        action='store_true',
        help='Do not use in-context examples from annotations'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: run on random ShapeTalk examples'
    )
    parser.add_argument(
        '--num_tests',
        type=int,
        default=5,
        help='Number of test examples (default: 5, used with --test)'
    )
    parser.add_argument(
        '--all_categories',
        action='store_true',
        help='Test on all categories, not just --category (used with --test)'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.test:
        # Test mode: run on random ShapeTalk examples
        print("=" * 70)
        print("SHAPETALK LLM TEST MODE")
        print("=" * 70)
        print(f"Backend: {args.backend}")
        if args.backend == "gemini":
            print(f"Gemini model: {args.gemini_model}")
        else:
            print(f"Qwen model: {args.model}")
        print(f"Num tests: {args.num_tests}")
        print(f"Using in-context examples: {not args.no_examples}")
        
        results = test_on_shapetalk(
            num_tests=args.num_tests,
            category=None if args.all_categories else args.category,
            backend=args.backend,
            model_name=args.model,
            gemini_model=args.gemini_model,
            max_new_tokens=args.max_tokens,
            use_in_context_examples=not args.no_examples,
        )
        
        print("\n" + "=" * 70)
        print(f"COMPLETED {len(results)} TESTS")
        print("=" * 70)
    
    else:
        # Single prompt mode
        if not args.prompt:
            print("Error: Please provide a prompt or use --test mode")
            print("Usage: python analyze_instruction.py 'prompt' --category chair")
            print("   or: python analyze_instruction.py --test --num_tests 5")
            print("   or: python analyze_instruction.py --test --backend gemini")
            exit(1)
        
        print("=" * 60)
        print("SHAPETALK PROMPT ANALYSIS")
        print("=" * 60)
        print(f"Backend: {args.backend}")
        print(f"Category: {args.category}")
        print(f"Prompt: \"{args.prompt}\"")
        print("-" * 60)
        
        result = analyze_shapetalk_prompt(
            prompt=args.prompt,
            category=args.category,
            backend=args.backend,
            model_name=args.model,
            gemini_model=args.gemini_model,
            max_new_tokens=args.max_tokens,
            use_in_context_examples=not args.no_examples,
        )
        
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        
        print(f"\nAPPEARANCE_DESCRIPTION: {result['appearance_description'] or ''}")
        print(f"STRUCTURAL_DESCRIPTION: {result['structural_description'] or ''}")
        
        if not any([result['appearance_description'], result['structural_description']]):
            print(f"\n--- Raw Response ---\n{result['raw_response']}")
