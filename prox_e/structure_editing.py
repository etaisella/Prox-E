"""
Structure editing utilities for VLM-based shape manipulation.
"""

import importlib.util
import json
import os
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict

import base64

from PIL import Image, ImageDraw, ImageFont
import torch


# Optional imports - will be loaded when needed
try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


# Module-level cache for local VLM model
_local_vlm_model = None
_local_vlm_processor = None

# Path to VLM instruction files
REPO_ROOT = Path(__file__).parent.resolve()
VLM_INSTRUCTION_PATH = REPO_ROOT / "instruction_prompts" / "vlm_instruction.txt"
VLM_FEEDBACK_INSTRUCTION_PATH = (
    REPO_ROOT / "instruction_prompts" / "vlm_feedback_instruction.txt"
)


def prepare_vlm_input(
    output_folder: Union[str, Path],
    original_filename: str = "original.png",
    view_prefix: str = "",
    original_title: str = "Original Shape",
    grid_title: str = "Abstracted Shape",
    output_original_filename: str = "vlm_original.png",
    output_grid_filename: str = "vlm_abstraction.png",
    skip_original: bool = False,
    font_size: int = 32,
    title_font_size: int = 48,
) -> Tuple[Optional[Path], Path]:
    """
    Prepare images for VLM input from rendered shape views.

    Loads original.png and the four views (front, back, left, right),
    then creates annotated images with titles.

    Args:
        output_folder: Path to folder containing the rendered images
        original_filename: Filename for the original shape image (default: "original.png")
        view_prefix: Prefix for view filenames (default: "" -> "front.png",
                     or e.g. "edited_iter1_" -> "edited_iter1_front.png")
        original_title: Title for the original image (default: "Original Shape")
        grid_title: Title for the grid image (default: "Abstracted Shape")
        output_original_filename: Output filename for original VLM image (default: "vlm_original.png")
        output_grid_filename: Output filename for grid VLM image (default: "vlm_abstraction.png")
        skip_original: If True, skip creating the original image (default: False)
        font_size: Font size for direction labels (default: 32)
        title_font_size: Font size for main titles (default: 48)

    Returns:
        Tuple of (original_vlm_path or None if skipped, grid_vlm_path)
    """
    output_folder = Path(output_folder)

    # Try to load a nice font, fall back to default
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", title_font_size
        )
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", font_size
            )
            title_font = ImageFont.truetype(
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", title_font_size
            )
        except (OSError, IOError):
            font = ImageFont.load_default()
            title_font = font

    # === Create original shape image with title (if not skipped) ===
    original_vlm_path = None
    if not skip_original:
        original_img = Image.open(output_folder / original_filename)
        original_vlm = _add_title_to_image(original_img, original_title, title_font)
        original_vlm_path = output_folder / output_original_filename
        original_vlm.save(original_vlm_path)

    # === Create grid image ===
    # Load the 4 views with prefix
    front_img = Image.open(output_folder / f"{view_prefix}front.png")
    back_img = Image.open(output_folder / f"{view_prefix}back.png")
    left_img = Image.open(output_folder / f"{view_prefix}left.png")
    right_img = Image.open(output_folder / f"{view_prefix}right.png")

    # Grid layout:
    #   front  |  right
    #   -------|-------
    #   left   |  back

    grid_vlm = _create_labeled_grid(
        images=[front_img, right_img, left_img, back_img],
        labels=["Front", "Right", "Left", "Back"],
        main_title=grid_title,
        font=font,
        title_font=title_font,
    )
    grid_vlm_path = output_folder / output_grid_filename
    grid_vlm.save(grid_vlm_path)

    if not skip_original:
        print("VLM input images saved:")
        print(f"  Original:    {original_vlm_path}")
        print(f"  Grid:        {grid_vlm_path}")
    else:
        print(f"VLM grid image saved: {grid_vlm_path}")

    return original_vlm_path, grid_vlm_path


def _add_title_to_image(
    img: Image.Image,
    title: str,
    font: ImageFont.ImageFont,
    padding: int = 20,
    bg_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Add a title above an image."""
    # Calculate title height
    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    bbox = dummy_draw.textbbox((0, 0), title, font=font)
    title_height = bbox[3] - bbox[1] + padding * 2

    # Create new image with space for title
    new_width = img.width
    new_height = img.height + title_height

    result = Image.new("RGBA", (new_width, new_height), bg_color)

    # Paste original image below title area
    if img.mode == "RGBA":
        result.paste(img, (0, title_height), img)
    else:
        result.paste(img, (0, title_height))

    # Draw title
    draw = ImageDraw.Draw(result)
    text_x = (new_width - (bbox[2] - bbox[0])) // 2
    text_y = padding
    draw.text((text_x, text_y), title, font=font, fill=text_color)

    return result


def _create_labeled_grid(
    images: list,
    labels: list,
    main_title: str,
    font: ImageFont.ImageFont,
    title_font: ImageFont.ImageFont,
    padding: int = 10,
    label_padding: int = 5,
    bg_color: Tuple[int, int, int, int] = (255, 255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Create a 2x2 grid of images with labels and a main title."""
    assert len(images) == 4 and len(labels) == 4, "Need exactly 4 images and labels"

    # Get dimensions (assume all images same size)
    img_width = images[0].width
    img_height = images[0].height

    # Calculate label height
    dummy_draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    label_bbox = dummy_draw.textbbox((0, 0), "Front", font=font)
    label_height = label_bbox[3] - label_bbox[1] + label_padding * 2

    # Calculate title height
    title_bbox = dummy_draw.textbbox((0, 0), main_title, font=title_font)
    title_height = title_bbox[3] - title_bbox[1] + padding * 2

    # Calculate total dimensions
    cell_height = label_height + img_height
    grid_width = img_width * 2 + padding * 3
    grid_height = title_height + cell_height * 2 + padding * 3

    # Create result image
    result = Image.new("RGBA", (grid_width, grid_height), bg_color)
    draw = ImageDraw.Draw(result)

    # Draw main title
    title_x = (grid_width - (title_bbox[2] - title_bbox[0])) // 2
    title_y = padding
    draw.text((title_x, title_y), main_title, font=title_font, fill=text_color)

    # Place images in grid with labels
    positions = [
        (padding, title_height + padding),  # top-left: front
        (padding * 2 + img_width, title_height + padding),  # top-right: right
        (padding, title_height + padding * 2 + cell_height),  # bottom-left: left
        (
            padding * 2 + img_width,
            title_height + padding * 2 + cell_height,
        ),  # bottom-right: back
    ]

    for img, label, (x, y) in zip(images, labels, positions):
        # Draw label
        label_bbox = draw.textbbox((0, 0), label, font=font)
        label_width = label_bbox[2] - label_bbox[0]
        label_x = x + (img_width - label_width) // 2
        label_y = y + label_padding
        draw.text((label_x, label_y), label, font=font, fill=text_color)

        # Paste image
        img_y = y + label_height
        if img.mode == "RGBA":
            result.paste(img, (x, img_y), img)
        else:
            result.paste(img, (x, img_y))

    return result


def query_gemini(
    output_folder: Union[str, Path],
    instruction: Optional[str] = None,
    model_name: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> str:
    """
    Query Gemini VLM with the prepared images and abstraction data.

    Sends the original shape image, abstraction grid image, and abstraction JSON
    to Gemini along with a user instruction.

    Args:
        output_folder: Path to folder containing vlm_original.png, vlm_abstraction.png,
                       and abstraction.json
        instruction: The instruction/prompt to send to Gemini. If None, uses a placeholder.
        model_name: Gemini model to use (default: "gemini-2.0-flash")
        api_key: Google API key. If None, reads from GOOGLE_API_KEY environment variable.

    Returns:
        Gemini's response text
    """
    output_folder = Path(output_folder)

    # Configure API key
    if api_key is None:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key is None:
            raise ValueError(
                "No API key provided. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
    genai.configure(api_key=api_key)

    # Load images
    vlm_original_path = output_folder / "vlm_original.png"
    vlm_abstraction_path = output_folder / "vlm_abstraction.png"
    abstraction_json_path = output_folder / "abstraction.json"

    if not vlm_original_path.exists():
        raise FileNotFoundError(f"VLM original image not found: {vlm_original_path}")
    if not vlm_abstraction_path.exists():
        raise FileNotFoundError(
            f"VLM abstraction image not found: {vlm_abstraction_path}"
        )
    if not abstraction_json_path.exists():
        raise FileNotFoundError(f"Abstraction JSON not found: {abstraction_json_path}")

    original_img = Image.open(vlm_original_path)
    abstraction_img = Image.open(vlm_abstraction_path)

    # Load abstraction JSON
    with open(abstraction_json_path, "r") as f:
        abstraction_data = json.load(f)
    abstraction_json_str = json.dumps(abstraction_data, indent=2)

    # Default instruction placeholder
    if instruction is None:
        instruction = """
        # TODO: Write your instruction here
        # 
        # You have access to:
        # 1. The original 3D shape (first image)
        # 2. The abstracted shape from 4 views (second image)  
        # 3. The abstraction JSON data describing each superquadric primitive
        #
        # Example instructions:
        # - "Describe this shape and its parts"
        # - "How would you modify this shape to make it taller?"
        # - "Which primitive should be removed to create a stool from this chair?"
        """

    # Build the prompt
    prompt = f"""
{instruction}

Here is the abstraction data in JSON format. Each entry represents a superquadric primitive with:
- index: primitive ID
- scale: size in x, y, z dimensions
- translation: position in 3D space
- rotation: 3x3 rotation matrix
- exponents: shape parameters (controls roundness/squareness)
- color: RGB color for visualization

```json
{abstraction_json_str}
```
"""

    # Create model and send request
    model = genai.GenerativeModel(model_name)

    print(f"Querying Gemini ({model_name})...")
    response = model.generate_content(
        [
            original_img,
            abstraction_img,
            prompt,
        ]
    )

    # Extract token usage
    # input_tokens = response.usage_metadata.prompt_token_count or 0
    # output_tokens = response.usage_metadata.candidates_token_count or 0
    print("Gemini response received.")
    return response.text


def query_gemini_for_edit(
    output_folder: Union[str, Path],
    edit_instruction: str,
    model_name: str = "gemini-2.0-flash",
    api_key: Optional[str] = None,
) -> Dict:
    """
    Query Gemini to suggest edits to the abstraction based on a natural language instruction.

    This is a convenience wrapper around query_gemini that formats the instruction
    to request structured edit suggestions.

    Args:
        output_folder: Path to folder containing the VLM inputs and abstraction
        edit_instruction: Natural language description of the desired edit
                          (e.g., "make the legs shorter", "remove the backrest")
        model_name: Gemini model to use
        api_key: Google API key

    Returns:
        Dict with Gemini's response (raw text in 'response' key)
    """
    instruction = f"""
You are a 3D shape editing assistant. You are given:
1. An image of an original 3D shape
2. A multi-view image of an abstracted version made of superquadric primitives
3. JSON data describing each primitive's parameters

The user wants to edit this shape. Their instruction is:
"{edit_instruction}"

Please analyze the shape and suggest which primitives should be modified and how.
For each suggested modification, specify:
- Which primitive(s) to modify (by index)
- What parameter(s) to change (scale, translation, rotation, or remove entirely)
- The suggested new values or changes

Be specific and reference the primitive indices from the JSON data.
"""

    response_text = query_gemini(
        output_folder=output_folder,
        instruction=instruction,
        model_name=model_name,
        api_key=api_key,
    )

    return {
        "response": response_text,
        "edit_instruction": edit_instruction,
    }


def _image_to_base64(image_path: Union[str, Path]) -> str:
    """Convert an image file to base64 string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_qwen(
    output_folder: Union[str, Path],
    instruction: Optional[str] = None,
    model_name: str = "qwen-vl-max-latest",
    api_key: Optional[str] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> str:
    """
    Query Qwen VLM with the prepared images and abstraction data.

    Sends the original shape image, abstraction grid image, and abstraction JSON
    to Qwen VL along with a user instruction.

    Args:
        output_folder: Path to folder containing vlm_original.png, vlm_abstraction.png,
                       and abstraction.json
        instruction: The instruction/prompt to send to Qwen. If None, uses a placeholder.
        model_name: Qwen model to use (default: "qwen-vl-max-latest")
                    Other options: "qwen-vl-plus-latest", "qwen2-vl-72b-instruct"
        api_key: Dashscope API key. If None, reads from DASHSCOPE_API_KEY environment variable.
        base_url: API base URL (default: Dashscope OpenAI-compatible endpoint)

    Returns:
        Qwen's response text
    """
    output_folder = Path(output_folder)

    # Configure API key
    if api_key is None:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if api_key is None:
            raise ValueError(
                "No API key provided. Set DASHSCOPE_API_KEY environment variable "
                "or pass api_key parameter."
            )

    # Load image paths
    vlm_original_path = output_folder / "vlm_original.png"
    vlm_abstraction_path = output_folder / "vlm_abstraction.png"
    abstraction_json_path = output_folder / "abstraction.json"

    if not vlm_original_path.exists():
        raise FileNotFoundError(f"VLM original image not found: {vlm_original_path}")
    if not vlm_abstraction_path.exists():
        raise FileNotFoundError(
            f"VLM abstraction image not found: {vlm_abstraction_path}"
        )
    if not abstraction_json_path.exists():
        raise FileNotFoundError(f"Abstraction JSON not found: {abstraction_json_path}")

    # Convert images to base64
    original_b64 = _image_to_base64(vlm_original_path)
    abstraction_b64 = _image_to_base64(vlm_abstraction_path)

    # Load abstraction JSON
    with open(abstraction_json_path, "r") as f:
        abstraction_data = json.load(f)
    abstraction_json_str = json.dumps(abstraction_data, indent=2)

    # Default instruction placeholder
    if instruction is None:
        instruction = """
        # TODO: Write your instruction here
        # 
        # You have access to:
        # 1. The original 3D shape (first image)
        # 2. The abstracted shape from 4 views (second image)  
        # 3. The abstraction JSON data describing each superquadric primitive
        #
        # Example instructions:
        # - "Describe this shape and its parts"
        # - "How would you modify this shape to make it taller?"
        # - "Which primitive should be removed to create a stool from this chair?"
        """

    # Build the text prompt
    text_prompt = f"""
{instruction}

Here is the abstraction data in JSON format. Each entry represents a superquadric primitive with:
- index: primitive ID
- scale: size in x, y, z dimensions
- translation: position in 3D space
- rotation: 3x3 rotation matrix
- exponents: shape parameters (controls roundness/squareness)
- color: RGB color for visualization

```json
{abstraction_json_str}
```
"""

    # Create OpenAI client with Dashscope endpoint
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # Build message with images
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{original_b64}"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{abstraction_b64}"},
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    print(f"Querying Qwen ({model_name})...")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    print("Qwen response received.")
    return response.choices[0].message.content


def query_qwen_for_edit(
    output_folder: Union[str, Path],
    edit_instruction: str,
    model_name: str = "qwen-vl-max-latest",
    api_key: Optional[str] = None,
    base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
) -> Dict:
    """
    Query Qwen to suggest edits to the abstraction based on a natural language instruction.

    This is a convenience wrapper around query_qwen that formats the instruction
    to request structured edit suggestions.

    Args:
        output_folder: Path to folder containing the VLM inputs and abstraction
        edit_instruction: Natural language description of the desired edit
                          (e.g., "make the legs shorter", "remove the backrest")
        model_name: Qwen model to use
        api_key: Dashscope API key
        base_url: API base URL

    Returns:
        Dict with Qwen's response (raw text in 'response' key)
    """
    instruction = f"""
You are a 3D shape editing assistant. You are given:
1. An image of an original 3D shape
2. A multi-view image of an abstracted version made of superquadric primitives
3. JSON data describing each primitive's parameters

The user wants to edit this shape. Their instruction is:
"{edit_instruction}"

Please analyze the shape and suggest which primitives should be modified and how.
For each suggested modification, specify:
- Which primitive(s) to modify (by index)
- What parameter(s) to change (scale, translation, rotation, or remove entirely)
- The suggested new values or changes

Be specific and reference the primitive indices from the JSON data.
"""

    response_text = query_qwen(
        output_folder=output_folder,
        instruction=instruction,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
    )

    return {
        "response": response_text,
        "edit_instruction": edit_instruction,
    }


def load_local_qwen(model_size: str = "4B") -> Tuple:
    """
    Load Qwen3-VL model locally for inference.

    Args:
        model_size: Size of Qwen3-VL model to load (default: "4B")
                    Options: "4B", "8B", etc.

    Returns:
        Tuple of (model, processor)
    """
    global _local_vlm_model, _local_vlm_processor

    if _local_vlm_model is not None:
        print(f"Using cached Qwen3-VL-{model_size} model")
        return _local_vlm_model, _local_vlm_processor

    print(f"Loading Qwen3-VL-{model_size}-Instruct locally...")

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    # Detect flash_attention_2 availability
    _has_fa2 = False
    try:
        _has_fa2 = importlib.util.find_spec("flash_attn") is not None
    except Exception:
        _has_fa2 = False

    _attn_impl = "flash_attention_2" if _has_fa2 else "eager"
    print(f"Using attention implementation: {_attn_impl}")

    # Load model
    _local_vlm_model = Qwen3VLForConditionalGeneration.from_pretrained(
        f"Qwen/Qwen3-VL-{model_size}-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation=_attn_impl,
        device_map="auto",
    )
    print(f"Model loaded on device: {_local_vlm_model.device}")

    # Load processor
    _local_vlm_processor = AutoProcessor.from_pretrained(
        f"Qwen/Qwen3-VL-{model_size}-Instruct"
    )
    print("Processor loaded")

    return _local_vlm_model, _local_vlm_processor


def query_qwen_local(
    output_folder: Union[str, Path],
    instruction: Optional[str] = None,
    model_size: str = "4B",
    max_new_tokens: int = 2048,
) -> str:
    """
    Query Qwen3-VL locally with the prepared images and abstraction data.

    Loads the model on first call (cached for subsequent calls).

    Args:
        output_folder: Path to folder containing vlm_original.png, vlm_abstraction.png,
                       and abstraction.json
        instruction: The instruction/prompt to send. If None, uses a placeholder.
        model_size: Size of Qwen3-VL model (default: "4B")
        max_new_tokens: Maximum tokens to generate (default: 2048)

    Returns:
        Model's response text
    """
    output_folder = Path(output_folder)

    # Load model (cached)
    vlm_model, vlm_processor = load_local_qwen(model_size)

    # Load images
    vlm_original_path = output_folder / "vlm_original.png"
    vlm_abstraction_path = output_folder / "vlm_abstraction.png"
    abstraction_json_path = output_folder / "abstraction.json"

    if not vlm_original_path.exists():
        raise FileNotFoundError(f"VLM original image not found: {vlm_original_path}")
    if not vlm_abstraction_path.exists():
        raise FileNotFoundError(
            f"VLM abstraction image not found: {vlm_abstraction_path}"
        )
    if not abstraction_json_path.exists():
        raise FileNotFoundError(f"Abstraction JSON not found: {abstraction_json_path}")

    original_img = Image.open(vlm_original_path)
    abstraction_img = Image.open(vlm_abstraction_path)

    # Load abstraction JSON
    with open(abstraction_json_path, "r") as f:
        abstraction_data = json.load(f)
    abstraction_json_str = json.dumps(abstraction_data, indent=2)

    # Default instruction placeholder
    if instruction is None:
        instruction = """
        # TODO: Write your instruction here
        # 
        # You have access to:
        # 1. The original 3D shape (first image)
        # 2. The abstracted shape from 4 views (second image)  
        # 3. The abstraction JSON data describing each superquadric primitive
        #
        # Example instructions:
        # - "Describe this shape and its parts"
        # - "How would you modify this shape to make it taller?"
        # - "Which primitive should be removed to create a stool from this chair?"
        """

    # Build the text prompt
    text_prompt = f"""
{instruction}

Here is the abstraction data in JSON format. Each entry represents a superquadric primitive with:
- index: primitive ID
- scale: size in x, y, z dimensions
- translation: position in 3D space
- rotation: 3x3 rotation matrix
- exponents: shape parameters (controls roundness/squareness)
- color: RGB color for visualization

```json
{abstraction_json_str}
```
"""

    # Build messages in Qwen3-VL format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the original 3D shape:"},
                {"type": "image", "image": original_img},
                {
                    "type": "text",
                    "text": "Here is the abstracted shape from multiple views:",
                },
                {"type": "image", "image": abstraction_img},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    print(f"Querying Qwen3-VL-{model_size} locally...")

    # Prepare inputs
    inputs = vlm_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(vlm_model.device)

    # Run inference
    with torch.no_grad():
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode output
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = vlm_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("Qwen response received.")
    return output_text[0]


def query_qwen_local_for_edit(
    output_folder: Union[str, Path],
    edit_instruction: str,
    model_size: str = "4B",
    max_new_tokens: int = 2048,
) -> Dict:
    """
    Query local Qwen3-VL to suggest edits to the abstraction.

    Args:
        output_folder: Path to folder containing the VLM inputs and abstraction
        edit_instruction: Natural language description of the desired edit
                          (e.g., "make the legs shorter", "remove the backrest")
        model_size: Size of Qwen3-VL model
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dict with response text in 'response' key
    """
    instruction = f"""
You are a 3D shape editing assistant. You are given:
1. An image of an original 3D shape
2. A multi-view image of an abstracted version made of superquadric primitives
3. JSON data describing each primitive's parameters

The user wants to edit this shape. Their instruction is:
"{edit_instruction}"

Please analyze the shape and suggest which primitives should be modified and how.
For each suggested modification, specify:
- Which primitive(s) to modify (by index)
- What parameter(s) to change (scale, translation, rotation, or remove entirely)
- The suggested new values or changes

Be specific and reference the primitive indices from the JSON data.
"""

    response_text = query_qwen_local(
        output_folder=output_folder,
        instruction=instruction,
        model_size=model_size,
        max_new_tokens=max_new_tokens,
    )

    return {
        "response": response_text,
        "edit_instruction": edit_instruction,
    }


def _load_vlm_instruction() -> str:
    """Load the VLM instruction from the text file."""
    if not VLM_INSTRUCTION_PATH.exists():
        raise FileNotFoundError(
            f"VLM instruction file not found: {VLM_INSTRUCTION_PATH}"
        )
    with open(VLM_INSTRUCTION_PATH, "r") as f:
        return f.read()


def _parse_json_from_response(response_text: str) -> Optional[List[Dict]]:
    """
    Parse JSON array from VLM response text.
    Looks for JSON in code blocks or raw JSON array.
    """
    import re

    # Try to find JSON in code block first
    json_block_pattern = r"```json\s*([\s\S]*?)\s*```"
    match = re.search(json_block_pattern, response_text)

    if match:
        json_str = match.group(1).strip()
    else:
        # Try to find raw JSON array
        array_pattern = r"\[\s*\{[\s\S]*\}\s*\]"
        match = re.search(array_pattern, response_text)
        if match:
            json_str = match.group(0)
        else:
            return None

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON: {e}")
        return None


def _get_gemini_client():
    """Get Gemini client using GOOGLE_API_KEY only."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is not set. Export your Google API key to use Gemini "
            "(see README VLM Setup)."
        )
    from google import genai

    return genai.Client(api_key=api_key)


# === Context Caching for Gemini ===


def _query_gemini_initial_edit(
    original_img_path: Union[str, Path],
    abstraction_img_path: Union[str, Path],
    system_instruction: str,
    abstraction_json_str: str,
    edit_instruction: str,
    model_name: str = "gemini-2.0-flash",
    use_cache: bool = True,
) -> Dict:
    """
    Query Gemini for initial edit, optionally creating a context cache for reuse.
    """
    from google.genai import types
    from PIL import Image

    client = _get_gemini_client()
    original_img = Image.open(original_img_path)
    abstraction_img = Image.open(abstraction_img_path)

    # --- 1. Prepare Content for Cache or Direct Call ---
    # I've restored the missing line here
    cache_contents = [
        "Here is the original 3D shape:",
        original_img,
        "Here is the abstracted shape from multiple views:",
        abstraction_img,
        f"""Here is the abstraction data in JSON format:
```json
{abstraction_json_str}
```

## Edit Instruction
{edit_instruction}

Please analyze the shape and provide the edited JSON.""",
    ]

    cache = None

    # --- 2. Try Creating Cache ---
    if use_cache:
        print("Creating Gemini context cache...")
        try:
            cache = client.caches.create(
                model=model_name,
                config=types.CreateCachedContentConfig(
                    display_name="tredit_edit_context",
                    system_instruction=system_instruction,
                    contents=cache_contents,
                    # FIX: ttl must be a string "30m" or "1800s" for google-genai
                    ttl="1800s",
                ),
            )
            print(f"  Cache created: {cache.name}")

            print(f"Querying Gemini {model_name} (using cache)...")

            # Since system_instruction and cache_contents are in the cache,
            # we only send the *trigger* for the edit here.
            result = client.models.generate_content(
                model=model_name,
                contents="Please analyze the shape and provide the edited JSON.",
                config=types.GenerateContentConfig(
                    temperature=1,
                    top_p=0.95,
                    max_output_tokens=32768,
                    cached_content=cache.name,  # <--- Pass the cache name here
                ),
            )

        except Exception as e:
            print(f"  Cache creation or usage failed: {e}")
            print("  Falling back to non-cached query...")
            cache = None
            # If cache fails, we fall through to the block below

    # --- 3. Non-Cached Fallback ---
    if not use_cache or cache is None:
        # Reconstruct full context + System Instruction for direct call
        # We manually prepend the system instruction since we aren't using the cache config
        full_contents = (
            [system_instruction]
            + cache_contents
            + ["\nPlease analyze the shape and provide the edited JSON."]
        )

        config = types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=32768,
            response_modalities=["TEXT"],
        )
        print(f"Querying Gemini {model_name} (Direct)...")
        result = client.models.generate_content(
            model=model_name, contents=full_contents, config=config
        )

    # --- 4. Extract Response & Token Usage ---
    response_text = ""
    if result.candidates and result.candidates[0].content.parts:
        for part in result.candidates[0].content.parts:
            if part.text:
                response_text += part.text

    token_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    if result.usage_metadata:
        token_usage["input_tokens"] = result.usage_metadata.prompt_token_count or 0
        token_usage["output_tokens"] = result.usage_metadata.candidates_token_count or 0
        token_usage["cached_tokens"] = (
            result.usage_metadata.cached_content_token_count or 0
        )

        print(
            f"  Tokens: {token_usage['input_tokens']} in "
            f"({token_usage['cached_tokens']} cached), "
            f"{token_usage['output_tokens']} out"
        )

    return {
        "text": response_text,
        "token_usage": token_usage,
        "cache_name": cache.name if cache else None,
        "client": client,
    }


def _query_gemini_verify(
    cached_model,
    edited_img_path: Union[str, Path],
    previous_response: str,
    model_name: str = "gemini-2.0-flash",
) -> Dict:
    """Query Gemini to verify/fix an edit using cached context."""

    edited_img = Image.open(edited_img_path)
    prompt = f"""## Your Previous Response
{previous_response}

## Rendered Result (your edited shape from multiple views)
{_load_vlm_verify_instruction()}"""

    print(f"Querying Gemini {model_name} for verify (using cache)...")
    result = cached_model.generate_content(
        [edited_img, prompt],
        generation_config={"temperature": 1, "top_p": 0.95, "max_output_tokens": 32768},
    )

    # Extract response text
    response_text = ""
    if hasattr(result, "text"):
        response_text = result.text
    elif result.candidates:
        for part in result.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                response_text += part.text

    # Extract token usage
    token_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    if hasattr(result, "usage_metadata") and result.usage_metadata:
        token_usage["input_tokens"] = (
            getattr(result.usage_metadata, "prompt_token_count", 0) or 0
        )
        token_usage["output_tokens"] = (
            getattr(result.usage_metadata, "candidates_token_count", 0) or 0
        )
        token_usage["cached_tokens"] = (
            getattr(result.usage_metadata, "cached_content_token_count", 0) or 0
        )
        print(
            f"  Tokens: {token_usage['input_tokens']} in ({token_usage['cached_tokens']} cached), {token_usage['output_tokens']} out"
        )

    return {"text": response_text, "token_usage": token_usage}


def _load_vlm_verify_instruction() -> str:
    """Load the VLM verify instruction from the text file."""
    verify_path = REPO_ROOT / "instruction_prompts" / "vlm_verify_instruction.txt"
    if not verify_path.exists():
        raise FileNotFoundError(f"VLM verify instruction file not found: {verify_path}")
    with open(verify_path, "r") as f:
        return f.read()


def _query_gemini_for_edit(
    original_img_path: Union[str, Path],
    abstraction_img_path: Union[str, Path],
    text_prompt: str,
    model_name: str = "gemini-2.5-pro",
) -> Dict:
    """
    Query Gemini model for shape editing.

    Args:
        original_img_path: Path to original shape image
        abstraction_img_path: Path to abstraction grid image
        text_prompt: The full prompt including instruction and JSON
        model_name: Model to use (default: "gemini-2.5-pro")

    Returns:
        Dict with 'text' (response text) and 'token_usage' (input/output token counts)
    """
    from google import genai

    client = _get_gemini_client()

    # Load images
    original_img = Image.open(original_img_path)
    abstraction_img = Image.open(abstraction_img_path)

    # Generation config (text only output)
    config = genai.types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=32768,
        response_modalities=["TEXT"],
    )

    print(f"Querying Gemini {model_name}...")

    result = client.models.generate_content(
        model=model_name,
        contents=[
            "Here is the original 3D shape:",
            original_img,
            "Here is the abstracted shape from multiple views:",
            abstraction_img,
            text_prompt,
        ],
        config=config,
    )

    # Extract text response
    response_text = ""
    if result.candidates and result.candidates[0].content:
        for part in result.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                response_text += part.text

    # Extract token usage
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    if hasattr(result, "usage_metadata") and result.usage_metadata:
        token_usage["input_tokens"] = (
            getattr(result.usage_metadata, "prompt_token_count", 0) or 0
        )
        token_usage["output_tokens"] = (
            getattr(result.usage_metadata, "candidates_token_count", 0) or 0
        )
        print(
            f"  Token usage: {token_usage['input_tokens']} input, {token_usage['output_tokens']} output"
        )

    return {"text": response_text, "token_usage": token_usage}


def _query_gpt_for_edit(
    original_img_path: Union[str, Path],
    abstraction_img_path: Union[str, Path],
    text_prompt: str,
    api_key: Optional[str] = None,
    model_name: str = "gpt-5.5",
) -> Dict:
    """
    Query OpenAI GPT model for shape editing.

    Args:
        original_img_path: Path to original shape image
        abstraction_img_path: Path to abstraction grid image
        text_prompt: The full prompt including instruction and JSON
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        model_name: Model to use (default: "o4-mini" - latest thinking model)

    Returns:
        Dict with 'text' (response text) and 'token_usage' (input/output token counts)
    """
    from openai import OpenAI as OpenAIClient

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

    client = OpenAIClient(api_key=api_key)

    # Convert images to base64
    original_b64 = _image_to_base64(original_img_path)
    abstraction_b64 = _image_to_base64(abstraction_img_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the original 3D shape:"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{original_b64}"},
                },
                {
                    "type": "text",
                    "text": "Here is the abstracted shape from multiple views:",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{abstraction_b64}"},
                },
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    print(f"Querying OpenAI {model_name}...")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    # Extract token usage
    token_usage = {"input_tokens": 0, "output_tokens": 0}
    if hasattr(response, "usage") and response.usage:
        token_usage["input_tokens"] = getattr(response.usage, "prompt_tokens", 0) or 0
        token_usage["output_tokens"] = (
            getattr(response.usage, "completion_tokens", 0) or 0
        )
        print(
            f"  Token usage: {token_usage['input_tokens']} input, {token_usage['output_tokens']} output"
        )

    return {"text": response.choices[0].message.content, "token_usage": token_usage}


def _query_qwen_for_edit(
    original_img: Image.Image,
    abstraction_img: Image.Image,
    text_prompt: str,
    model_size: str = "4B",
    max_new_tokens: int = 4096,
) -> Dict:
    """
    Query local Qwen3-VL model for shape editing.

    Args:
        original_img: PIL Image of original shape
        abstraction_img: PIL Image of abstraction grid
        text_prompt: The full prompt including instruction and JSON
        model_size: Size of Qwen3-VL model
        max_new_tokens: Maximum tokens to generate

    Returns:
        Dict with 'text' (response text) and 'token_usage' (input/output token counts)
    """
    vlm_model, vlm_processor = load_local_qwen(model_size)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the original 3D shape:"},
                {"type": "image", "image": original_img},
                {
                    "type": "text",
                    "text": "Here is the abstracted shape from multiple views:",
                },
                {"type": "image", "image": abstraction_img},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]

    print(f"Querying Qwen3-VL-{model_size}...")

    inputs = vlm_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(vlm_model.device)

    with torch.no_grad():
        generated_ids = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = vlm_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    # Calculate token usage for local Qwen
    token_usage = {
        "input_tokens": inputs.input_ids.shape[1],
        "output_tokens": generated_ids[0].shape[0] - inputs.input_ids.shape[1],
    }
    print(
        f"  Token usage: {token_usage['input_tokens']} input, {token_usage['output_tokens']} output"
    )

    return {"text": output_text, "token_usage": token_usage}


def _build_edit_prompt(
    system_instruction: str,
    abstraction_json_str: str,
    edit_instruction: str,
) -> str:
    """
    Build the text prompt for VLM shape editing.

    Args:
        system_instruction: The system instruction loaded from file
        abstraction_json_str: JSON string of the abstraction data
        edit_instruction: The edit instruction from the user

    Returns:
        Formatted text prompt string
    """
    return f"""
{system_instruction}

## Current Shape JSON
```json
{abstraction_json_str}
```

## Edit Instruction
{edit_instruction}

Please analyze the shape and provide the edited JSON.
"""


def _query_vlm_for_edit(
    vlm: str,
    original_img_path: Union[str, Path],
    abstraction_img_path: Union[str, Path],
    text_prompt: str,
    model_size: str = "4B",
    max_new_tokens: int = 4096,
    gemini_model: str = "gemini-2.5-pro",
    gpt_model: str = "gpt-5.5",
    api_key: Optional[str] = None,
) -> Dict:
    """
    Unified VLM query function for shape editing.

    Dispatches to the appropriate backend (Gemini, GPT, or Qwen) based on vlm parameter.

    Args:
        vlm: Which VLM to use - "gemini", "gpt", or "qwen"
        original_img_path: Path to original shape image
        abstraction_img_path: Path to abstraction grid image
        text_prompt: The full prompt including instruction and JSON
        model_size: Size of Qwen3-VL model if using qwen
        max_new_tokens: Maximum tokens to generate for qwen
        gemini_model: Gemini model name if using gemini
        gpt_model: GPT model name if using gpt
        api_key: API key for GPT (reads from env if None)

    Returns:
        Dict with 'text' (response text) and 'token_usage' (input/output token counts)
    """
    vlm_lower = vlm.lower()

    if vlm_lower == "gemini":
        return _query_gemini_for_edit(
            original_img_path=original_img_path,
            abstraction_img_path=abstraction_img_path,
            text_prompt=text_prompt,
            model_name=gemini_model,
        )
    elif vlm_lower == "gpt":
        return _query_gpt_for_edit(
            original_img_path=original_img_path,
            abstraction_img_path=abstraction_img_path,
            text_prompt=text_prompt,
            api_key=api_key,
            model_name=gpt_model,
        )
    elif vlm_lower == "qwen":
        original_img = Image.open(original_img_path)
        abstraction_img = Image.open(abstraction_img_path)
        return _query_qwen_for_edit(
            original_img=original_img,
            abstraction_img=abstraction_img,
            text_prompt=text_prompt,
            model_size=model_size,
            max_new_tokens=max_new_tokens,
        )
    else:
        raise ValueError(f"Unknown VLM: {vlm}. Use 'gemini', 'gpt', or 'qwen'.")


def edit_shape_with_vlm(
    output_folder: Union[str, Path],
    edit_instruction: str,
    vlm: str = "gemini",
    model_size: str = "4B",
    max_new_tokens: int = 4096 * 4,
    gemini_model: str = "gemini-2.5-pro",
    gpt_model: str = "gpt-5.5",
    api_key: Optional[str] = None,
    max_iterations: int = 1,
    use_gemini_cache: bool = False,
) -> Dict:
    """
    Edit a shape using VLM based on a natural language instruction.

    This function:
    1. Loads the VLM instruction template
    2. Queries the selected VLM (Gemini, GPT, or Qwen) with the edit instruction
    3. Parses the updated JSON from the response
    4. If max_iterations > 1, shows the result back to VLM for feedback
    5. Repeats until VLM approves or max_iterations reached
    6. Renders and saves all outputs

    Args:
        output_folder: Path to folder containing the VLM inputs and abstraction
        edit_instruction: Natural language description of the desired edit
                          (e.g., "make the seat twice as thick", "remove the backrest")
        vlm: Which VLM to use: "gemini", "gpt", or "qwen" (default: "gemini")
        model_size: Size of Qwen3-VL model if using qwen (default: "4B")
        max_new_tokens: Maximum tokens to generate for qwen (default: 4096)
        gemini_model: Gemini model id if using gemini (default: "gemini-2.5-pro")
        gpt_model: OpenAI model id if using gpt (default: "gpt-5.5")
        api_key: API key for GPT (reads from OPENAI_API_KEY env var if None)
        max_iterations: Maximum feedback iterations (default: 1 = no feedback loop)
        use_gemini_cache: Whether to use Gemini context caching for multi-iteration
                          editing (default: False, requires special permissions)

    Returns:
        Dict with keys:
            - 'success': Whether editing completed successfully (always True if max_iterations=1)
            - 'iterations': Number of iterations performed
            - 'response': Final VLM response text
            - 'edited_abstraction': Final parsed JSON list (or None if parsing failed)
            - 'edit_instruction': The edit instruction used
            - 'history': List of all responses (only if max_iterations > 1)
    """
    output_folder = Path(output_folder)

    # Load image paths
    vlm_original_path = output_folder / "vlm_original.png"
    vlm_abstraction_path = output_folder / "vlm_abstraction.png"
    abstraction_json_path = output_folder / "abstraction.json"

    if not vlm_original_path.exists():
        raise FileNotFoundError(f"VLM original image not found: {vlm_original_path}")
    if not vlm_abstraction_path.exists():
        raise FileNotFoundError(
            f"VLM abstraction image not found: {vlm_abstraction_path}"
        )
    if not abstraction_json_path.exists():
        raise FileNotFoundError(f"Abstraction JSON not found: {abstraction_json_path}")

    # Load original abstraction JSON
    with open(abstraction_json_path, "r") as f:
        original_abstraction = json.load(f)

    # Load system instruction
    system_instruction = _load_vlm_instruction()

    history = []
    current_abstraction = None
    final_response = None
    success = False

    # Track token usage across all VLM calls
    token_usage_log = []  # List of dicts with iteration, type, input_tokens, output_tokens, cached_tokens
    total_input_tokens = 0
    total_output_tokens = 0
    total_cached_tokens = 0

    # For Gemini context caching (disabled by default, requires special permissions)
    gemini_cache = None
    gemini_cached_model = None
    # Only use cache if explicitly enabled AND using gemini with multiple iterations
    use_gemini_cache = (
        use_gemini_cache and vlm.lower() == "gemini" and max_iterations > 1
    )

    # Track actual loop iterations
    actual_loop_iterations = 0

    try:
        from prox_e.abstraction import mesh_from_abstraction, render_multiview

        # === STEP 1: Initial Edit ===
        print(f"\n{'=' * 80}")
        print("INITIAL EDIT")
        print(f"{'=' * 80}")

        abstraction_json_str = json.dumps(original_abstraction, indent=2)
        print(f"Edit instruction: {edit_instruction}")

        # Build the edit prompt
        text_prompt = _build_edit_prompt(
            system_instruction=system_instruction,
            abstraction_json_str=abstraction_json_str,
            edit_instruction=edit_instruction,
        )

        # Query the selected VLM for initial edit
        if vlm.lower() == "gemini" and use_gemini_cache:
            vlm_result = _query_gemini_initial_edit(
                original_img_path=vlm_original_path,
                abstraction_img_path=vlm_abstraction_path,
                system_instruction=system_instruction,
                abstraction_json_str=abstraction_json_str,
                edit_instruction=edit_instruction,
                model_name=gemini_model,
                use_cache=True,
            )
            gemini_cache = vlm_result.get("cache")
            gemini_cached_model = vlm_result.get("cached_model")
        else:
            vlm_result = _query_vlm_for_edit(
                vlm=vlm,
                original_img_path=vlm_original_path,
                abstraction_img_path=vlm_abstraction_path,
                text_prompt=text_prompt,
                model_size=model_size,
                max_new_tokens=max_new_tokens,
                gemini_model=gemini_model,
                gpt_model=gpt_model,
                api_key=api_key,
            )

        output_text = vlm_result["text"]
        token_usage = vlm_result["token_usage"]

        # Log token usage
        cached_tokens = token_usage.get("cached_tokens", 0)
        token_usage_log.append(
            {
                "iteration": 0,
                "type": "initial_edit",
                "input_tokens": token_usage["input_tokens"],
                "output_tokens": token_usage["output_tokens"],
                "cached_tokens": cached_tokens,
            }
        )
        total_input_tokens += token_usage["input_tokens"]
        total_output_tokens += token_usage["output_tokens"]
        total_cached_tokens += cached_tokens

        print("VLM response received.")

        # Save initial edit response
        response_path = output_folder / "vlm_edit_response.txt"
        with open(response_path, "w") as f:
            f.write(f"Edit Instruction: {edit_instruction}\n")
            f.write("=" * 80 + "\n\n")
            f.write(output_text)
        print(f"Saved VLM response to: {response_path}")

        # Parse JSON from initial edit
        current_abstraction = _parse_json_from_response(output_text)
        final_response = output_text

        if current_abstraction is None:
            print("Warning: Could not parse JSON from VLM response")
            history.append(
                {
                    "iteration": 0,
                    "type": "initial_edit",
                    "status": "parse_error",
                    "response": output_text,
                }
            )
        else:
            print(f"Parsed initial edit ({len(current_abstraction)} primitives)")
            history.append(
                {
                    "iteration": 0,
                    "type": "initial_edit",
                    "status": "ok",
                    "response": output_text,
                }
            )
            actual_loop_iterations = 1

            # If only 1 iteration requested, we're done (no feedback loop)
            if max_iterations == 1:
                success = True
            else:
                # === STEP 2: Feedback + Fix Loop ===
                # Each iteration: render current → get feedback+fix → use fix for next iteration
                for iteration in range(1, max_iterations + 1):
                    actual_loop_iterations = iteration

                    print(f"\n{'=' * 80}")
                    print(f"FEEDBACK+FIX ITERATION {iteration}/{max_iterations}")
                    print(f"{'=' * 80}")

                    # Render the current abstraction
                    print(
                        f"Rendering current abstraction ({len(current_abstraction)} primitives)..."
                    )
                    edited_mesh = mesh_from_abstraction(
                        current_abstraction, resolution=30
                    )
                    edited_obj_path = output_folder / f"edited_iter{iteration}.obj"
                    edited_mesh.export(str(edited_obj_path))
                    render_multiview(
                        edited_obj_path,
                        output_folder,
                        prefix=f"edited_iter{iteration}_",
                    )

                    _, edited_grid_path = prepare_vlm_input(
                        output_folder=output_folder,
                        view_prefix=f"edited_iter{iteration}_",
                        grid_title=f"Edited Shape (Iteration {iteration})",
                        output_grid_filename=f"vlm_edited_iter{iteration}.png",
                        skip_original=True,
                    )

                    # Query VLM for feedback+fix
                    print("Getting VLM feedback+fix...")

                    if vlm.lower() == "gemini" and gemini_cached_model:
                        feedback_result = _query_gemini_verify(
                            cached_model=gemini_cached_model,
                            edited_img_path=edited_grid_path,
                            previous_response=final_response,
                            model_name=gemini_model,
                        )
                    else:
                        feedback_result = _query_vlm_for_feedback(
                            vlm=vlm,
                            original_img_path=vlm_original_path,
                            abstraction_img_path=vlm_abstraction_path,
                            edited_img_path=edited_grid_path,
                            edit_instruction=edit_instruction,
                            previous_reasoning=final_response,
                            current_json=current_abstraction,
                            original_json=original_abstraction,
                            model_size=model_size,
                            max_new_tokens=max_new_tokens,
                            gemini_model=gemini_model,
                            gpt_model=gpt_model,
                            api_key=api_key,
                        )

                    feedback_response = feedback_result["text"]
                    feedback_token_usage = feedback_result["token_usage"]

                    cached_tokens = feedback_token_usage.get("cached_tokens", 0)
                    token_usage_log.append(
                        {
                            "iteration": iteration,
                            "type": "feedback_fix",
                            "input_tokens": feedback_token_usage["input_tokens"],
                            "output_tokens": feedback_token_usage["output_tokens"],
                            "cached_tokens": cached_tokens,
                        }
                    )
                    total_input_tokens += feedback_token_usage["input_tokens"]
                    total_output_tokens += feedback_token_usage["output_tokens"]
                    total_cached_tokens += cached_tokens

                    # Save feedback response
                    feedback_path = output_folder / f"vlm_feedback_iter{iteration}.txt"
                    with open(feedback_path, "w") as f:
                        f.write(f"Iteration: {iteration}\n")
                        f.write(f"Edit Instruction: {edit_instruction}\n")
                        f.write("=" * 80 + "\n\n")
                        f.write(feedback_response)
                    print(f"Saved feedback to: {feedback_path}")

                    # Check if VLM says we're done
                    if _check_if_done(feedback_response):
                        print("\n*** VLM indicates editing is COMPLETE ***")
                        success = True
                        history.append(
                            {
                                "iteration": iteration,
                                "type": "feedback_fix",
                                "status": "done",
                                "response": feedback_response,
                            }
                        )
                        break

                    # Parse the fix from feedback response
                    new_abstraction = _parse_json_from_response(feedback_response)

                    if new_abstraction is None:
                        print(
                            "Warning: Could not parse JSON from feedback, keeping previous version"
                        )
                        history.append(
                            {
                                "iteration": iteration,
                                "type": "feedback_fix",
                                "status": "parse_error",
                                "response": feedback_response,
                            }
                        )
                        # Continue to next iteration with unchanged abstraction
                    else:
                        print(
                            f"Parsed fix from feedback ({len(new_abstraction)} primitives)"
                        )
                        current_abstraction = new_abstraction
                        final_response = (
                            feedback_response  # Use this as context for next iteration
                        )
                        history.append(
                            {
                                "iteration": iteration,
                                "type": "feedback_fix",
                                "status": "fixed",
                                "response": feedback_response,
                            }
                        )

                        # Save iteration JSON
                        iter_json_path = (
                            output_folder / f"edited_abstraction_iter{iteration}.json"
                        )
                        with open(iter_json_path, "w") as f:
                            json.dump(current_abstraction, f, indent=2)
                        print(f"Saved iteration JSON to: {iter_json_path}")

        # Save final results
        edited_json_path = None
        edited_render_path = None

        if current_abstraction is not None:
            # Save final JSON (consistent naming)
            edited_json_path = output_folder / "edited_abstraction_final.json"
            with open(edited_json_path, "w") as f:
                json.dump(current_abstraction, f, indent=2)
            print(f"Saved edited abstraction to: {edited_json_path}")

            # Render final result
            try:
                from prox_e.abstraction import mesh_from_abstraction
                from prox_e.utils import render_obj_with_blender

                print("Generating mesh from edited abstraction...")
                edited_mesh = mesh_from_abstraction(current_abstraction, resolution=30)
                edited_obj_path = output_folder / "edited_final.obj"
                edited_render_path = output_folder / "edited_final_render.png"
                edited_mesh.export(str(edited_obj_path))
                print(f"Exported edited mesh to: {edited_obj_path}")
                render_obj_with_blender(str(edited_obj_path), str(edited_render_path))
                print(f"Saved edited render to: {edited_render_path}")
            except Exception as e:
                print(f"Warning: Failed to render edited shape: {e}")

        # Save history
        if history:
            history_path = output_folder / "edit_history.json"
            with open(history_path, "w") as f:
                json.dump(
                    [
                        {
                            "iteration": h["iteration"],
                            "type": h.get("type", "unknown"),
                            "status": h["status"],
                            "response_preview": h["response"][:500] + "..."
                            if len(h["response"]) > 500
                            else h["response"],
                        }
                        for h in history
                    ],
                    f,
                    indent=2,
                )

        # Print token usage summary
        print("\n--- VLM Token Usage Summary ---")
        print(f"  Total input tokens:  {total_input_tokens:,}")
        print(f"  Total output tokens: {total_output_tokens:,}")
        if total_cached_tokens > 0:
            print(f"  Cached tokens:       {total_cached_tokens:,} (reduced cost)")
        print(f"  Total tokens:        {total_input_tokens + total_output_tokens:,}")
        for entry in token_usage_log:
            cached_str = (
                f", {entry.get('cached_tokens', 0):,} cached"
                if entry.get("cached_tokens", 0) > 0
                else ""
            )
            print(
                f"    Iter {entry['iteration']} ({entry['type']}): {entry['input_tokens']:,} in, {entry['output_tokens']:,} out{cached_str}"
            )

        return {
            "success": success,
            "iterations": actual_loop_iterations,
            "response": final_response,
            "edited_abstraction": current_abstraction,
            "edit_instruction": edit_instruction,
            "edited_json_path": edited_json_path if current_abstraction else None,
            "edited_render_path": edited_render_path if current_abstraction else None,
            "history": history,
            "token_usage": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_cached_tokens": total_cached_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
                "log": token_usage_log,
            },
        }

    finally:
        # Clean up Gemini cache if it was created
        if gemini_cache:
            print("\nCleaning up Gemini context cache...")
            try:
                gemini_cache.delete()
                print(f"Deleted cache: {gemini_cache.name}")
            except Exception as e:
                print(f"Warning: Failed to delete cache: {e}")


def _load_vlm_feedback_instruction() -> str:
    """Load the VLM feedback instruction from the text file."""
    if not VLM_FEEDBACK_INSTRUCTION_PATH.exists():
        raise FileNotFoundError(
            f"VLM feedback instruction file not found: {VLM_FEEDBACK_INSTRUCTION_PATH}"
        )
    with open(VLM_FEEDBACK_INSTRUCTION_PATH, "r") as f:
        return f.read()


def _values_differ(a, b, tol: float = 1e-5) -> bool:
    """Return True if two JSON-serializable values differ (with float tolerance)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) > tol
    return a != b


def _format_scalar_list_diff(field: str, before, after, tol: float = 1e-5) -> List[str]:
    """One line per changed component in a numeric list (scale, translation, etc.)."""
    lines: List[str] = []
    if not isinstance(before, list) or not isinstance(after, list):
        if _values_differ(before, after, tol):
            lines.append(f"  - {field}: {before!r} -> {after!r}")
        return lines
    n = max(len(before), len(after))
    labels = ["x", "y", "z", "w"]
    for i in range(n):
        b = before[i] if i < len(before) else None
        a = after[i] if i < len(after) else None
        if _values_differ(b, a, tol):
            axis = labels[i] if i < len(labels) else str(i)
            lines.append(f"  - {field}[{axis}]: {b} -> {a}")
    return lines


def _format_abstraction_diff(
    original: List[Dict],
    current: List[Dict],
    tol: float = 1e-5,
) -> str:
    """
    Human-readable summary of parameter changes between two abstraction JSON lists.
    Used in the VLM feedback prompt so the model does not have to diff JSON from memory.
    """

    def _by_index(primitives: List[Dict]) -> Dict:
        out: Dict = {}
        for i, p in enumerate(primitives):
            out[p.get("index", i)] = p
        return out

    orig_map = _by_index(original)
    curr_map = _by_index(current)
    all_indices = sorted(set(orig_map.keys()) | set(curr_map.keys()))

    blocks: List[str] = []
    for idx in all_indices:
        if idx not in orig_map:
            blocks.append(f"Primitive {idx}: ADDED (not in original abstraction)")
            continue
        if idx not in curr_map:
            blocks.append(f"Primitive {idx}: REMOVED (omitted from current JSON)")
            continue

        o, c = orig_map[idx], curr_map[idx]
        changes: List[str] = []
        for field in ("scale", "translation", "exponents", "color"):
            if field in o or field in c:
                changes.extend(
                    _format_scalar_list_diff(field, o.get(field), c.get(field), tol=tol)
                )
        if "rotation" in o or "rotation" in c:
            if _values_differ(o.get("rotation"), c.get("rotation"), tol):
                changes.append("  - rotation: matrix changed")
        if not changes:
            continue
        blocks.append(f"Primitive {idx}:\n" + "\n".join(changes))

    if not blocks:
        return (
            "(No numeric changes detected between the original abstraction JSON "
            "and the current JSON.)"
        )
    return "\n\n".join(blocks)


def _query_vlm_for_feedback(
    vlm: str,
    original_img_path: Path,
    abstraction_img_path: Path,
    edited_img_path: Path,
    edit_instruction: str,
    previous_reasoning: str,
    current_json: List[Dict],
    original_json: Optional[List[Dict]] = None,
    model_size: str = "4B",
    max_new_tokens: int = 4096,
    gemini_model: str = "gemini-2.5-pro",
    gpt_model: str = "gpt-5.5",
    api_key: Optional[str] = None,
) -> Dict:
    """
    Query VLM for feedback on the edit.

    Returns:
        Dict with 'text' (response text) and 'token_usage' (input/output token counts)
    """

    feedback_instruction = _load_vlm_feedback_instruction()
    current_json_str = json.dumps(current_json, indent=2)

    if original_json is not None:
        json_diff_summary = _format_abstraction_diff(original_json, current_json)
    else:
        json_diff_summary = "(Original abstraction JSON not provided for diff.)"

    text_prompt = f"""
{feedback_instruction}

## Original Edit Instruction
{edit_instruction}

## Your Previous Reasoning and Output
{previous_reasoning}

## Parameter changes (original abstraction JSON -> current JSON)
Use this diff as ground truth for whether parameters changed. Do not claim the JSON is
unchanged if this section lists changes.

{json_diff_summary}

## Current JSON (result of your edit)
```json
{current_json_str}
```

Please evaluate whether the edited shape (shown in the third image) correctly addresses the original instruction.
Compare it to the original shape and abstraction to verify the changes are correct.
"""

    if vlm.lower() == "gemini":
        from google import genai

        client = _get_gemini_client()

        original_img = Image.open(original_img_path)
        abstraction_img = Image.open(abstraction_img_path)
        edited_img = Image.open(edited_img_path)

        config = genai.types.GenerateContentConfig(
            temperature=1,
            top_p=0.95,
            max_output_tokens=32768,
            response_modalities=["TEXT"],
        )

        print(f"Querying Gemini {gemini_model} for feedback...")

        result = client.models.generate_content(
            model=gemini_model,
            contents=[
                "ORIGINAL shape:",
                original_img,
                "ORIGINAL abstraction (multiple views):",
                abstraction_img,
                "YOUR EDITED shape (multiple views):",
                edited_img,
                text_prompt,
            ],
            config=config,
        )

        response_text = ""
        if result.candidates and result.candidates[0].content:
            for part in result.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    response_text += part.text

        # Extract token usage
        token_usage = {"input_tokens": 0, "output_tokens": 0}
        if hasattr(result, "usage_metadata") and result.usage_metadata:
            token_usage["input_tokens"] = (
                getattr(result.usage_metadata, "prompt_token_count", 0) or 0
            )
            token_usage["output_tokens"] = (
                getattr(result.usage_metadata, "candidates_token_count", 0) or 0
            )
            print(
                f"  Token usage: {token_usage['input_tokens']} input, {token_usage['output_tokens']} output"
            )

        return {"text": response_text, "token_usage": token_usage}

    elif vlm.lower() == "gpt":
        from openai import OpenAI as OpenAIClient

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        client = OpenAIClient(api_key=api_key)

        original_b64 = _image_to_base64(original_img_path)
        abstraction_b64 = _image_to_base64(abstraction_img_path)
        edited_b64 = _image_to_base64(edited_img_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ORIGINAL shape:"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{original_b64}"},
                    },
                    {"type": "text", "text": "ORIGINAL abstraction (multiple views):"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{abstraction_b64}"
                        },
                    },
                    {"type": "text", "text": "YOUR EDITED shape (multiple views):"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{edited_b64}"},
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        print(f"Querying OpenAI {gpt_model} for feedback...")
        response = client.chat.completions.create(
            model=gpt_model,
            messages=messages,
        )

        # Extract token usage
        token_usage = {"input_tokens": 0, "output_tokens": 0}
        if hasattr(response, "usage") and response.usage:
            token_usage["input_tokens"] = (
                getattr(response.usage, "prompt_tokens", 0) or 0
            )
            token_usage["output_tokens"] = (
                getattr(response.usage, "completion_tokens", 0) or 0
            )
            print(
                f"  Token usage: {token_usage['input_tokens']} input, {token_usage['output_tokens']} output"
            )

        return {"text": response.choices[0].message.content, "token_usage": token_usage}

    elif vlm.lower() == "qwen":
        vlm_model, vlm_processor = load_local_qwen(model_size)

        original_img = Image.open(original_img_path)
        abstraction_img = Image.open(abstraction_img_path)
        edited_img = Image.open(edited_img_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "ORIGINAL shape:"},
                    {"type": "image", "image": original_img},
                    {"type": "text", "text": "ORIGINAL abstraction (multiple views):"},
                    {"type": "image", "image": abstraction_img},
                    {"type": "text", "text": "YOUR EDITED shape (multiple views):"},
                    {"type": "image", "image": edited_img},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        print(f"Querying Qwen3-VL-{model_size} for feedback...")

        inputs = vlm_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(vlm_model.device)

        with torch.no_grad():
            generated_ids = vlm_model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = vlm_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Calculate token usage for local Qwen
        token_usage = {
            "input_tokens": inputs.input_ids.shape[1],
            "output_tokens": generated_ids[0].shape[0] - inputs.input_ids.shape[1],
        }
        print(
            f"  Token usage: {token_usage['input_tokens']} input, {token_usage['output_tokens']} output"
        )

        return {"text": output_text, "token_usage": token_usage}
    else:
        raise ValueError(f"Unknown VLM: {vlm}")


def _check_if_done(response_text: str) -> bool:
    """Check if VLM response indicates editing is complete."""
    return (
        "STATUS: DONE" in response_text.upper()
        or "STATUS:DONE" in response_text.upper()
    )
