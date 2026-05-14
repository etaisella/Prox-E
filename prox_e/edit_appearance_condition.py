"""Edit appearance conditioning image using 2D image editing models."""

import io
import os
from pathlib import Path
from PIL import Image

from google import genai


def _get_gemini_client():
    """Get Gemini client with API key or internal GCP credentials."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)

    from prox_e.gemini_auth import import_token_source_v2

    TokenSourceV2 = import_token_source_v2()

    project_number = "380907735821"
    pool_id = "craftworks-team-sa"
    prd_id = "440036398022-3100921420"
    service_account = "craftworks-team-sa@research-prototypes.iam.gserviceaccount.com"
    
    gcp_audience = f"//iam.googleapis.com/projects/{project_number}/locations/global/workloadIdentityPools/{pool_id}/providers/{prd_id}"
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    credentials = TokenSourceV2(service_account, gcp_audience, scopes)
    
    return genai.Client(
        project="research-prototypes",
        location="global",
        vertexai=True,
        credentials=credentials,
    )


def _decode_image_from_response(response) -> Image.Image:
    """Decode image from Gemini response."""
    inline_data = next(
        part.inline_data
        for part in response.candidates[0].content.parts
        if part.inline_data is not None
    )
    return Image.open(io.BytesIO(inline_data.data))


def edit_with_gemini(image_path: str, prompt: str) -> Image.Image:
    """Edit image using Gemini nano-banana model."""
    image = Image.open(image_path)
    client = _get_gemini_client()
    
    config = genai.types.GenerateContentConfig(
        temperature=1,
        top_p=0.95,
        max_output_tokens=32768,
        response_modalities=["IMAGE"],
        safety_settings=[
            genai.types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            genai.types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            genai.types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            genai.types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
        ],
    )
    
    print(f"  Editing image with Gemini...")
    print(f"  Prompt: {prompt}")
    
    result = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20",
        contents=[image, prompt],
        config=config,
    )
    
    return _decode_image_from_response(result)


def edit_with_kontext(image_path: str, prompt: str) -> Image.Image:
    """Edit image using Kontext model locally via diffusers."""
    import torch
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    
    image = load_image(image_path)
    
    print(f"  Loading Kontext pipeline...")
    pipe = FluxKontextPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Kontext-dev",
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    
    print(f"  Editing image with Kontext...")
    print(f"  Prompt: {prompt}")
    
    result = pipe(
        image=image,
        prompt=prompt,
        guidance_scale=2.5,
        num_inference_steps=28,
    )
    
    edited_image = result.images[0]
    
    # Clean up GPU memory
    del pipe
    torch.cuda.empty_cache()
    print(f"  Cleaned up Kontext pipeline from GPU")
    
    return edited_image


def edit_appearance_image(
    image_path: str,
    category: str,
    appearance_description: str,
    output_path: str,
    model: str = "gemini",
) -> Image.Image:
    """
    Edit an appearance conditioning image based on the appearance description.
    
    Args:
        image_path: Path to the input image (e.g., conditioning_render.png)
        category: Shape category (e.g., "chair")
        appearance_description: Target appearance (e.g., "an ornate wooden chair")
        output_path: Path to save the edited image
        model: Model to use for editing - "gemini" or "kontext"
    
    Returns:
        The edited PIL Image
    """
    print(f"\n--- Editing Appearance Condition ---")
    print(f"  Input: {image_path}")
    print(f"  Model: {model}")
    
    # Build editing prompt
    prompt = f"make this {category} into {appearance_description}"
    
    # Edit with selected model
    if model.lower() == "gemini":
        edited_image = edit_with_gemini(image_path, prompt)
    elif model.lower() == "kontext":
        edited_image = edit_with_kontext(image_path, prompt)
    else:
        raise ValueError(f"Unknown model: {model}. Use 'gemini' or 'kontext'.")
    
    # Save edited image
    edited_image.save(output_path)
    print(f"  Saved: {output_path}")
    
    return edited_image


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Edit appearance conditioning image")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--category", default="chair", help="Shape category")
    parser.add_argument("--appearance", required=True, help="Target appearance description")
    parser.add_argument("--output", default="edited_appearance.png", help="Output path")
    parser.add_argument("--model", default="gemini", choices=["gemini", "kontext"], help="Model to use")
    
    args = parser.parse_args()
    
    edit_appearance_image(
        image_path=args.image_path,
        category=args.category,
        appearance_description=args.appearance,
        output_path=args.output,
        model=args.model,
    )
