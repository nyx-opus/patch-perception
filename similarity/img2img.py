#!/usr/bin/env python3
"""
Image-to-image generation via Replicate API.

Two modes:
- Canny edge-guided (flux-canny-pro): preserves edges/structure, changes content
- Img2img (stability-ai/stable-diffusion-img2img): varies content with strength control

Usage:
    from img2img import generate_canny, generate_img2img

    # Edge-guided: same shapes, different subject
    result = generate_canny(patch_image, "a coral reef fragment", guidance=15)

    # Img2img: gradual transformation
    result = generate_img2img(patch_image, "alien bioluminescence", strength=0.7)
"""

import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image


def _get_token() -> str:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("Error: REPLICATE_API_TOKEN not set")
        sys.exit(1)
    return token


def _image_to_data_uri(image: Image.Image, max_size: int = 512,
                       fmt: str = "PNG") -> str:
    """Convert PIL image to base64 data URI."""
    img = image.copy()
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _run_prediction(model: str, input_data: dict, timeout: int = 120,
                    version: str | None = None) -> str | None:
    """Run a Replicate prediction and return the output URL.

    Args:
        model: Model identifier (e.g. "black-forest-labs/flux-canny-pro")
        input_data: Model input parameters
        timeout: HTTP timeout in seconds
        version: If set, use version-based endpoint (needed for older models)
    """
    token = _get_token()

    if version:
        # Older models need /v1/predictions with explicit version
        payload = json.dumps({"version": version, "input": input_data}).encode()
        url = "https://api.replicate.com/v1/predictions"
    else:
        payload = json.dumps({"input": input_data}).encode()
        url = f"https://api.replicate.com/v1/models/{model}/predictions"

    req = urllib.request.Request(url, data=payload, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    })

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        print(f"  API error: {e}")
        return None

    output = result.get("output")

    # Poll if not ready
    if not output and result.get("urls", {}).get("get"):
        poll_url = result["urls"]["get"]
        for _ in range(60):
            time.sleep(2)
            req2 = urllib.request.Request(poll_url, headers={
                "Authorization": f"Bearer {token}",
            })
            with urllib.request.urlopen(req2) as resp2:
                result = json.loads(resp2.read())
            if result.get("status") == "succeeded":
                output = result.get("output")
                break
            elif result.get("status") == "failed":
                print(f"  Generation failed: {result.get('error')}")
                return None

    if not output:
        return None

    return output[0] if isinstance(output, list) else output


def _download_image(url: str) -> Image.Image | None:
    """Download image from URL and return as PIL Image."""
    try:
        with urllib.request.urlopen(url) as resp:
            return Image.open(io.BytesIO(resp.read())).convert("RGB")
    except Exception as e:
        print(f"  Download error: {e}")
        return None


def generate_canny(image: Image.Image, prompt: str,
                   guidance: float = 15, steps: int = 28) -> Image.Image | None:
    """Generate using flux-canny-pro: edge-guided, preserves structure.

    Args:
        image: Source patch image
        prompt: What the output should depict (keeping the same edges)
        guidance: How closely to follow prompt vs edges (1-100, default 15).
                  Lower = more creative, higher = more literal.
        steps: Diffusion steps (15-50, default 28)

    Returns:
        Generated PIL Image or None on failure.
    """
    data_uri = _image_to_data_uri(image)
    input_data = {
        "prompt": prompt,
        "control_image": data_uri,
        "guidance": guidance,
        "steps": steps,
        "output_format": "png",
    }
    url = _run_prediction("black-forest-labs/flux-canny-pro", input_data)
    if url:
        return _download_image(url)
    return None


def generate_img2img(image: Image.Image, prompt: str,
                     strength: float = 0.7,
                     guidance_scale: float = 7.5) -> Image.Image | None:
    """Generate using SD img2img: varies content with strength control.

    Args:
        image: Source patch image
        prompt: What to transform toward
        strength: How far to diverge (0.0=keep original, 1.0=completely new).
                  Good range: 0.5-0.8
        guidance_scale: Text adherence (default 7.5)

    Returns:
        Generated PIL Image or None on failure.
    """
    data_uri = _image_to_data_uri(image, fmt="PNG")
    input_data = {
        "image": data_uri,
        "prompt": prompt,
        "prompt_strength": strength,
        "guidance_scale": guidance_scale,
        "num_outputs": 1,
        "num_inference_steps": 25,
    }
    # SD img2img needs version-based endpoint
    SD_IMG2IMG_VERSION = "15a3689ee13b0d2616e98820eca31d4c3abcd36672df6afce5cb6feb1d66087d"
    url = _run_prediction("stability-ai/stable-diffusion-img2img", input_data,
                          version=SD_IMG2IMG_VERSION)
    if url:
        return _download_image(url)
    return None


def generate_text2img(prompt: str,
                      model: str = "black-forest-labs/flux-schnell") -> Image.Image | None:
    """Generate from text only (existing capability, unified interface).

    Args:
        prompt: Text description
        model: Replicate model (default: flux-schnell)

    Returns:
        Generated PIL Image or None on failure.
    """
    input_data = {
        "prompt": prompt,
        "num_outputs": 1,
        "aspect_ratio": "1:1",
        "output_format": "png",
    }
    url = _run_prediction(model, input_data)
    if url:
        return _download_image(url)
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test img2img generation")
    parser.add_argument("image", help="Source image")
    parser.add_argument("prompt", help="Generation prompt")
    parser.add_argument("--mode", choices=["canny", "img2img", "text"],
                        default="canny", help="Generation mode")
    parser.add_argument("--strength", type=float, default=0.7,
                        help="img2img strength (0-1)")
    parser.add_argument("--guidance", type=float, default=15,
                        help="canny guidance (1-100)")
    parser.add_argument("--output", "-o", default="output.png",
                        help="Output file")
    args = parser.parse_args()

    if args.mode == "text":
        print(f"Generating from text: {args.prompt}")
        result = generate_text2img(args.prompt)
    else:
        source = Image.open(args.image).convert("RGB")
        print(f"Source: {args.image} ({source.size[0]}x{source.size[1]})")
        print(f"Mode: {args.mode}")
        print(f"Prompt: {args.prompt}")

        if args.mode == "canny":
            print(f"Guidance: {args.guidance}")
            result = generate_canny(source, args.prompt, guidance=args.guidance)
        else:
            print(f"Strength: {args.strength}")
            result = generate_img2img(source, args.prompt, strength=args.strength)

    if result:
        result.save(args.output)
        print(f"Saved: {args.output}")
    else:
        print("Generation failed")
        sys.exit(1)
