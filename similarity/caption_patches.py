#!/usr/bin/env python3
"""
Caption image patches using a vision-language model via Replicate.

Instead of ranking a fixed vocabulary (like CLIP), this asks a model to
freely describe what it sees in each patch — its own strange and wonderful
associations, unconstrained by our word list.

Usage:
    python3 caption_patches.py <image_path> [--grid 4] [--output captions.json]
    python3 caption_patches.py <image_path> --prompt "Describe this image fragment"

Environment:
    REPLICATE_API_TOKEN=your_token_here
"""

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

from PIL import Image
from embed_patches import split_into_patches


DEFAULT_PROMPT = (
    "You are looking at a small cropped patch from a larger photograph. "
    "You cannot see the full image — only this fragment. "
    "Describe what you see in 2-3 short phrases. What does it look like? "
    "What could it be? Include your uncertain impressions — if it could be "
    "feathers or grass or brushstrokes, say so. Be specific about textures, "
    "colours, and shapes. Do not guess what the full image is."
)


def image_to_data_uri(image: Image.Image, max_size: int = 512) -> str:
    """Convert PIL image to base64 data URI for API consumption."""
    # Resize if needed
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def caption_image(image: Image.Image, prompt: str,
                  model: str = "meta/llama-4-scout-instruct") -> str | None:
    """Send an image patch to a vision model and get a free-text caption."""
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("Error: REPLICATE_API_TOKEN not set")
        sys.exit(1)

    data_uri = image_to_data_uri(image)

    payload = json.dumps({
        "input": {
            "prompt": prompt,
            "image": data_uri,
        }
    }).encode()

    url = f"https://api.replicate.com/v1/models/{model}/predictions"
    req = urllib.request.Request(url, data=payload, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "wait",
    })

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        print(f"  API error: {e}")
        return None

    output = result.get("output")

    # Poll if needed
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
                print(f"  Caption failed: {result.get('error')}")
                return None

    if output is None:
        return None

    # Output format varies by model — handle string or list
    if isinstance(output, list):
        return "".join(output)
    return str(output)


def main():
    parser = argparse.ArgumentParser(description="Caption image patches with a vision model")
    parser.add_argument("image", help="Input image")
    parser.add_argument("--grid", type=int, default=4, help="Grid size (default: 4)")
    parser.add_argument("--output", "-o", help="Save results as JSON")
    parser.add_argument("--model", default="meta/llama-4-scout-instruct",
                        help="Replicate vision model (default: llama-4-scout-instruct)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Caption prompt")
    parser.add_argument("--patches", help="Only caption specific patches, e.g. '0,2 2,1'")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: Set REPLICATE_API_TOKEN environment variable")
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Splitting {image_path.name} into {args.grid}x{args.grid} patches...")

    image = Image.open(image_path).convert("RGB")
    patches = split_into_patches(image, args.grid)

    # Filter patches if specified
    if args.patches:
        selected = set()
        for p in args.patches.split():
            r, c = p.split(",")
            selected.add((int(r), int(c)))
    else:
        selected = None

    results = []
    for i, patch in enumerate(patches):
        row = i // args.grid
        col = i % args.grid

        if selected and (row, col) not in selected:
            continue

        print(f"\nPatch [{row},{col}]:")
        caption = caption_image(patch, args.prompt, model=args.model)
        if caption:
            print(f"  {caption}")
            results.append({"row": row, "col": col, "caption": caption})
        else:
            print(f"  (no caption)")
            results.append({"row": row, "col": col, "caption": None})

    if args.output:
        output = {
            "source_image": str(image_path),
            "grid_size": args.grid,
            "model": args.model,
            "prompt": args.prompt,
            "patches": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
