#!/usr/bin/env python3
"""
3-tier vision captioning for patch-perception v2.

Each patch gets three different descriptions from a vision model:
- REPORT: Describe what you see (literal, grounded)
- INTERPRET: What might be just outside the frame? (contextual, imaginative)
- DREAM: What else has these exact shapes/colours/textures? (associative, wild)

These feed into the text-first generation path (3 images per caption = 9 total).

Usage:
    python3 caption_v2.py <image_path> [--grid 4] [--output captions_v2.json]
    python3 caption_v2.py <image_path> --patches "1,1 2,3"

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


# Model registry: models that need version-based API endpoint
MODEL_VERSIONS = {
    "yorickvp/llava-v1.6-mistral-7b": "19be067b589d0c46689ffa7cc3ff321447a441986a7694c01225973c2eafc874",
    "yorickvp/llava-13b": "80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
    "lucataco/moondream2": "72ccb656353c348c10205e7451a1aa72cfb3abd2ac3e42e7b58c403e78c1fa85",
}

PROMPTS = {
    "report": (
        "You are looking at a small cropped patch from a larger photograph. "
        "You cannot see the full image — only this fragment. "
        "Describe exactly what you see: textures, colours, shapes, materials. "
        "Be precise and specific. Two to three short phrases. "
        "Do not guess what the full image shows."
    ),
    "interpret": (
        "You are looking at a small cropped patch from a larger photograph. "
        "You cannot see the full image — only this fragment. "
        "What might be just outside the frame? What larger scene could this "
        "be part of? Describe 2-3 possibilities for what the full image shows, "
        "based only on the clues in this fragment."
    ),
    "dream": (
        "Look at this image fragment and forget what it actually is. "
        "What else has this exact combination of shapes, colours and textures? "
        "What could you mistake this for in dim light, or half-asleep, or from "
        "the corner of your eye? List 3-5 wildly different things this could be. "
        "Be specific and surprising."
    ),
}

DEFAULT_MODEL = "yorickvp/llava-v1.6-mistral-7b"


def _image_to_data_uri(image: Image.Image, max_size: int = 512) -> str:
    """Convert PIL image to base64 data URI."""
    img = image.copy()
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def caption_image(image: Image.Image, prompt: str,
                  model: str = DEFAULT_MODEL) -> str | None:
    """Send an image patch to a vision model and get a free-text caption."""
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("Error: REPLICATE_API_TOKEN not set")
        sys.exit(1)

    data_uri = _image_to_data_uri(image)
    input_data = {
        "prompt": prompt,
        "image": data_uri,
        "max_tokens": 512,
        "temperature": 0.8,
    }

    version = MODEL_VERSIONS.get(model)
    if version:
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

    # Output format varies — handle string or list of tokens
    if isinstance(output, list):
        return "".join(output)
    return str(output)


def caption_patch_3tier(patch: Image.Image, model: str,
                        tiers: list[str] | None = None) -> dict[str, str | None]:
    """Run all three caption tiers on a single patch."""
    tiers = tiers or list(PROMPTS.keys())
    results = {}
    for tier in tiers:
        prompt = PROMPTS[tier]
        caption = caption_image(patch, prompt, model=model)
        results[tier] = caption
    return results


def main():
    parser = argparse.ArgumentParser(
        description="3-tier vision captioning for patch-perception v2")
    parser.add_argument("image", help="Input image")
    parser.add_argument("--grid", type=int, default=4, help="Grid size (default: 4)")
    parser.add_argument("--output", "-o", help="Save results as JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Replicate vision model (default: {DEFAULT_MODEL})")
    parser.add_argument("--patches", help="Only caption specific patches, e.g. '0,2 2,1'")
    parser.add_argument("--tiers", help="Which tiers to run (default: all). "
                        "Comma-separated: report,interpret,dream")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: Set REPLICATE_API_TOKEN environment variable")
        sys.exit(1)

    tiers = args.tiers.split(",") if args.tiers else list(PROMPTS.keys())
    for t in tiers:
        if t not in PROMPTS:
            print(f"Unknown tier: {t}. Available: {', '.join(PROMPTS.keys())}")
            sys.exit(1)

    selected = None
    if args.patches:
        selected = set()
        for p in args.patches.split():
            r, c = p.split(",")
            selected.add((int(r), int(c)))

    print(f"Model: {args.model}")
    print(f"Tiers: {', '.join(tiers)}")
    print(f"Splitting {image_path.name} into {args.grid}x{args.grid} patches...")

    image = Image.open(image_path).convert("RGB")
    patches = split_into_patches(image, args.grid)

    results = []
    total = len(patches) if not selected else len(selected)
    done = 0

    for i, patch in enumerate(patches):
        row = i // args.grid
        col = i % args.grid

        if selected and (row, col) not in selected:
            continue

        done += 1
        print(f"\nPatch [{row},{col}] ({done}/{total}):")
        captions = caption_patch_3tier(patch, args.model, tiers=tiers)

        for tier, text in captions.items():
            preview = (text[:80] + "...") if text and len(text) > 80 else text
            print(f"  {tier}: {preview}")

        results.append({
            "row": row,
            "col": col,
            "captions": captions,
        })

    if args.output:
        output = {
            "source_image": str(image_path),
            "grid_size": args.grid,
            "model": args.model,
            "tiers": tiers,
            "patches": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")

    n_calls = done * len(tiers)
    print(f"\nAPI calls made: {n_calls} ({done} patches × {len(tiers)} tiers)")


if __name__ == "__main__":
    main()
