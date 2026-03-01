#!/usr/bin/env python3
"""
Generate alternative patch images using CLIP descriptions + Replicate API.

Takes CLIP patch descriptions (from describe_patches.py) and generates
"what this patch could also be" images via an image generation model.

Usage:
    python3 generate_alternatives.py <descriptions.json> [--model flux-schnell] [--per-patch 3]
    python3 generate_alternatives.py <descriptions.json> --output generated/

Or directly from an image (runs describe_patches internally):
    python3 generate_alternatives.py --image <image_path> --grid 4

Environment:
    REPLICATE_API_TOKEN=your_token_here
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def generate_image(prompt: str, model: str = "black-forest-labs/flux-schnell",
                   size: int = 256) -> bytes | None:
    """Generate an image via Replicate API. Returns image bytes or None."""
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("Error: REPLICATE_API_TOKEN not set")
        sys.exit(1)

    # Start prediction
    payload = json.dumps({
        "input": {
            "prompt": prompt,
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "png",
        }
    }).encode()

    url = f"https://api.replicate.com/v1/models/{model}/predictions"
    req = urllib.request.Request(url, data=payload, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Prefer": "wait",  # Synchronous — wait for result
    })

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        print(f"  API error: {e}")
        return None

    # Get the output URL
    output = result.get("output")
    if not output:
        # If not ready yet, poll
        poll_url = result.get("urls", {}).get("get")
        if poll_url:
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

    # output is a list of URLs (or a single URL for some models)
    img_url = output[0] if isinstance(output, list) else output

    # Download the image
    with urllib.request.urlopen(img_url) as img_resp:
        return img_resp.read()


def build_prompt(descriptions: list[dict], patch_row: int, patch_col: int) -> str:
    """Build a generation prompt from CLIP descriptions of a patch."""
    # Take top 3 terms
    top_terms = [d["term"] for d in descriptions[:3]]
    terms_str = ", ".join(top_terms)

    return (
        f"A tightly cropped portion of a larger photograph showing {terms_str}. "
        f"Macro detail, natural lighting, photographic texture. "
        f"No text, no borders, no frame."
    )


def main():
    parser = argparse.ArgumentParser(description="Generate alternative patch images")
    parser.add_argument("descriptions", nargs="?", help="JSON file from describe_patches.py")
    parser.add_argument("--image", help="Generate descriptions on the fly from an image")
    parser.add_argument("--grid", type=int, default=4, help="Grid size (default: 4)")
    parser.add_argument("--model", default="black-forest-labs/flux-schnell",
                        help="Replicate model (default: flux-schnell)")
    parser.add_argument("--per-patch", type=int, default=3,
                        help="Alternatives to generate per patch (default: 3)")
    parser.add_argument("--output", "-o", default="generated",
                        help="Output directory (default: generated/)")
    parser.add_argument("--patches", help="Only generate for specific patches, e.g. '1,1 2,1 0,2'")
    parser.add_argument("--custom-prompt", help="Override prompt template. Use {terms} for CLIP terms.")
    args = parser.parse_args()

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: Set REPLICATE_API_TOKEN environment variable")
        print("  export REPLICATE_API_TOKEN=r8_...")
        sys.exit(1)

    # Load or generate descriptions
    if args.descriptions:
        with open(args.descriptions) as f:
            data = json.load(f)
        patches = data["patches"]
        source_image = data.get("source_image", "unknown")
    elif args.image:
        print("Generating CLIP descriptions first...")
        import subprocess
        desc_file = "/tmp/patch_descriptions.json"
        subprocess.run([
            sys.executable, "describe_patches.py", args.image,
            "--grid", str(args.grid), "--top-k", "5", "--output", desc_file
        ], check=True)
        with open(desc_file) as f:
            data = json.load(f)
        patches = data["patches"]
        source_image = args.image
    else:
        print("Provide either a descriptions JSON file or --image <path>")
        sys.exit(1)

    # Filter patches if specified
    if args.patches:
        selected = set()
        for p in args.patches.split():
            r, c = p.split(",")
            selected.add((int(r), int(c)))
        patches = [p for p in patches if (p["row"], p["col"]) in selected]

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source: {source_image}")
    print(f"Model: {args.model}")
    print(f"Generating {args.per_patch} alternatives for {len(patches)} patches")
    print(f"Output: {output_dir}/")
    print()

    manifest = {"source_image": str(source_image), "model": args.model, "patches": []}

    for patch in patches:
        row, col = patch["row"], patch["col"]
        descriptions = patch["descriptions"]
        top_terms = [d["term"] for d in descriptions[:3]]

        print(f"Patch [{row},{col}] — {', '.join(top_terms)}")

        patch_results = {"row": row, "col": col, "terms": top_terms, "generated": []}

        for i in range(args.per_patch):
            # Vary the prompt slightly for each alternative
            if i == 0:
                terms_to_use = descriptions[:3]
            elif i == 1:
                terms_to_use = descriptions[1:4]
            else:
                terms_to_use = descriptions[2:5]

            if args.custom_prompt:
                terms_str = ", ".join(d["term"] for d in terms_to_use)
                prompt = args.custom_prompt.replace("{terms}", terms_str)
            else:
                prompt = build_prompt(terms_to_use, row, col)

            print(f"  [{i+1}/{args.per_patch}] {prompt[:80]}...")

            img_bytes = generate_image(prompt, model=args.model)
            if img_bytes:
                filename = f"patch_{row}_{col}_alt_{i}.png"
                filepath = output_dir / filename
                with open(filepath, "wb") as f:
                    f.write(img_bytes)
                print(f"    Saved: {filename}")
                patch_results["generated"].append({
                    "filename": filename,
                    "prompt": prompt,
                    "terms": [d["term"] for d in terms_to_use],
                })
            else:
                print(f"    Failed to generate")

        manifest["patches"].append(patch_results)

    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to {manifest_path}")
    print(f"Generated {sum(len(p['generated']) for p in manifest['patches'])} images")


if __name__ == "__main__":
    main()
