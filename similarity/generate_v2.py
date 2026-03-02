#!/usr/bin/env python3
"""
V2 generation: text-first + image-first paths for patch-perception.

Takes 3-tier captions (from caption_v2.py) and source image, produces 15 alternatives
per patch:
- TEXT-FIRST (9): 3 images from each caption tier (report, interpret, dream)
- IMAGE-FIRST (6): 3 pure img2img + 3 img2img with caption influence

Total with original: 16 images per patch for CLIP variance measurement.

Usage:
    python3 generate_v2.py <captions_v2.json> <source_image> [--output generated_v2/]
    python3 generate_v2.py <captions_v2.json> <source_image> --patches "1,1 2,3"

Environment:
    REPLICATE_API_TOKEN=your_token_here
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image
from embed_patches import split_into_patches
from img2img import generate_text2img, generate_canny, generate_img2img


def _clean_caption_lines(caption: str) -> list[str]:
    """Extract meaningful content lines from a vision model caption.

    Strips preamble, formatting, meta-commentary, and keeps concrete descriptions.
    """
    lines = [l.strip() for l in caption.strip().split("\n") if l.strip()]

    # Skip common preamble lines
    skip_phrases = [
        "unfortunately", "i'm a text", "i don't see", "i cannot",
        "let me", "here are", "these are", "below are", "with that in mind",
        "based on", "looking at this",
    ]

    cleaned = []
    for line in lines:
        # Strip numbered/bulleted list formatting
        for prefix in ["1.", "2.", "3.", "4.", "5.", "- ", "* ", "**"]:
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        # Strip bold markdown
        line = line.replace("**", "")
        # Skip short lines, preamble, and meta-commentary
        if len(line) < 15:
            continue
        if any(line.lower().startswith(p) for p in skip_phrases):
            continue
        cleaned.append(line)

    return cleaned or lines[:3]


def build_text_prompt(caption: str, tier: str) -> str:
    """Build a text-to-image prompt from a caption tier."""
    cleaned = _clean_caption_lines(caption)

    # Use first 2-3 meaningful lines as prompt basis
    description = ". ".join(cleaned[:3])
    if len(description) > 300:
        description = description[:300]

    return (
        f"A tightly cropped macro photograph showing: {description}. "
        f"Natural lighting, photographic texture, high detail. "
        f"No text, no borders, no frame."
    )


def build_img2img_prompt(caption: str | None = None) -> str:
    """Build a prompt for image-first generation.

    If caption provided, incorporates the textual idea.
    If not, gives a generic transformation prompt.
    """
    if caption:
        cleaned = _clean_caption_lines(caption)
        # Pick the most interesting/surprising line (longest = most descriptive)
        best = max(cleaned, key=len) if cleaned else caption[:100]
        return (
            f"Transform this image into: {best}. "
            f"Keep the same composition and layout. "
            f"Photographic quality, natural lighting."
        )
    else:
        return (
            "A completely different subject with the same shapes and composition. "
            "Photographic quality, natural lighting, macro detail."
        )


# Generic prompts for pure img2img (no caption influence)
PURE_IMG2IMG_PROMPTS = [
    "A completely different organic material with the same texture and shape. "
    "Photographic quality, natural lighting, macro detail.",

    "An alien landscape fragment with the same composition. "
    "Strange colours, bioluminescent textures, otherworldly but photographic.",

    "An underwater scene preserving the same shapes and layout. "
    "Coral, sea creatures, deep ocean colours. Photographic quality.",
]


def generate_for_patch(patch_image: Image.Image, captions: dict[str, str],
                       output_dir: Path, row: int, col: int) -> list[dict]:
    """Generate all 15 alternatives for a single patch.

    Returns list of generation result dicts.
    """
    results = []
    idx = 0

    # === TEXT-FIRST PATH (9 images) ===
    # 3 images from each caption tier
    for tier in ["report", "interpret", "dream"]:
        caption = captions.get(tier)
        if not caption:
            print(f"    Skipping {tier} (no caption)")
            continue

        prompt = build_text_prompt(caption, tier)

        for i in range(3):
            label = f"text_{tier}_{i}"
            print(f"    [{idx+1}/15] text-first/{tier}/{i} — {prompt[:60]}...")

            result = generate_text2img(prompt)
            if result:
                filename = f"patch_{row}_{col}_{label}.png"
                result.save(output_dir / filename)
                results.append({
                    "filename": filename,
                    "path": "text-first",
                    "tier": tier,
                    "variant": i,
                    "prompt": prompt,
                })
                print(f"      Saved: {filename}")
            else:
                print(f"      Failed")
            idx += 1

    # === IMAGE-FIRST PATH (6 images) ===
    # 3× pure img2img: same shape, different subject (canny edge-guided)
    for i, prompt in enumerate(PURE_IMG2IMG_PROMPTS):
        label = f"img_pure_{i}"
        print(f"    [{idx+1}/15] image-first/pure/{i} — {prompt[:60]}...")

        result = generate_canny(patch_image, prompt, guidance=15)
        if result:
            filename = f"patch_{row}_{col}_{label}.png"
            result.save(output_dir / filename)
            results.append({
                "filename": filename,
                "path": "image-first",
                "subpath": "pure",
                "variant": i,
                "prompt": prompt,
            })
            print(f"      Saved: {filename}")
        else:
            print(f"      Failed")
        idx += 1

    # 3× img2img with caption influence (SD img2img)
    # Use one idea from each caption tier
    for i, tier in enumerate(["report", "interpret", "dream"]):
        caption = captions.get(tier)
        prompt = build_img2img_prompt(caption)
        label = f"img_guided_{tier}"
        print(f"    [{idx+1}/15] image-first/guided/{tier} — {prompt[:60]}...")

        result = generate_img2img(patch_image, prompt, strength=0.65)
        if result:
            filename = f"patch_{row}_{col}_{label}.png"
            result.save(output_dir / filename)
            results.append({
                "filename": filename,
                "path": "image-first",
                "subpath": "guided",
                "tier": tier,
                "prompt": prompt,
            })
            print(f"      Saved: {filename}")
        else:
            print(f"      Failed")
        idx += 1

    return results


def main():
    parser = argparse.ArgumentParser(
        description="V2 generation: text-first + image-first paths")
    parser.add_argument("captions", help="Captions JSON from caption_v2.py")
    parser.add_argument("image", help="Source image")
    parser.add_argument("--grid", type=int, default=4, help="Grid size (default: 4)")
    parser.add_argument("--output", "-o", default="generated_v2",
                        help="Output directory (default: generated_v2/)")
    parser.add_argument("--patches", help="Only generate for specific patches, e.g. '1,1 2,3'")
    args = parser.parse_args()

    captions_path = Path(args.captions)
    image_path = Path(args.image)

    if not captions_path.exists():
        print(f"Captions not found: {captions_path}")
        sys.exit(1)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)
    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: Set REPLICATE_API_TOKEN environment variable")
        sys.exit(1)

    with open(captions_path) as f:
        captions_data = json.load(f)

    # Build caption lookup by (row, col)
    caption_lookup = {}
    for p in captions_data["patches"]:
        caption_lookup[(p["row"], p["col"])] = p["captions"]

    # Filter patches
    selected = None
    if args.patches:
        selected = set()
        for p in args.patches.split():
            r, c = p.split(",")
            selected.add((int(r), int(c)))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    patches = split_into_patches(image, args.grid)

    manifest = {
        "source_image": str(image_path),
        "captions_file": str(captions_path),
        "grid_size": args.grid,
        "patches": [],
    }

    total = len(caption_lookup) if not selected else len(selected)
    done = 0

    for i, patch_image in enumerate(patches):
        row = i // args.grid
        col = i % args.grid

        if selected and (row, col) not in selected:
            continue
        if (row, col) not in caption_lookup:
            continue

        done += 1
        captions = caption_lookup[(row, col)]
        print(f"\nPatch [{row},{col}] ({done}/{total}):")

        results = generate_for_patch(patch_image, captions, output_dir, row, col)

        manifest["patches"].append({
            "row": row,
            "col": col,
            "generated": results,
            "n_text_first": sum(1 for r in results if r["path"] == "text-first"),
            "n_image_first": sum(1 for r in results if r["path"] == "image-first"),
        })

    manifest_path = output_dir / "manifest_v2.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_gen = sum(len(p["generated"]) for p in manifest["patches"])
    print(f"\n{'='*60}")
    print(f"  Generated {total_gen} images for {len(manifest['patches'])} patches")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}")

    # Cost estimate
    n_text = sum(p["n_text_first"] for p in manifest["patches"])
    n_img = sum(p["n_image_first"] for p in manifest["patches"])
    cost_text = n_text * 0.003  # flux-schnell ~$0.003
    cost_canny = (n_img // 2) * 0.012  # flux-canny-pro ~$0.012
    cost_sd = (n_img - n_img // 2) * 0.004  # SD img2img ~$0.004
    print(f"\n  Estimated cost: ~${cost_text + cost_canny + cost_sd:.2f}")
    print(f"    {n_text} text-to-image + {n_img} img2img")


if __name__ == "__main__":
    main()
