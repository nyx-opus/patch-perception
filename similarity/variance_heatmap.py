#!/usr/bin/env python3
"""
Generate a variance heatmap overlay from v2 pipeline variance data.

Creates a visual overlay showing which patches are perceptually ambiguous
(high variance, warm colours) vs perceptually stable (low variance, cool colours).

Usage:
    python3 variance_heatmap.py <variance.json> <source_image> [--output heatmap.png]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_heatmap(variance_path: str, image_path: str,
                   grid_size: int = 4, opacity: float = 0.5) -> Image.Image:
    """Create a variance heatmap overlay on the source image."""

    with open(variance_path) as f:
        data = json.load(f)

    source = Image.open(image_path).convert("RGBA")
    w, h = source.size
    patch_w = w // grid_size
    patch_h = h // grid_size

    # Extract variance values
    variance_map = {}
    for patch in data["patches"]:
        key = (patch["row"], patch["col"])
        variance_map[key] = patch["full"]["variance"]

    if not variance_map:
        print("No variance data found")
        return source

    variances = list(variance_map.values())
    v_min = min(variances)
    v_max = max(variances)
    v_range = v_max - v_min if v_max > v_min else 1.0

    # Create overlay
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for (row, col), variance in variance_map.items():
        # Normalize to [0, 1]
        t = (variance - v_min) / v_range

        # Cool (blue, confident) → Warm (orange, uncertain)
        # Blue: (50, 100, 200) → Orange: (230, 120, 30)
        r = int(50 + t * 180)
        g = int(100 + t * 20)
        b = int(200 - t * 170)
        a = int(opacity * 255)

        x0 = col * patch_w
        y0 = row * patch_h
        x1 = x0 + patch_w
        y1 = y0 + patch_h

        draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, a))

        # Add variance text
        text = f"{variance:.3f}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except OSError:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = x0 + (patch_w - tw) // 2
        ty = y0 + (patch_h - th) // 2

        # Text shadow for readability
        draw.text((tx + 1, ty + 1), text, fill=(0, 0, 0, 200), font=font)
        draw.text((tx, ty), text, fill=(255, 255, 255, 230), font=font)

    # Composite
    result = Image.alpha_composite(source, overlay)
    return result.convert("RGB")


def main():
    parser = argparse.ArgumentParser(description="Generate variance heatmap overlay")
    parser.add_argument("variance", help="Variance JSON from measure_variance.py")
    parser.add_argument("image", help="Source image")
    parser.add_argument("--grid", type=int, default=4, help="Grid size")
    parser.add_argument("--opacity", type=float, default=0.45, help="Overlay opacity (0-1)")
    parser.add_argument("--output", "-o", default="variance_heatmap.png", help="Output file")
    args = parser.parse_args()

    result = create_heatmap(args.variance, args.image,
                            grid_size=args.grid, opacity=args.opacity)
    result.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
