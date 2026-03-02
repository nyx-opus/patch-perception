#!/usr/bin/env python3
"""
CLIP variance measurement across image sets for patch-perception v2.

Takes the v2 generation manifest + source image and computes CLIP embeddings
for all 16 images per patch (1 original + 15 generated), then measures variance
as an empirical uncertainty score.

High variance = genuinely ambiguous patch (alternatives scatter in CLIP space)
Low variance = perceptually stable patch (even wild prompts produce similar embeddings)

Usage:
    python3 measure_variance.py <manifest_v2.json> <source_image> [--grid 4]
    python3 measure_variance.py <manifest_v2.json> <source_image> --output variance.json

Environment:
    Needs open_clip (included in venv)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import open_clip

from embed_patches import split_into_patches


def compute_set_variance(images: list[Image.Image], model, preprocess) -> dict:
    """Compute CLIP variance across a set of images.

    Returns dict with:
    - mean_similarity: average pairwise cosine similarity
    - variance: variance of pairwise similarities
    - min_similarity: minimum pairwise similarity
    - max_similarity: maximum pairwise similarity
    - n_images: number of images in set
    """
    if len(images) < 2:
        return {
            "mean_similarity": 1.0,
            "variance": 0.0,
            "min_similarity": 1.0,
            "max_similarity": 1.0,
            "n_images": len(images),
        }

    # Embed all images
    inputs = torch.stack([preprocess(img) for img in images])
    with torch.no_grad():
        features = model.encode_image(inputs)
        features /= features.norm(dim=-1, keepdim=True)

    # Compute pairwise cosine similarities
    similarity_matrix = (features @ features.T).numpy()

    # Extract upper triangle (excluding diagonal)
    n = len(images)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append(float(similarity_matrix[i, j]))

    pairs = np.array(pairs)

    return {
        "mean_similarity": float(pairs.mean()),
        "variance": float(pairs.var()),
        "std": float(pairs.std()),
        "min_similarity": float(pairs.min()),
        "max_similarity": float(pairs.max()),
        "n_images": n,
        "n_pairs": len(pairs),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure CLIP variance across v2 image sets")
    parser.add_argument("manifest", help="V2 manifest JSON from generate_v2.py")
    parser.add_argument("image", help="Source image")
    parser.add_argument("--grid", type=int, default=4, help="Grid size (default: 4)")
    parser.add_argument("--output", "-o", help="Save variance results as JSON")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    image_path = Path(args.image)
    gen_dir = manifest_path.parent

    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()

    # Split source image into patches
    source = Image.open(image_path).convert("RGB")
    source_patches = split_into_patches(source, args.grid)

    results = []

    for patch_data in manifest["patches"]:
        row, col = patch_data["row"], patch_data["col"]
        patch_idx = row * args.grid + col
        original = source_patches[patch_idx]

        print(f"\nPatch [{row},{col}]:")

        # Collect all images: original + generated
        images = [original]
        labels = ["original"]

        for gen in patch_data["generated"]:
            img_path = gen_dir / gen["filename"]
            if img_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                    labels.append(gen["filename"])
                except Exception as e:
                    print(f"  Skipping {gen['filename']}: {e}")
            else:
                print(f"  Missing: {gen['filename']}")

        print(f"  {len(images)} images (1 original + {len(images)-1} generated)")

        # Compute full set variance
        full_stats = compute_set_variance(images, model, preprocess)
        print(f"  Full set: mean_sim={full_stats['mean_similarity']:.3f}, "
              f"var={full_stats['variance']:.4f}, "
              f"range=[{full_stats['min_similarity']:.3f}, {full_stats['max_similarity']:.3f}]")

        # Also compute text-first vs image-first separately
        text_images = [original] + [
            Image.open(gen_dir / g["filename"]).convert("RGB")
            for g in patch_data["generated"]
            if g["path"] == "text-first" and (gen_dir / g["filename"]).exists()
        ]
        img_images = [original] + [
            Image.open(gen_dir / g["filename"]).convert("RGB")
            for g in patch_data["generated"]
            if g["path"] == "image-first" and (gen_dir / g["filename"]).exists()
        ]

        text_stats = compute_set_variance(text_images, model, preprocess)
        img_stats = compute_set_variance(img_images, model, preprocess)

        print(f"  Text-first ({text_stats['n_images']}): "
              f"mean_sim={text_stats['mean_similarity']:.3f}, var={text_stats['variance']:.4f}")
        print(f"  Image-first ({img_stats['n_images']}): "
              f"mean_sim={img_stats['mean_similarity']:.3f}, var={img_stats['variance']:.4f}")

        results.append({
            "row": row,
            "col": col,
            "full": full_stats,
            "text_first": text_stats,
            "image_first": img_stats,
        })

    if args.output:
        output = {
            "source_image": str(image_path),
            "manifest": str(manifest_path),
            "grid_size": args.grid,
            "patches": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nVariance results saved to {args.output}")

    # Summary
    if results:
        variances = [r["full"]["variance"] for r in results]
        print(f"\n{'='*60}")
        print(f"  Variance summary across {len(results)} patches:")
        print(f"  Min: {min(variances):.4f}")
        print(f"  Max: {max(variances):.4f}")
        print(f"  Mean: {np.mean(variances):.4f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
