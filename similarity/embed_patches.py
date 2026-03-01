#!/usr/bin/env python3
"""
Embed image patches using CLIP.

Takes an image, splits it into a grid of patches, and computes CLIP embeddings
for each patch. Saves embeddings to a .npz file for later similarity search.

Usage:
    python3 embed_patches.py <image_path> [--grid 4] [--output patches.npz]
    python3 embed_patches.py <directory> [--grid 4] [--output database.npz]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import open_clip


def split_into_patches(image: Image.Image, grid_size: int = 4) -> list[Image.Image]:
    """Split an image into a grid of patches."""
    w, h = image.size
    patch_w = w // grid_size
    patch_h = h // grid_size
    patches = []
    for row in range(grid_size):
        for col in range(grid_size):
            box = (col * patch_w, row * patch_h, (col + 1) * patch_w, (row + 1) * patch_h)
            patches.append(image.crop(box))
    return patches


def embed_patches(patches: list[Image.Image], model, preprocess) -> np.ndarray:
    """Compute CLIP embeddings for a list of image patches."""
    inputs = torch.stack([preprocess(p) for p in patches])
    with torch.no_grad():
        features = model.encode_image(inputs)
        features /= features.norm(dim=-1, keepdim=True)
    return features.numpy()


def main():
    parser = argparse.ArgumentParser(description="Embed image patches with CLIP")
    parser.add_argument("path", help="Image file or directory of images")
    parser.add_argument("--grid", type=int, default=4, help="Grid size (default: 4, giving 16 patches)")
    parser.add_argument("--output", "-o", help="Output .npz file (default: auto-named)")
    args = parser.parse_args()

    path = Path(args.path)

    # Collect image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    if path.is_file():
        image_files = [path]
    elif path.is_dir():
        image_files = sorted(f for f in path.iterdir() if f.suffix.lower() in image_extensions)
        if not image_files:
            print(f"No images found in {path}")
            sys.exit(1)
    else:
        print(f"Path not found: {path}")
        sys.exit(1)

    print(f"Found {len(image_files)} image(s), grid size {args.grid}x{args.grid}")
    print("Loading CLIP model...")

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()

    all_embeddings = []
    all_metadata = []  # (source_file, patch_index, row, col)

    for i, img_file in enumerate(image_files):
        print(f"  [{i+1}/{len(image_files)}] {img_file.name}...", end=" ", flush=True)
        start = time.time()

        image = Image.open(img_file).convert("RGB")
        patches = split_into_patches(image, args.grid)
        embeddings = embed_patches(patches, model, preprocess)

        for idx, emb in enumerate(embeddings):
            row = idx // args.grid
            col = idx % args.grid
            all_embeddings.append(emb)
            all_metadata.append({
                "source": str(img_file),
                "patch_index": idx,
                "row": row,
                "col": col,
                "grid_size": args.grid,
            })

        elapsed = time.time() - start
        print(f"{len(patches)} patches in {elapsed:.1f}s")

    embeddings_array = np.array(all_embeddings)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif path.is_file():
        output_path = path.with_suffix('.npz')
    else:
        output_path = path / "patch_database.npz"

    # Save embeddings and metadata
    np.savez(
        output_path,
        embeddings=embeddings_array,
        metadata=np.array([str(m) for m in all_metadata]),  # JSON-encoded metadata
    )

    print(f"\nSaved {len(all_embeddings)} patch embeddings to {output_path}")
    print(f"Embedding shape: {embeddings_array.shape}")
    print(f"Database size: {output_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
