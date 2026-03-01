#!/usr/bin/env python3
"""
Find similar patches for each patch in an image.

Takes an image and a pre-built patch database, then for each patch in the image,
finds the most visually/semantically similar patches from the database.

This is the core of the similarity engine: "what could this patch be mistaken for?"

Usage:
    python3 find_similar.py <image_path> --database <database.npz> [--grid 4] [--top-k 5]
    python3 find_similar.py <image_path> --database <database.npz> --output similar.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import open_clip

from embed_patches import split_into_patches, embed_patches


def find_similar_patches(query_embeddings: np.ndarray, database_embeddings: np.ndarray,
                         database_metadata: list[dict], top_k: int = 5,
                         exclude_source: str = None) -> list[list[dict]]:
    """
    For each query patch, find the top-k most similar patches in the database.

    Returns a list of lists: for each query patch, a list of similar patches
    with their similarity scores and metadata.
    """
    results = []

    for i, query in enumerate(query_embeddings):
        # Cosine similarity (embeddings are already normalized)
        similarities = database_embeddings @ query

        # Get top matches
        indices = np.argsort(similarities)[::-1]

        matches = []
        for idx in indices:
            meta = database_metadata[idx]
            # Optionally exclude patches from the same source image
            if exclude_source and meta.get("source") == exclude_source:
                continue
            matches.append({
                "similarity": float(similarities[idx]),
                "source": meta.get("source", "unknown"),
                "patch_index": meta.get("patch_index", -1),
                "row": meta.get("row", -1),
                "col": meta.get("col", -1),
            })
            if len(matches) >= top_k:
                break

        results.append(matches)

    return results


def extract_patch_image(source_path: str, patch_index: int, grid_size: int) -> Image.Image:
    """Extract a specific patch from a source image."""
    image = Image.open(source_path).convert("RGB")
    patches = split_into_patches(image, grid_size)
    return patches[patch_index]


def main():
    parser = argparse.ArgumentParser(description="Find similar patches for an image")
    parser.add_argument("image", help="Input image to find similar patches for")
    parser.add_argument("--database", "-d", required=True, help="Path to patch database (.npz)")
    parser.add_argument("--grid", type=int, default=4, help="Grid size (default: 4)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of similar patches per query (default: 5)")
    parser.add_argument("--output", "-o", help="Save results as JSON")
    parser.add_argument("--export-patches", help="Directory to export similar patch images")
    parser.add_argument("--exclude-self", action="store_true", help="Exclude matches from the same source image")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    db_path = Path(args.database)
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        sys.exit(1)

    # Load database
    print("Loading patch database...")
    data = np.load(db_path, allow_pickle=True)
    db_embeddings = data["embeddings"]
    db_metadata = [eval(m) for m in data["metadata"]]  # Parse string-encoded dicts
    print(f"  {len(db_metadata)} patches in database")

    # Load model and embed query patches
    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()

    print(f"Splitting {image_path.name} into {args.grid}x{args.grid} patches...")
    image = Image.open(image_path).convert("RGB")
    patches = split_into_patches(image, args.grid)
    query_embeddings = embed_patches(patches, model, preprocess)

    # Find similar patches
    print(f"Finding top-{args.top_k} similar patches...")
    start = time.time()
    results = find_similar_patches(
        query_embeddings, db_embeddings, db_metadata,
        top_k=args.top_k,
        exclude_source=str(image_path) if args.exclude_self else None,
    )
    elapsed = time.time() - start

    # Display results
    for i, matches in enumerate(results):
        row = i // args.grid
        col = i % args.grid
        print(f"\nPatch [{row},{col}]:")
        for j, match in enumerate(matches):
            src = Path(match["source"]).name
            print(f"  {j+1}. {src} [{match['row']},{match['col']}] — similarity: {match['similarity']:.4f}")

    print(f"\nSearch completed in {elapsed:.2f}s")

    # Export similar patch images if requested
    if args.export_patches:
        export_dir = Path(args.export_patches)
        export_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting patch images to {export_dir}/...")
        for i, matches in enumerate(results):
            row = i // args.grid
            col = i % args.grid

            # Save the original patch
            patches[i].save(export_dir / f"patch_{row}_{col}_original.png")

            # Save similar patches
            for j, match in enumerate(matches):
                try:
                    similar = extract_patch_image(
                        match["source"], match["patch_index"],
                        db_metadata[0].get("grid_size", args.grid)
                    )
                    similar.save(export_dir / f"patch_{row}_{col}_similar_{j}_{match['similarity']:.3f}.png")
                except Exception as e:
                    print(f"  Warning: couldn't export {match['source']}: {e}")

        print(f"Exported to {export_dir}/")

    # Save JSON results if requested
    if args.output:
        output = {
            "source_image": str(image_path),
            "grid_size": args.grid,
            "patches": []
        }
        for i, matches in enumerate(results):
            output["patches"].append({
                "row": i // args.grid,
                "col": i % args.grid,
                "similar": matches,
            })
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
