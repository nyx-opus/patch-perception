#!/usr/bin/env python3
"""
Describe image patches using CLIP zero-shot classification.

For each patch, ranks a large vocabulary of descriptive terms by CLIP similarity.
This tells us what the model "thinks" each patch could be — the raw material
for finding genuinely confusable alternatives.

Usage:
    python3 describe_patches.py <image_path> [--grid 4] [--top-k 10]
    python3 describe_patches.py <image_path> --output descriptions.json
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

# A broad vocabulary of visual concepts — things patches might look like.
# Deliberately diverse: textures, objects, scenes, materials, animals, colours.
VOCABULARY = [
    # Textures and materials
    "fur", "feathers", "scales", "bark", "wood grain", "fabric", "wool", "silk",
    "leather", "metal surface", "rust", "concrete", "brick", "stone", "gravel",
    "sand", "soil", "mud", "grass", "moss", "lichen", "ice", "snow", "water",
    "glass", "plastic", "paper", "cardboard", "rope", "wire", "mesh",
    # Natural elements
    "leaves", "petals", "branches", "roots", "thorns", "seeds", "mushroom",
    "coral", "shell", "bone", "horn", "claw", "tooth", "eye", "wing",
    "sky", "clouds", "sunlight", "shadow", "moonlight", "fog", "rain",
    # Animals and creatures
    "hedgehog", "cat", "dog", "bird", "parrot", "owl", "mouse", "rabbit",
    "squirrel", "fox", "deer", "horse", "cow", "sheep", "pig", "chicken",
    "fish", "frog", "snake", "spider", "bee", "butterfly", "beetle",
    "animal face", "animal body", "animal fur", "paw", "tail", "beak", "snout",
    # Human-made objects
    "book", "bottle", "cup", "plate", "bowl", "knife", "key", "coin",
    "button", "wheel", "door", "window", "wall", "floor", "ceiling", "roof",
    "chair", "table", "shelf", "lamp", "clock", "mirror", "frame",
    "clothing", "shoe", "hat", "bag", "blanket", "towel", "curtain",
    # Scenes and spaces
    "garden", "forest", "field", "mountain", "river", "lake", "ocean", "beach",
    "street", "building", "church", "castle", "bridge", "fence", "path",
    "kitchen", "bedroom", "bathroom", "living room", "workshop", "office",
    # Colours and patterns
    "dark area", "bright area", "blurred area", "sharp detail",
    "striped pattern", "spotted pattern", "checkered pattern",
    "warm colours", "cool colours", "earthy tones", "pastel colours",
    "gradient", "edge", "corner", "curve", "straight line",
    # Abstract / ambiguous
    "texture", "pattern", "surface", "background", "foreground",
    "organic shape", "geometric shape", "rough surface", "smooth surface",
    "tangled", "layered", "crumpled", "folded", "woven",
    "spiky", "fluffy", "glossy", "matte", "translucent",
]


def describe_patch(patch_embedding: np.ndarray, text_features: np.ndarray,
                   vocabulary: list[str], top_k: int = 10) -> list[dict]:
    """Rank vocabulary terms by similarity to a patch embedding."""
    similarities = text_features @ patch_embedding
    indices = np.argsort(similarities)[::-1][:top_k]
    return [{"term": vocabulary[i], "similarity": float(similarities[i])} for i in indices]


def main():
    parser = argparse.ArgumentParser(description="Describe image patches with CLIP")
    parser.add_argument("image", help="Input image")
    parser.add_argument("--grid", type=int, default=4, help="Grid size (default: 4)")
    parser.add_argument("--top-k", type=int, default=10, help="Top terms per patch (default: 10)")
    parser.add_argument("--output", "-o", help="Save results as JSON")
    parser.add_argument("--extra-vocab", help="File with additional vocabulary terms (one per line)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    vocab = VOCABULARY.copy()
    if args.extra_vocab:
        with open(args.extra_vocab) as f:
            extras = [line.strip() for line in f if line.strip()]
            vocab.extend(extras)
            print(f"Added {len(extras)} extra vocabulary terms")

    print(f"Vocabulary: {len(vocab)} terms")
    print("Loading CLIP model...")

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

    # Pre-compute text embeddings for entire vocabulary
    print("Encoding vocabulary...")
    prompts = [f"a photo of {term}" for term in vocab]
    with torch.no_grad():
        text_tokens = tokenizer(prompts)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    text_features = text_features.numpy()

    # Split image and embed patches
    print(f"Splitting {image_path.name} into {args.grid}x{args.grid} patches...")
    image = Image.open(image_path).convert("RGB")
    patches = split_into_patches(image, args.grid)

    inputs = torch.stack([preprocess(p) for p in patches])
    with torch.no_grad():
        patch_features = model.encode_image(inputs)
        patch_features /= patch_features.norm(dim=-1, keepdim=True)
    patch_features = patch_features.numpy()

    # Describe each patch
    results = []
    for i in range(len(patches)):
        row = i // args.grid
        col = i % args.grid
        descriptions = describe_patch(patch_features[i], text_features, vocab, args.top_k)
        results.append({"row": row, "col": col, "descriptions": descriptions})

        print(f"\nPatch [{row},{col}]:")
        for d in descriptions:
            bar = "#" * int(d["similarity"] * 40)
            print(f"  {d['similarity']:.3f} {bar} {d['term']}")

    # Save JSON if requested
    if args.output:
        output = {
            "source_image": str(image_path),
            "grid_size": args.grid,
            "vocabulary_size": len(vocab),
            "patches": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
