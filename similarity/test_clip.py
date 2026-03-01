#!/usr/bin/env python3
"""Quick test that CLIP is installed and working."""

import time
print("Loading CLIP model (first run downloads ~400MB)...")
start = time.time()

import open_clip
import torch
from PIL import Image

# Load the smallest CLIP model
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()
tokenizer = open_clip.get_tokenizer('ViT-B-32')

load_time = time.time() - start
print(f"Model loaded in {load_time:.1f}s")

# Test with a text query
with torch.no_grad():
    text = tokenizer(["a photo of a cat", "a photo of a garden", "a kitchen counter"])
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

print(f"Text embedding shape: {text_features.shape}")
print(f"Embedding dimension: {text_features.shape[1]}")

# Test with a random image
import numpy as np
dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
image_input = preprocess(dummy_image).unsqueeze(0)

with torch.no_grad():
    image_features = model.encode_image(image_input)
    image_features /= image_features.norm(dim=-1, keepdim=True)

print(f"Image embedding shape: {image_features.shape}")

# Test similarity between text and image
similarity = (text_features @ image_features.T).squeeze()
print(f"\nSimilarity scores (random image vs text):")
for label, score in zip(["cat", "garden", "kitchen"], similarity.tolist()):
    print(f"  {label}: {score:.4f}")

print(f"\nCLIP is working! Ready for patch similarity.")
