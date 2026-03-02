"""Shared patch utilities that don't require torch/CLIP."""

from PIL import Image


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
