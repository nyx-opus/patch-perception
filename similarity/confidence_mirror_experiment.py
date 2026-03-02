#!/usr/bin/env python3
"""
Confidence Landscape Mirror Experiment

Test whether CLIP perception uncertainty correlates with how much a diffusion
model transforms an image — i.e., whether perception and generation share the
same confidence landscape.

Hypothesis: patches with HIGH CLIP variance (uncertain) will be MORE transformed
by img2img (lower similarity to original) because they occupy sparse regions
of the shared visual feature space.

Uses hedgehog patches with known variance data.
"""

import json
import sys
import time
from pathlib import Path

from PIL import Image

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from img2img import generate_img2img


def load_patches(export_dir: str, grid_size: int = 4):
    """Load original patches from export directory."""
    patches = {}
    export = Path(export_dir)
    for r in range(grid_size):
        for c in range(grid_size):
            path = export / f"patch_{r}_{c}_original.png"
            if path.exists():
                patches[(r, c)] = Image.open(path).convert("RGB")
    return patches


def load_variance(variance_path: str):
    """Load variance data from JSON."""
    data = json.load(open(variance_path))
    result = {}
    for p in data.get("patches", []):
        full = p.get("full", p)
        result[(p["row"], p["col"])] = {
            "variance": full.get("variance", 0),
            "mean_similarity": full.get("mean_similarity", 0),
        }
    return result


def run_experiment(
    export_dir: str,
    variance_path: str,
    output_dir: str,
    prompt: str = "an image",
    strength: float = 0.7,
    grid_size: int = 4,
):
    """Run the mirror experiment."""
    patches = load_patches(export_dir, grid_size)
    variance = load_variance(variance_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(patches)} patches, {len(variance)} variance entries")
    print(f"Prompt: '{prompt}', strength: {strength}")
    print()

    results = []

    for (r, c), patch_img in sorted(patches.items()):
        if (r, c) not in variance:
            continue

        var_data = variance[(r, c)]
        var = var_data["variance"]

        print(f"[{r},{c}] variance={var:.4f} — generating...", end=" ", flush=True)

        # Generate img2img with minimal prompt
        result_img = generate_img2img(patch_img, prompt, strength=strength,
                                       guidance_scale=3.0)  # low guidance = more prior

        if result_img is None:
            print("FAILED")
            continue

        # Save the output
        result_path = out / f"mirror_{r}_{c}.png"
        result_img.save(result_path)
        print(f"saved → {result_path.name}")

        results.append({
            "row": r,
            "col": c,
            "variance": var,
            "mean_similarity": var_data["mean_similarity"],
            "output": str(result_path),
        })

        # Rate limit
        time.sleep(2)

    # Save results for CLIP analysis
    results_path = out / "mirror_results.json"
    json.dump({"prompt": prompt, "strength": strength, "results": results},
              open(results_path, "w"), indent=2)
    print(f"\nSaved {len(results)} results to {results_path}")
    print("\nNext step: run CLIP similarity between originals and outputs")
    print("  /tmp/clip_env/bin/python3 measure_mirror_similarity.py")


if __name__ == "__main__":
    run_experiment(
        export_dir=str(Path(__file__).parent / "hedgehog_export"),
        variance_path="/mnt/file_server/Nyx/hedgehog-v2/hedgehog_variance.json",
        output_dir=str(Path(__file__).parent / "mirror_experiment"),
        prompt="an image",
        strength=0.7,
    )
