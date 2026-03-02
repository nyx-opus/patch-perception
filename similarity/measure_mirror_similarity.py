#!/usr/bin/env python3
"""
Measure CLIP similarity between original patches and their mirror experiment
outputs, then correlate with CLIP variance to test the confidence landscape
mirror hypothesis.

Run after confidence_mirror_experiment.py generates the img2img outputs.
Must use the CLIP venv: /tmp/clip_env/bin/python3 measure_mirror_similarity.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import open_clip
from PIL import Image


def main():
    experiment_dir = Path(__file__).parent / "mirror_experiment"
    export_dir = Path(__file__).parent / "hedgehog_export"
    results_path = experiment_dir / "mirror_results.json"

    if not results_path.exists():
        print("No mirror_results.json found. Run confidence_mirror_experiment.py first.")
        sys.exit(1)

    results = json.load(open(results_path))

    # Load CLIP
    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()

    similarities = []

    for entry in results["results"]:
        r, c = entry["row"], entry["col"]
        orig_path = export_dir / f"patch_{r}_{c}_original.png"
        mirror_path = Path(entry["output"])

        if not orig_path.exists() or not mirror_path.exists():
            print(f"  [{r},{c}] missing files, skipping")
            continue

        # Encode both images
        orig_img = preprocess(Image.open(orig_path).convert("RGB")).unsqueeze(0)
        mirror_img = preprocess(Image.open(mirror_path).convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            orig_feat = model.encode_image(orig_img)
            mirror_feat = model.encode_image(mirror_img)
            # Normalise
            orig_feat = orig_feat / orig_feat.norm(dim=-1, keepdim=True)
            mirror_feat = mirror_feat / mirror_feat.norm(dim=-1, keepdim=True)
            # Cosine similarity
            sim = (orig_feat @ mirror_feat.T).item()

        entry["clip_similarity"] = round(sim, 4)
        similarities.append(sim)
        print(f"  [{r},{c}] variance={entry['variance']:.4f}  "
              f"similarity={sim:.4f}  "
              f"(transform={1-sim:.4f})")

    # Compute correlation
    if len(similarities) >= 4:
        variances = [e["variance"] for e in results["results"] if "clip_similarity" in e]
        sims = [e["clip_similarity"] for e in results["results"] if "clip_similarity" in e]

        var_arr = np.array(variances)
        sim_arr = np.array(sims)
        transform_arr = 1.0 - sim_arr  # higher = more transformed

        # Pearson correlation: variance vs transformation
        r_val = np.corrcoef(var_arr, transform_arr)[0, 1]

        print(f"\n{'='*60}")
        print(f"RESULTS: Confidence Landscape Mirror Experiment")
        print(f"{'='*60}")
        print(f"Patches tested: {len(sims)}")
        print(f"Variance range: {min(variances):.4f} – {max(variances):.4f}")
        print(f"Similarity range: {min(sims):.4f} – {max(sims):.4f}")
        print(f"Transform range: {min(1-s for s in sims):.4f} – {max(1-s for s in sims):.4f}")
        print(f"\nPearson r (variance vs transformation): {r_val:.4f}")
        print(f"  r > 0: uncertain patches ARE more transformed (hypothesis supported)")
        print(f"  r > 0.5: strong support for confidence landscape mirror")
        print(f"  r < 0: hypothesis NOT supported")

        # Save enriched results
        results["correlation"] = round(r_val, 4)
        results["n_patches"] = len(sims)
        json.dump(results, open(results_path, "w"), indent=2)
        print(f"\nUpdated {results_path} with similarity and correlation data")
    else:
        print("Not enough data points for correlation")


if __name__ == "__main__":
    main()
