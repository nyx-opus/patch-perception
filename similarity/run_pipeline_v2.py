#!/usr/bin/env python3
"""
V2 Pipeline: image → 3-tier captions → text-first + img2img generation → CLIP variance → flicker HTML.

One command to go from a photograph to a v2 flicker visualiser with 16 images per patch.

Usage:
    python3 run_pipeline_v2.py photo.jpg
    python3 run_pipeline_v2.py photo.jpg --name hedgehog --grid 4
    python3 run_pipeline_v2.py photo.jpg --patches "1,1 2,3"  # specific patches only
    python3 run_pipeline_v2.py photo.jpg --skip-variance  # skip CLIP measurement

Environment:
    REPLICATE_API_TOKEN=your_token_here
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_step(description: str, cmd: list[str], timeout: int | None = None) -> bool:
    """Run a pipeline step, printing status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    kwargs = {}
    if timeout:
        kwargs["timeout"] = timeout
    try:
        result = subprocess.run(cmd, **kwargs)
        if result.returncode != 0:
            print(f"\n  FAILED (exit code {result.returncode})")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"\n  TIMEOUT")
        return False


def convert_manifest_v1(manifest_v2_path: Path, gen_dir: Path) -> Path:
    """Convert v2 manifest to v1-compatible format for build_flicker.py."""
    with open(manifest_v2_path) as f:
        v2 = json.load(f)

    v1 = {
        "source_image": v2["source_image"],
        "model": "v2-pipeline",
        "patches": [],
    }

    for patch in v2["patches"]:
        v1_patch = {
            "row": patch["row"],
            "col": patch["col"],
            "terms": [],
            "generated": [],
        }
        for gen in patch["generated"]:
            v1_patch["generated"].append({
                "filename": gen["filename"],
                "prompt": gen.get("prompt", ""),
                "terms": gen.get("tier", gen.get("subpath", "unknown")).split(","),
            })
        v1["patches"].append(v1_patch)

    v1_path = gen_dir / "manifest.json"
    with open(v1_path, "w") as f:
        json.dump(v1, f, indent=2)
    return v1_path


def main():
    parser = argparse.ArgumentParser(
        description="V2 Pipeline: image → captions → generation → variance → flicker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Input photograph")
    parser.add_argument("--name", help="Output name (default: derived from filename)")
    parser.add_argument("--grid", type=int, default=4, help="Grid size (default: 4)")
    parser.add_argument("--patches", help="Only process specific patches, e.g. '1,1 2,3'")
    parser.add_argument("--skip-variance", action="store_true",
                        help="Skip CLIP variance measurement (faster)")
    parser.add_argument("--skip-clip", action="store_true",
                        help="Skip initial CLIP descriptions (v1 compatibility)")
    parser.add_argument("--caption-model", default="yorickvp/llava-v1.6-mistral-7b",
                        help="Vision model for captioning")
    parser.add_argument("--output-dir", help="Output directory (default: <name>_v2_output/)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: {image_path} not found")
        sys.exit(1)

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: Set REPLICATE_API_TOKEN environment variable")
        sys.exit(1)

    name = args.name or image_path.stem.replace(" ", "_").lower()
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"{name}_v2_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    gen_dir = out_dir / "generated_v2"
    desc_file = out_dir / f"{name}_desc.json"
    caption_file = out_dir / f"{name}_captions_v2.json"
    variance_file = out_dir / f"{name}_variance.json"
    html_file = out_dir / f"{name}-flicker-v2.html"

    python = sys.executable
    patches_arg = ["--patches", args.patches] if args.patches else []

    n_patches = args.grid ** 2
    if args.patches:
        n_patches = len(args.patches.split())

    print(f"Patch Perception V2 Pipeline")
    print(f"  Image: {image_path}")
    print(f"  Grid: {args.grid}x{args.grid}")
    print(f"  Patches: {n_patches}")
    print(f"  Caption model: {args.caption_model}")
    print(f"  Output: {out_dir}/")

    # Estimate cost
    n_captions = n_patches * 3  # 3 tiers
    n_text_gen = n_patches * 9  # 3 per tier × 3 tiers
    n_img_gen = n_patches * 6   # 3 pure + 3 guided
    cost = n_captions * 0.002 + n_text_gen * 0.003 + n_img_gen * 0.008
    print(f"  Estimated cost: ~${cost:.2f}")
    print(f"    ({n_captions} captions + {n_text_gen} text-gen + {n_img_gen} img-gen)")

    # Step 1: CLIP descriptions (local, free) — for compatibility and initial understanding
    if not args.skip_clip:
        if not run_step(
            "Step 1/5: CLIP patch descriptions (local)",
            [python, "describe_patches.py", str(image_path),
             "--grid", str(args.grid), "--top-k", "10",
             "--output", str(desc_file)],
        ):
            print("  CLIP step failed — continuing without descriptions")

    # Step 2: 3-tier captioning (Replicate API)
    if not run_step(
        "Step 2/5: 3-tier vision captioning (Replicate API)",
        [python, "caption_v2.py", str(image_path),
         "--grid", str(args.grid),
         "--model", args.caption_model,
         "--output", str(caption_file)] + patches_arg,
    ):
        print("  Captioning failed")
        sys.exit(1)

    # Step 3: V2 generation — text-first + image-first (Replicate API)
    if not run_step(
        "Step 3/5: V2 generation — text-first + image-first (Replicate API)",
        [python, "generate_v2.py", str(caption_file), str(image_path),
         "--grid", str(args.grid),
         "--output", str(gen_dir)] + patches_arg,
        timeout=1800,  # 30 min max for generation
    ):
        print("  Generation failed or timed out")
        sys.exit(1)

    # Step 4: CLIP variance measurement (local)
    manifest_v2 = gen_dir / "manifest_v2.json"
    if not args.skip_variance and manifest_v2.exists():
        run_step(
            "Step 4/5: CLIP variance measurement (local)",
            [python, "measure_variance.py", str(manifest_v2), str(image_path),
             "--grid", str(args.grid),
             "--output", str(variance_file)],
        )
    else:
        print(f"\n{'='*60}")
        print(f"  Step 4/5: Skipped")
        print(f"{'='*60}")

    # Step 5: Build flicker HTML
    # Convert v2 manifest to v1-compatible format for existing flicker builder
    v1_manifest = convert_manifest_v1(manifest_v2, gen_dir)

    build_cmd = [
        python, "build_flicker.py", str(image_path), str(v1_manifest),
        "--grid", str(args.grid),
        "--output", str(html_file),
    ]
    if desc_file.exists():
        build_cmd += ["--descriptions", str(desc_file)]
    if variance_file.exists():
        build_cmd += ["--variance", str(variance_file)]

    if not run_step(
        "Step 5/5: Build flicker HTML",
        build_cmd,
    ):
        print("  Flicker build failed")
        sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"  V2 PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Flicker: {html_file}")
    if variance_file.exists():
        print(f"  Variance: {variance_file}")
    print(f"  Captions: {caption_file}")
    print(f"  Generated: {gen_dir}/")
    print(f"\n  Open {html_file} in a browser to see the result.")


if __name__ == "__main__":
    main()
