#!/usr/bin/env python3
"""
Run the full patch-perception pipeline: image → CLIP → generate → caption → flicker HTML.

One command to go from a photograph to a self-contained flicker visualiser.

Usage:
    python3 run_pipeline.py photo.jpg
    python3 run_pipeline.py photo.jpg --name hedgehog --grid 4 --per-patch 2
    python3 run_pipeline.py photo.jpg --skip-captions  # faster, no vision model step
    python3 run_pipeline.py photo.jpg --only-interesting  # skip uniform background patches

Environment:
    REPLICATE_API_TOKEN=your_token_here

Output:
    Creates a directory with all intermediate files and a final flicker.html
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run_step(description: str, cmd: list[str]) -> bool:
    """Run a pipeline step, printing status."""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode})")
        return False
    return True


def filter_interesting_patches(desc_path: str, threshold: float = 0.26) -> str | None:
    """Return a --patches string of patches with interesting (non-background) descriptions.

    Patches where the top term is generic ('background', 'dark area', 'blurred area',
    'organic shape') and below the confidence threshold are skipped.
    """
    with open(desc_path) as f:
        data = json.load(f)

    boring_terms = {
        "background", "dark area", "blurred area", "organic shape",
        "smooth surface", "rough surface", "gradient", "foreground",
    }

    interesting = []
    for p in data["patches"]:
        descs = p.get("descriptions", [])
        if not descs:
            continue
        top_term = descs[0]["term"]
        top_score = descs[0].get("similarity", descs[0].get("score", 0))
        # Keep if the top term is specific OR confidence is high
        if top_term not in boring_terms or top_score >= threshold:
            interesting.append(f"{p['row']},{p['col']}")

    if not interesting:
        return None
    return " ".join(interesting)


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: image → flicker visualiser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 run_pipeline.py hedgehog.jpg
    python3 run_pipeline.py garden.jpg --name garden --per-patch 3
    python3 run_pipeline.py portrait.jpg --skip-captions --only-interesting
        """,
    )
    parser.add_argument("image", help="Input photograph")
    parser.add_argument("--name", help="Output name (default: derived from filename)")
    parser.add_argument("--grid", type=int, default=4, help="Grid size (default: 4)")
    parser.add_argument("--per-patch", type=int, default=2,
                        help="Alternatives per patch (default: 2)")
    parser.add_argument("--skip-captions", action="store_true",
                        help="Skip vision model captioning (faster)")
    parser.add_argument("--only-interesting", action="store_true",
                        help="Only generate for non-background patches (cheaper)")
    parser.add_argument("--caption-prompt", default=None,
                        help="Custom caption prompt (default: wild associations)")
    parser.add_argument("--output-dir", help="Output directory (default: <name>_output/)")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: {image_path} not found")
        sys.exit(1)

    if not os.environ.get("REPLICATE_API_TOKEN"):
        print("Error: Set REPLICATE_API_TOKEN environment variable")
        print("  export REPLICATE_API_TOKEN=r8_...")
        sys.exit(1)

    name = args.name or image_path.stem.replace(" ", "_").lower()
    out_dir = Path(args.output_dir) if args.output_dir else Path(f"{name}_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    desc_file = out_dir / f"{name}_desc.json"
    gen_dir = out_dir / "generated"
    manifest_file = gen_dir / "manifest.json"
    caption_file = out_dir / f"{name}_captions.json"
    html_file = out_dir / f"{name}-flicker.html"

    python = sys.executable

    print(f"Patch Perception Pipeline")
    print(f"  Image: {image_path}")
    print(f"  Grid: {args.grid}x{args.grid} ({args.grid**2} patches)")
    print(f"  Alternatives per patch: {args.per_patch}")
    print(f"  Output: {out_dir}/")

    # Step 1: CLIP descriptions (local, free)
    if not run_step(
        "Step 1/4: CLIP patch descriptions (local)",
        [python, "describe_patches.py", str(image_path),
         "--grid", str(args.grid), "--top-k", "10",
         "--output", str(desc_file)],
    ):
        sys.exit(1)

    # Determine which patches to process
    patches_arg = []
    if args.only_interesting:
        patch_str = filter_interesting_patches(str(desc_file))
        if patch_str:
            patches_arg = ["--patches", patch_str]
            n_interesting = len(patch_str.split())
            print(f"\n  Filtered to {n_interesting}/{args.grid**2} interesting patches")
        else:
            print("\n  No patches above threshold — processing all")

    # Step 2: Generate alternatives (Replicate API, ~$0.003/image)
    gen_cmd = [
        python, "generate_alternatives.py", str(desc_file),
        "--per-patch", str(args.per_patch),
        "--output", str(gen_dir),
    ] + patches_arg
    if not run_step(
        "Step 2/4: Generate alternatives (Replicate API)",
        gen_cmd,
    ):
        sys.exit(1)

    # Step 3: Vision model captions (optional, Replicate API)
    caption_arg = []
    if not args.skip_captions:
        wild_prompt = (
            "You are looking at a small cropped fragment from a larger photograph. "
            "You have no idea what the full image shows. Looking only at this fragment, "
            "list 5 wildly different things this fragment could be part of. "
            "Be creative and specific. Format: one per line, no numbering."
        )
        prompt = args.caption_prompt or wild_prompt
        caption_cmd = [
            python, "caption_patches.py", str(image_path),
            "--grid", str(args.grid),
            "--prompt", prompt,
            "--output", str(caption_file),
        ] + patches_arg
        if run_step(
            "Step 3/4: Creative captioning (Replicate API)",
            caption_cmd,
        ):
            caption_arg = ["--captions", str(caption_file)]
        else:
            print("  Captioning failed — continuing without captions")
    else:
        print(f"\n{'='*60}")
        print(f"  Step 3/4: Skipped (--skip-captions)")
        print(f"{'='*60}")

    # Step 4: Build flicker HTML (local, free)
    build_cmd = [
        python, "build_flicker.py", str(image_path), str(manifest_file),
        "--descriptions", str(desc_file),
        "--grid", str(args.grid),
        "--output", str(html_file),
    ] + caption_arg
    if not run_step(
        "Step 4/4: Build flicker HTML",
        build_cmd,
    ):
        sys.exit(1)

    # Summary
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"\n  Output: {html_file}")
    print(f"  Open in a browser to see the flicker effect.")

    # Cost estimate
    n_patches = args.grid ** 2
    if args.only_interesting and patches_arg:
        n_patches = len(patches_arg[1].split()) if len(patches_arg) > 1 else n_patches
    n_generations = n_patches * args.per_patch
    n_captions = 0 if args.skip_captions else n_patches
    gen_cost = n_generations * 0.003
    cap_cost = n_captions * 0.003
    print(f"\n  Estimated API cost: ~${gen_cost + cap_cost:.3f}")
    print(f"    ({n_generations} generations + {n_captions} captions)")


if __name__ == "__main__":
    main()
