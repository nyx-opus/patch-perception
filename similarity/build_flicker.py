#!/usr/bin/env python3
"""
Build a self-contained HTML flicker visualiser from a source image and generated alternatives.

Takes the source image, splits it into patches, loads the manifest of generated alternatives,
and produces a single HTML file with everything embedded as base64 data URIs.

The result: open one HTML file, see each patch flickering between what it actually is
and what the model thinks it could be. Perceptual ambiguity made visceral.

Usage:
    python3 build_flicker.py <source_image> <manifest.json> [--output flicker.html]
    python3 build_flicker.py <source_image> <manifest.json> --descriptions desc.json
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image
from embed_patches import split_into_patches


def image_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    """Convert PIL image to base64 data URI."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def build_flicker_html(
    source_path: str,
    manifest_path: str,
    descriptions_path: str | None = None,
    grid_size: int = 4,
) -> str:
    """Build the complete HTML string with embedded images."""

    source = Image.open(source_path).convert("RGB")
    manifest = json.loads(Path(manifest_path).read_text())
    gen_dir = Path(manifest_path).parent

    descriptions = None
    if descriptions_path:
        descriptions = json.loads(Path(descriptions_path).read_text())

    # Split source into patches
    patches = split_into_patches(source, grid_size)

    # Build patch data: base64 of each original patch
    patch_data = []
    for i, patch in enumerate(patches):
        row = i // grid_size
        col = i % grid_size
        patch_data.append({
            "row": row,
            "col": col,
            "original": image_to_b64(patch),
            "alternatives": [],
            "terms": [],
        })

    # Load descriptions if available
    if descriptions:
        desc_map = {}
        for p in descriptions.get("patches", []):
            desc_map[(p["row"], p["col"])] = [
                d["term"] for d in p.get("descriptions", [])
            ]
        for pd in patch_data:
            key = (pd["row"], pd["col"])
            if key in desc_map:
                pd["terms"] = desc_map[key]

    # Load generated alternatives from manifest
    alt_map = {}
    for entry in manifest.get("patches", []):
        key = (entry["row"], entry["col"])
        alts = []
        for gen in entry.get("generated", []):
            img_path = gen_dir / gen["filename"]
            if img_path.exists():
                alt_img = Image.open(img_path).convert("RGB")
                # Resize alternative to match patch size
                patch_idx = entry["row"] * grid_size + entry["col"]
                orig_patch = patches[patch_idx]
                alt_img = alt_img.resize(orig_patch.size, Image.LANCZOS)
                alts.append({
                    "src": image_to_b64(alt_img),
                    "terms": gen.get("terms", []),
                    "prompt": gen.get("prompt", ""),
                })
            else:
                print(f"  Warning: {img_path} not found, skipping")
        alt_map[key] = alts

    for pd in patch_data:
        key = (pd["row"], pd["col"])
        if key in alt_map:
            pd["alternatives"] = alt_map[key]

    # Source image as base64 for the overview
    source_b64 = image_to_b64(source)

    # Count how many patches have alternatives
    n_with_alts = sum(1 for pd in patch_data if pd["alternatives"])

    # Build the HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Patch Perception — Flicker</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}

  body {{
    background: #0a0a0f;
    color: #e0e0e8;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem 1rem;
  }}

  .header {{
    text-align: center;
    margin-bottom: 1.5rem;
  }}

  .header h1 {{
    font-size: 1.8rem;
    font-weight: 300;
    letter-spacing: 0.05em;
    color: #c8b8e8;
  }}

  .header .subtitle {{
    font-size: 0.9rem;
    color: #888;
    margin-top: 0.3rem;
    font-style: italic;
  }}

  .controls {{
    display: flex;
    gap: 1.5rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
    justify-content: center;
    align-items: center;
  }}

  .control-group {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}

  .control-group label {{
    font-size: 0.8rem;
    color: #888;
  }}

  .control-group input[type="range"] {{
    width: 120px;
    accent-color: #c8b8e8;
  }}

  .control-group span {{
    font-size: 0.75rem;
    color: #aaa;
    min-width: 3em;
  }}

  .mode-btn {{
    background: #1a1a2e;
    border: 1px solid #2a2a4e;
    color: #888;
    padding: 0.5rem 1rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8rem;
    transition: all 0.3s ease;
  }}

  .mode-btn:hover {{
    border-color: #c8b8e8;
    color: #c8b8e8;
  }}

  .mode-btn.active {{
    background: #2a1a4e;
    border-color: #c8b8e8;
    color: #e0d0f8;
  }}

  .grid-container {{
    position: relative;
    display: inline-block;
    border: 1px solid #2a2a4e;
    border-radius: 4px;
    overflow: hidden;
    cursor: crosshair;
  }}

  .grid-container canvas {{
    display: block;
  }}

  .tooltip {{
    position: fixed;
    background: rgba(10, 10, 15, 0.95);
    border: 1px solid #c8b8e8;
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
    color: #ddd;
    pointer-events: none;
    z-index: 100;
    max-width: 280px;
    line-height: 1.4;
    display: none;
  }}

  .tooltip .patch-label {{
    color: #c8b8e8;
    font-weight: 500;
    margin-bottom: 0.3rem;
  }}

  .tooltip .terms {{
    color: #aaa;
  }}

  .tooltip .term {{
    display: inline-block;
    background: rgba(200, 184, 232, 0.15);
    padding: 1px 6px;
    border-radius: 3px;
    margin: 2px 2px;
    font-size: 0.75rem;
  }}

  .tooltip .alt-label {{
    color: #e8a878;
    margin-top: 0.3rem;
    font-size: 0.75rem;
  }}

  .info {{
    margin-top: 1.5rem;
    text-align: center;
    max-width: 600px;
    font-size: 0.85rem;
    line-height: 1.6;
    color: #888;
  }}

  .info em {{
    color: #c8b8e8;
    font-style: normal;
  }}

  .footer {{
    margin-top: 2rem;
    font-size: 0.7rem;
    color: #555;
  }}
</style>
</head>
<body>

<div class="header">
  <h1>Patch Perception</h1>
  <div class="subtitle">{n_with_alts} patches with generated alternatives — hover to see what the model thinks</div>
</div>

<div class="controls">
  <button class="mode-btn active" data-mode="flicker">Flicker</button>
  <button class="mode-btn" data-mode="dissolve">Dissolve</button>
  <button class="mode-btn" data-mode="original">Original</button>
  <button class="mode-btn" data-mode="grid">Grid</button>
  <button class="mode-btn" data-mode="side-by-side">Side by Side</button>

  <div class="control-group">
    <label>Speed:</label>
    <input type="range" id="speed" min="200" max="3000" value="800" step="100">
    <span id="speed-val">0.8s</span>
  </div>

  <div class="control-group">
    <label>Gap:</label>
    <input type="range" id="gap" min="0" max="4" value="1" step="1">
    <span id="gap-val">1px</span>
  </div>
</div>

<div class="grid-container">
  <canvas id="main-canvas"></canvas>
</div>

<div class="tooltip" id="tooltip"></div>

<div class="info">
  Each patch was independently described by CLIP — ranked against 169 vocabulary terms by visual similarity.
  For patches with high ambiguity, we asked: <em>what else could this be?</em> and generated alternatives.
  The flicker shows you the perceptual boundary — the moment where hedgehog spines become feathers,
  where a snout becomes any animal's face, where texture dissolves into material.
</div>

<div class="footer">Nyx & Amy — patch-perception 2026</div>

<script>
const GRID = {grid_size};
const PATCHES = {json.dumps(patch_data)};
const SOURCE = "{source_b64}";

const canvas = document.getElementById('main-canvas');
const ctx = canvas.getContext('2d');
const tooltip = document.getElementById('tooltip');

let mode = 'flicker';
let flickerInterval = null;
let flickerState = new Map(); // patch index -> {{ altIdx, showingAlt }}
let patchImages = new Map(); // cache loaded images
let sourceImg = null;
let gap = 1;

// Preload source image
function loadSourceImage() {{
  return new Promise(resolve => {{
    const img = new Image();
    img.onload = () => {{
      sourceImg = img;
      resolve(img);
    }};
    img.src = SOURCE;
  }});
}}

// Preload all patch images (original + alternatives)
async function preloadPatchImages() {{
  const promises = [];
  PATCHES.forEach((p, i) => {{
    // Original
    const origP = new Promise(resolve => {{
      const img = new Image();
      img.onload = () => {{
        patchImages.set(`${{i}}_orig`, img);
        resolve();
      }};
      img.src = p.original;
    }});
    promises.push(origP);

    // Alternatives
    p.alternatives.forEach((alt, ai) => {{
      const altP = new Promise(resolve => {{
        const img = new Image();
        img.onload = () => {{
          patchImages.set(`${{i}}_alt_${{ai}}`, img);
          resolve();
        }};
        img.src = alt.src;
      }});
      promises.push(altP);
    }});

    // Init flicker state
    if (p.alternatives.length > 0) {{
      flickerState.set(i, {{
        altIdx: 0,
        showingAlt: false,
        blend: 0, // 0 = original, 1 = alternative (for dissolve mode)
        // Each patch flickers at a slightly different phase
        phase: Math.random() * Math.PI * 2,
      }});
    }}
  }});
  await Promise.all(promises);
}}

function getPatchSize() {{
  if (!sourceImg) return {{ w: 0, h: 0 }};
  return {{
    w: Math.ceil(sourceImg.width / GRID),
    h: Math.ceil(sourceImg.height / GRID),
  }};
}}

function render() {{
  if (!sourceImg) return;
  const ps = getPatchSize();

  canvas.width = sourceImg.width + gap * (GRID - 1);
  canvas.height = sourceImg.height + gap * (GRID - 1);

  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (mode === 'original') {{
    ctx.drawImage(sourceImg, 0, 0);
    return;
  }}

  if (mode === 'side-by-side') {{
    renderSideBySide(ps);
    return;
  }}

  // Grid, Flicker, or Dissolve mode
  PATCHES.forEach((p, i) => {{
    const dx = p.col * (ps.w + gap);
    const dy = p.row * (ps.h + gap);

    if (mode === 'dissolve' && flickerState.has(i)) {{
      // Crossfade: draw original at (1-blend), alt at blend
      const state = flickerState.get(i);
      const origImg = patchImages.get(`${{i}}_orig`);
      const altImg = patchImages.get(`${{i}}_alt_${{state.altIdx}}`);

      if (origImg) {{
        ctx.globalAlpha = 1 - state.blend;
        ctx.drawImage(origImg, dx, dy, ps.w, ps.h);
      }}
      if (altImg && state.blend > 0.01) {{
        ctx.globalAlpha = state.blend;
        ctx.drawImage(altImg, dx, dy, ps.w, ps.h);
      }}
      ctx.globalAlpha = 1;
    }} else {{
      let imgKey = `${{i}}_orig`;

      if (mode === 'flicker' && flickerState.has(i)) {{
        const state = flickerState.get(i);
        if (state.showingAlt) {{
          imgKey = `${{i}}_alt_${{state.altIdx}}`;
        }}
      }}

      const img = patchImages.get(imgKey);
      if (img) {{
        ctx.drawImage(img, dx, dy, ps.w, ps.h);
      }}

      // Subtle border for patches showing alternatives
      if (mode === 'flicker' && flickerState.has(i)) {{
        const state = flickerState.get(i);
        if (state.showingAlt) {{
          ctx.strokeStyle = 'rgba(232, 168, 120, 0.6)';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(dx + 0.5, dy + 0.5, ps.w - 1, ps.h - 1);
        }}
      }}
    }}
  }});
}}

function renderSideBySide(ps) {{
  // Show original on left, all-alternatives on right
  const totalW = sourceImg.width * 2 + gap * (GRID - 1) * 2 + 20;
  const totalH = sourceImg.height + gap * (GRID - 1);
  canvas.width = totalW;
  canvas.height = totalH;

  ctx.fillStyle = '#0a0a0f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // Left: original patches
  PATCHES.forEach((p, i) => {{
    const dx = p.col * (ps.w + gap);
    const dy = p.row * (ps.h + gap);
    const img = patchImages.get(`${{i}}_orig`);
    if (img) ctx.drawImage(img, dx, dy, ps.w, ps.h);
  }});

  // Right: alternatives where available, originals elsewhere
  const offsetX = sourceImg.width + gap * (GRID - 1) + 20;
  PATCHES.forEach((p, i) => {{
    const dx = offsetX + p.col * (ps.w + gap);
    const dy = p.row * (ps.h + gap);

    if (flickerState.has(i)) {{
      const state = flickerState.get(i);
      const img = patchImages.get(`${{i}}_alt_${{state.altIdx}}`);
      if (img) {{
        ctx.drawImage(img, dx, dy, ps.w, ps.h);
        ctx.strokeStyle = 'rgba(232, 168, 120, 0.4)';
        ctx.lineWidth = 1;
        ctx.strokeRect(dx, dy, ps.w, ps.h);
      }}
    }} else {{
      const img = patchImages.get(`${{i}}_orig`);
      if (img) {{
        ctx.globalAlpha = 0.4;
        ctx.drawImage(img, dx, dy, ps.w, ps.h);
        ctx.globalAlpha = 1;
      }}
    }}
  }});

  // Labels
  ctx.fillStyle = '#888';
  ctx.font = '12px system-ui';
  ctx.fillText('Original', 4, totalH - 4);
  ctx.fillText('What else it could be', offsetX + 4, totalH - 4);
}}

function startFlicker() {{
  stopFlicker();
  const speed = parseInt(document.getElementById('speed').value);

  flickerInterval = setInterval(() => {{
    const now = Date.now();
    flickerState.forEach((state, i) => {{
      const patch = PATCHES[i];
      // Use sine wave with unique phase for organic feel
      const t = (now / speed + state.phase) % (Math.PI * 2);

      if (mode === 'dissolve') {{
        // Smooth sine blend: 0 to 1 and back
        const raw = (Math.sin(t) + 1) / 2; // 0..1
        const prev = state.blend;
        state.blend = raw;
        // Cycle alternative when crossing back through zero
        if (raw < 0.05 && prev > 0.05) {{
          state.altIdx = (state.altIdx + 1) % patch.alternatives.length;
        }}
      }} else {{
        // Hard flicker
        const shouldShowAlt = Math.sin(t) > 0.2;
        if (shouldShowAlt !== state.showingAlt) {{
          state.showingAlt = shouldShowAlt;
          if (shouldShowAlt) {{
            state.altIdx = (state.altIdx + 1) % patch.alternatives.length;
          }}
        }}
      }}
    }});
    render();
  }}, 50); // 20fps render loop, flicker timing controlled by speed
}}

function stopFlicker() {{
  if (flickerInterval) {{
    clearInterval(flickerInterval);
    flickerInterval = null;
  }}
}}

// Tooltip on hover
canvas.addEventListener('mousemove', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  const ps = getPatchSize();

  // Find which patch we're over
  const col = Math.floor(x / (ps.w + gap));
  const row = Math.floor(y / (ps.h + gap));

  if (col >= 0 && col < GRID && row >= 0 && row < GRID) {{
    const idx = row * GRID + col;
    const patch = PATCHES[idx];

    let html = `<div class="patch-label">Patch [${{row}},${{col}}]</div>`;
    if (patch.terms.length > 0) {{
      html += '<div class="terms">';
      patch.terms.forEach(t => {{
        html += `<span class="term">${{t}}</span>`;
      }});
      html += '</div>';
    }}
    if (patch.alternatives.length > 0) {{
      const state = flickerState.get(idx);
      const alt = patch.alternatives[state ? state.altIdx : 0];
      html += `<div class="alt-label">${{patch.alternatives.length}} alternative(s) generated</div>`;
    }}

    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.clientX + 15) + 'px';
    tooltip.style.top = (e.clientY + 15) + 'px';
  }} else {{
    tooltip.style.display = 'none';
  }}
}});

canvas.addEventListener('mouseleave', () => {{
  tooltip.style.display = 'none';
}});

// Click to pause/resume flicker on specific patch
canvas.addEventListener('click', (e) => {{
  if (mode !== 'flicker') return;
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (canvas.width / rect.width);
  const y = (e.clientY - rect.top) * (canvas.height / rect.height);
  const ps = getPatchSize();

  const col = Math.floor(x / (ps.w + gap));
  const row = Math.floor(y / (ps.h + gap));
  const idx = row * GRID + col;

  if (flickerState.has(idx)) {{
    const state = flickerState.get(idx);
    // Toggle: if showing alt, snap to original; if original, snap to next alt
    state.showingAlt = !state.showingAlt;
    if (state.showingAlt) {{
      state.altIdx = (state.altIdx + 1) % PATCHES[idx].alternatives.length;
    }}
    render();
  }}
}});

// Mode buttons
document.querySelectorAll('.mode-btn').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    mode = btn.dataset.mode;

    if (mode === 'flicker' || mode === 'dissolve') {{
      startFlicker();
    }} else {{
      stopFlicker();
      // Reset all to original for non-flicker modes
      flickerState.forEach(state => {{ state.showingAlt = false; state.blend = 0; }});
      render();
    }}
  }});
}});

// Speed control
document.getElementById('speed').addEventListener('input', (e) => {{
  document.getElementById('speed-val').textContent = (e.target.value / 1000).toFixed(1) + 's';
  if (mode === 'flicker') startFlicker();
}});

// Gap control
document.getElementById('gap').addEventListener('input', (e) => {{
  gap = parseInt(e.target.value);
  document.getElementById('gap-val').textContent = gap + 'px';
  render();
}});

// Init
(async () => {{
  await loadSourceImage();
  await preloadPatchImages();
  render();
  startFlicker();
}})();
</script>
</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(
        description="Build a self-contained HTML flicker visualiser"
    )
    parser.add_argument("image", help="Source image")
    parser.add_argument("manifest", help="Generated alternatives manifest (JSON)")
    parser.add_argument("--descriptions", "-d", help="CLIP descriptions JSON")
    parser.add_argument("--grid", type=int, default=4, help="Grid size (default: 4)")
    parser.add_argument("--output", "-o", default="flicker.html",
                        help="Output HTML file (default: flicker.html)")
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Image not found: {args.image}")
        sys.exit(1)
    if not Path(args.manifest).exists():
        print(f"Manifest not found: {args.manifest}")
        sys.exit(1)

    print(f"Building flicker visualiser...")
    print(f"  Source: {args.image}")
    print(f"  Manifest: {args.manifest}")
    print(f"  Grid: {args.grid}x{args.grid}")

    html = build_flicker_html(
        args.image,
        args.manifest,
        args.descriptions,
        args.grid,
    )

    output_path = Path(args.output)
    output_path.write_text(html)
    size_kb = output_path.stat().st_size / 1024
    print(f"\nWritten to {output_path} ({size_kb:.0f} KB)")
    print(f"Open in a browser to see the flicker effect.")


if __name__ == "__main__":
    main()
