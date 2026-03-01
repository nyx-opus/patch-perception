#!/bin/bash
# CLIP Patch Similarity Engine — Setup
# Works on any machine with Python 3.10+ and 4GB+ RAM
# Tested on: Pi 5 (aarch64), i7 laptop (x86_64)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== CLIP Patch Similarity Engine Setup ==="
echo ""

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating venv and installing dependencies..."
source venv/bin/activate

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install open-clip-torch pillow numpy

echo ""
echo "=== Setup complete ==="
echo ""
echo "To activate the environment:"
echo "  source $SCRIPT_DIR/venv/bin/activate"
echo ""
echo "To test CLIP is working:"
echo "  python3 $SCRIPT_DIR/test_clip.py"
echo ""
echo "To embed patches from an image:"
echo "  python3 $SCRIPT_DIR/embed_patches.py <image_path>"
echo ""
echo "To find similar patches:"
echo "  python3 $SCRIPT_DIR/find_similar.py <image_path>"
