#!/usr/bin/env bash
#
# Compress a 3DGS PLY file to Spark-compatible .sog format.
#
# Usage:  ./compress.sh path/to/scene.ply
# Output: results/scene.sog
#
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input.ply>"
    exit 1
fi

INPUT_PLY="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv_sogs_cpu"

if [ ! -f "$INPUT_PLY" ]; then
    echo "Error: file not found: $INPUT_PLY"
    exit 1
fi

# Derive output name from input filename (e.g. kitchen.ply → results/kitchen.sog)
BASENAME="$(basename "$INPUT_PLY" .ply)"
INTERMEDIATE_DIR="$SCRIPT_DIR/results/standalone_compression"
COMPRESSED_DIR="$INTERMEDIATE_DIR/exr_jxl_quant_5_norm"
OUTPUT_SOG="$SCRIPT_DIR/results/${BASENAME}.sog"

# Activate venv
source "$VENV/bin/activate"

echo "============================================================"
echo "Input:  $INPUT_PLY"
echo "Output: $OUTPUT_SOG"
echo "============================================================"

# Step 1: SOGS compression (prune, sort, compress)
echo ""
echo "Step 1/2: SOGS compression..."
python "$SCRIPT_DIR/standalone_compress.py" "$INPUT_PLY" "$INTERMEDIATE_DIR"

# Step 2: Convert to Spark .sog format
echo ""
echo "Step 2/2: Converting to .sog..."
python "$SCRIPT_DIR/convert_to_spark_sog.py" "$COMPRESSED_DIR" "$OUTPUT_SOG"

echo ""
echo "============================================================"
echo "Done! Output: $OUTPUT_SOG ($(du -h "$OUTPUT_SOG" | cut -f1))"
echo "============================================================"
