#!/usr/bin/env bash
#
# Compress a 3DGS PLY file to Spark-compatible .sog format.
#
# Usage:  ./compress.sh path/to/scene.ply
# Output: results/scene_YYYYMMDD_HHMMSS.sog + results/scene_YYYYMMDD_HHMMSS_params.json
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

# Derive unique output name with timestamp
BASENAME="$(basename "$INPUT_PLY" .ply)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${BASENAME}_${TIMESTAMP}"

INTERMEDIATE_DIR="$SCRIPT_DIR/results/standalone_compression_${RUN_NAME}"
COMPRESSED_DIR="$INTERMEDIATE_DIR/exr_jxl_quant_5_norm"
OUTPUT_SOG="$SCRIPT_DIR/results/${RUN_NAME}.sog"
OUTPUT_PARAMS="$SCRIPT_DIR/results/${RUN_NAME}_params.json"

# Activate venv
source "$VENV/bin/activate"

echo "============================================================"
echo "Input:  $INPUT_PLY"
echo "Output: $OUTPUT_SOG"
echo "Params: $OUTPUT_PARAMS"
echo "============================================================"

# Step 1: SOGS compression (prune, sort, compress)
echo ""
echo "Step 1/2: SOGS compression..."
python "$SCRIPT_DIR/standalone_compress.py" "$INPUT_PLY" "$INTERMEDIATE_DIR"

# Step 2: Convert to Spark .sog format
echo ""
echo "Step 2/2: Converting to .sog..."
python "$SCRIPT_DIR/convert_to_spark_sog.py" "$COMPRESSED_DIR" "$OUTPUT_SOG"

# Step 3: Save params file alongside the .sog
echo ""
echo "Saving parameters..."
python -c "
import json, yaml, os
config_path = '$SCRIPT_DIR/config/compression/umbrella_sh.yaml'
with open(config_path) as f:
    compression_config = yaml.safe_load(f)
params = {
    'input_ply': '$INPUT_PLY',
    'timestamp': '$TIMESTAMP',
    'sog_file': '${RUN_NAME}.sog',
    'compression_config': compression_config['experiments'][0],
    'sog_size_bytes': os.path.getsize('$OUTPUT_SOG'),
}
with open('$OUTPUT_PARAMS', 'w') as f:
    json.dump(params, f, indent=2)
"

echo ""
echo "============================================================"
echo "Done!"
echo "  SOG:    $OUTPUT_SOG ($(du -h "$OUTPUT_SOG" | cut -f1))"
echo "  Params: $OUTPUT_PARAMS"
echo "============================================================"
