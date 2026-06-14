#!/usr/bin/env bash
set -euo pipefail

# Argument parsing with default values
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

# Make output directory if needed
out_dir="$(dirname "$OUTPUT_PATH")"
if [ -n "$out_dir" ] && [ "$out_dir" != "." ]; then
  mkdir -p "$out_dir"
fi

# Detect python interpreter (use local venv if present, otherwise fallback to system python)
if [ -f "./venv/bin/python" ]; then
  PYTHON_BIN="./venv/bin/python"
else
  PYTHON_BIN="python"
fi

# Run feature generation
$PYTHON_BIN src/generate_features.py --data-dir "$DATA_DIR" --out features.parquet

# Run predictions
$PYTHON_BIN src/predict.py --features features.parquet --model "$MODEL_PATH" --output "$OUTPUT_PATH"

echo "Done. Predictions written to $OUTPUT_PATH"
