#!/bin/bash

# Predictive Maintenance RUL Prediction - Complete Pipeline

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$SCRIPT_DIR/.venv/bin/python"

echo "================================"
echo "RUL Prediction Pipeline"
echo "================================"

cd "$SCRIPT_DIR"

echo ""
echo "1. Downloading official NASA C-MAPSS data..."
"$PYTHON_BIN" src/download_data.py

echo ""
echo "2. Training model on FD001..."
"$PYTHON_BIN" src/train.py --train-datasets FD001 --test-datasets FD001 --mode in-distribution

echo ""
echo "3. Evaluating saved checkpoint..."
"$PYTHON_BIN" src/evaluate.py --dataset FD001

echo ""
echo "================================"
echo "Pipeline completed successfully!"
echo "================================"
