#!/usr/bin/env bash

set -e

cd "$(dirname "$0")/.."

python scripts/run_surveillance_baselines.py \
    --config configs/experiments/surveillance_baselines_2024_2025.json \
    --resume

python scripts/run_temporal_surveillance.py \
    --config configs/experiments/temporal_surveillance_2024_2025.json \
    --resume

python scripts/analyze_temporal_surveillance.py \
    --config configs/experiments/temporal_surveillance_analysis.json

python scripts/explain_temporal_surveillance.py \
    --config configs/experiments/surveillance_explainability.json
