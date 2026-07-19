#!/usr/bin/env bash

set -e

cd "$(dirname "$0")/.."

python scripts/finalize_surveillance_evidence.py \
    --config configs/experiments/surveillance_robustness.json

python scripts/validate_explanation_sanity.py \
    --config configs/experiments/surveillance_explanation_sanity.json
