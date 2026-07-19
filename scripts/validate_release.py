"""Check the public release structure, sample panel, and JSON configurations."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_temporal_surveillance import CONTEXT, FEATURES  # noqa: E402


def main() -> None:
    sample_path = (
        ROOT
        / "data"
        / "processed"
        / "transaction_hypergraph"
        / "benchmark"
        / "stock_stress_labels.parquet"
    )
    panel = pd.read_parquet(sample_path)
    manifest = json.loads(
        (sample_path.parent / "sample_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    if digest != manifest["sha256"]:
        raise ValueError("Processed sample checksum does not match manifest.")
    panel["date"] = pd.to_datetime(panel["date"])
    missing = sorted(set(FEATURES + CONTEXT) - set(panel.columns))
    if missing:
        raise ValueError(f"Sample is missing required columns: {missing}")
    years = sorted(panel["date"].dt.year.unique().tolist())
    if years != [2021, 2022, 2023, 2024, 2025]:
        raise ValueError(f"Unexpected sample years: {years}")
    if panel["next_range_stress"].nunique() != 2:
        raise ValueError("Sample target must contain both classes.")

    configs = sorted((ROOT / "configs" / "experiments").glob("*.json"))
    for path in configs:
        json.loads(path.read_text(encoding="utf-8"))

    print(
        "Release validation passed: "
        f"{len(panel):,} rows, {panel['security_id'].nunique()} securities, "
        f"{len(panel.columns)} columns, {len(configs)} configurations."
    )


if __name__ == "__main__":
    main()
