"""Check the public release structure, sample panel, and JSON configurations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_temporal_surveillance import CONTEXT, FEATURES  # noqa: E402
from nepse_ai.utils import sha256_file  # noqa: E402


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
    digest = sha256_file(sample_path)
    if digest != manifest["sha256"]:
        raise ValueError("Processed sample checksum does not match manifest.")
    panel["date"] = pd.to_datetime(panel["date"])
    expected_dimensions = {
        "rows": len(panel),
        "columns": len(panel.columns),
        "securities": panel["security_id"].nunique(),
        "events": int(panel["next_range_stress"].sum()),
        "date_min": panel["date"].min().date().isoformat(),
        "date_max": panel["date"].max().date().isoformat(),
    }
    mismatches = sorted(
        key
        for key, observed in expected_dimensions.items()
        if observed != manifest[key]
    )
    if mismatches:
        raise ValueError(
            f"Sample manifest values do not match the panel: {mismatches}"
        )
    missing = sorted(set(FEATURES + CONTEXT) - set(panel.columns))
    if missing:
        raise ValueError(f"Sample is missing required columns: {missing}")
    years = sorted(panel["date"].dt.year.unique().tolist())
    if years != [2021, 2022, 2023, 2024, 2025]:
        raise ValueError(f"Unexpected sample years: {years}")
    if panel["next_range_stress"].nunique() != 2:
        raise ValueError("Sample target must contain both classes.")

    configs = sorted((ROOT / "configs" / "experiments").glob("*.json"))
    if not configs:
        raise ValueError("No experiment configurations were found.")
    for path in configs:
        configuration = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(configuration, dict):
            raise ValueError(f"Configuration must be a JSON object: {path}")

    print(
        "Release validation passed: "
        f"{len(panel):,} rows, {panel['security_id'].nunique()} securities, "
        f"{len(panel.columns)} columns, {len(configs)} configurations."
    )


if __name__ == "__main__":
    main()
