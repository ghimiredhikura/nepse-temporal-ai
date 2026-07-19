"""Render the headline paper results bundled with this release.

The program reads locked aggregate result artefacts, creates one performance
table and one uncertainty figure, and checks the included illustrative sample.
It does not refit a model or claim that the sample reproduces the paper.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parent
PAPER_RESULTS = ROOT / "results" / "paper"
OUTPUT = ROOT / "results" / "summary"
METRICS = (
    PAPER_RESULTS
    / "temporal_surveillance_analysis"
    / "surveillance_metrics.csv"
)
BOOTSTRAP = (
    PAPER_RESULTS
    / "surveillance_journal_robustness"
    / "year_specific_bootstrap_summary.csv"
)
SAMPLE = (
    ROOT
    / "data"
    / "processed"
    / "transaction_hypergraph"
    / "benchmark"
    / "stock_stress_labels.parquet"
)
SAMPLE_MANIFEST = SAMPLE.with_name("sample_manifest.json")

MODEL_ORDER = [
    "range_persistence",
    "logit_state",
    "lgbm_price_liquidity",
    "lgbm_state",
    "temporal_tabular",
]
MODEL_LABELS = {
    "range_persistence": "Range persistence",
    "logit_state": "Logistic state",
    "lgbm_price_liquidity": "LightGBM price-liquidity",
    "lgbm_state": "LightGBM full state",
    "temporal_tabular": "Temporal GRU",
}


def build_main_table() -> pd.DataFrame:
    """Create the outer-year and pooled performance table."""
    metrics = pd.read_csv(METRICS)
    metrics["evaluation_year"] = metrics["evaluation_year"].astype(str)
    metrics = metrics.loc[metrics["model"].isin(MODEL_ORDER)].copy()

    records: list[dict[str, str | float]] = []
    for model in MODEL_ORDER:
        record: dict[str, str | float] = {"Model": MODEL_LABELS[model]}
        for period, label in [("2024", "2024"), ("2025", "2025"), ("pooled", "Pooled")]:
            row = metrics.loc[
                metrics["model"].eq(model)
                & metrics["evaluation_year"].eq(period)
            ]
            if len(row) != 1:
                raise ValueError(
                    f"Expected one {period} result for {model}; found {len(row)}."
                )
            record[f"{label} PR-AUC"] = float(row.iloc[0]["pr_auc"])
            record[f"{label} Brier"] = float(row.iloc[0]["brier"])
        records.append(record)

    table = pd.DataFrame.from_records(records)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        OUTPUT / "main_result_table.csv",
        index=False,
        float_format="%.4f",
    )

    headers = list(table.columns)
    markdown = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in table.itertuples(index=False, name=None):
        values = [str(row[0])] + [f"{float(value):.4f}" for value in row[1:]]
        markdown.append("| " + " | ".join(values) + " |")
    (OUTPUT / "main_result_table.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )
    return table


def build_main_figure() -> None:
    """Plot paired block-bootstrap improvements over full-state LightGBM."""
    summary = pd.read_csv(BOOTSTRAP)
    specifications = [
        ("pr_auc_difference", "PR-AUC gain", False),
        ("brier_difference", "Brier-score reduction", True),
    ]
    colors = {2024: "#0072B2", 2025: "#D55E00"}

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Liberation Serif",
            ],
            "font.size": 10,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    for axis, (metric, title, reverse_sign) in zip(axes, specifications):
        selected = summary.loc[summary["metric"].eq(metric)].copy()
        selected = selected.sort_values("evaluation_year", ascending=False)
        y_positions = range(len(selected))
        for y_position, row in zip(y_positions, selected.itertuples(index=False)):
            point = float(row.point_estimate)
            lower = float(row.ci_lower)
            upper = float(row.ci_upper)
            if reverse_sign:
                point, lower, upper = -point, -upper, -lower
            axis.errorbar(
                point,
                y_position,
                xerr=[[point - lower], [upper - point]],
                fmt="o",
                color=colors[int(row.evaluation_year)],
                capsize=3,
                markersize=6,
                linewidth=1.4,
            )
        axis.axvline(0, color="#777777", linestyle="--", linewidth=0.9)
        axis.set_yticks(list(y_positions), selected["evaluation_year"].astype(str))
        axis.set_title(title)
        axis.set_xlabel("Improvement over full-state LightGBM")
        axis.grid(axis="x", alpha=0.18, linewidth=0.6)

    figure.tight_layout(w_pad=2.2)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT / "main_result_figure.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT / "main_result_figure.pdf",
        bbox_inches="tight",
        metadata={"Creator": "NEPSE release main.py"},
    )
    plt.close(figure)


def validate_sample() -> dict[str, object]:
    """Check the dimensions and checksum of the processed sample."""
    manifest = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    sample = pd.read_parquet(SAMPLE)
    checksum = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    checks = {
        "rows": len(sample) == int(manifest["rows"]),
        "columns": len(sample.columns) == int(manifest["columns"]),
        "sha256": checksum == manifest["sha256"],
    }
    if not all(checks.values()):
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(f"Processed sample validation failed: {failed}")
    return manifest


def main() -> None:
    table = build_main_table()
    build_main_figure()
    sample = validate_sample()

    print("\nHeadline outer-evaluation results")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        "\nProcessed illustrative sample: "
        f"{int(sample['rows']):,} rows, {int(sample['securities'])} securities, "
        f"{int(sample['columns'])} columns, "
        f"{sample['date_min']} to {sample['date_max']}."
    )
    print(
        "This sample supports code and schema checks; it does not reproduce "
        "the full paper estimates."
    )
    print(f"\nGenerated outputs: {OUTPUT}")


if __name__ == "__main__":
    main()
