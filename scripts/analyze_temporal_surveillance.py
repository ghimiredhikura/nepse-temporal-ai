"""Calibration, uncertainty, regimes, and inference for 2024-2025."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from nepse_ai.evaluation import (
    conformal_quantile,
    conformal_set_metrics,
    extended_classification_metrics,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


IDENTITY = [
    "fold",
    "evaluation_year",
    "date",
    "symbol",
    "security_id",
    "sector",
    "listing_status",
    "next_range_stress",
    "range_t",
    "log_turnover_t",
    "gross_flow_hhi",
    "broker_count",
    "mkt_return",
    "mkt_abs_return",
    "mkt_log_turnover",
    "model",
]


def load_prediction_files(
    root: Path,
    filename: str,
) -> pd.DataFrame:
    frames = []
    for path in sorted(root.glob(f"eval_*/{filename}")):
        frame = pd.read_parquet(path)
        frame["fold"] = path.parent.name
        frame["evaluation_year"] = int(path.parent.name.rsplit("_", 1)[1])
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No {filename} files under {root}")
    return pd.concat(frames, ignore_index=True)


def ensemble_predictions(
    predictions: pd.DataFrame,
    calibration: bool,
) -> pd.DataFrame:
    frame = predictions.copy()
    probability = np.clip(
        frame["probability_calibrated"].to_numpy(), 1e-12, 1 - 1e-12
    )
    frame["_seed_entropy"] = -(
        probability * np.log(probability)
        + (1 - probability) * np.log(1 - probability)
    )
    keys = IDENTITY + (["calibration_part"] if calibration else [])
    ensemble = (
        frame.groupby(keys, as_index=False, observed=True)
        .agg(
            probability=("probability_calibrated", "mean"),
            uncertainty_std=("probability_calibrated", lambda x: x.std(ddof=0)),
            expected_seed_entropy=("_seed_entropy", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    mean = np.clip(ensemble["probability"].to_numpy(), 1e-12, 1 - 1e-12)
    ensemble["predictive_entropy"] = -(
        mean * np.log(mean) + (1 - mean) * np.log(1 - mean)
    )
    ensemble["mutual_information"] = (
        ensemble["predictive_entropy"] - ensemble["expected_seed_entropy"]
    ).clip(lower=0)
    return ensemble


def metrics_table(
    evaluation: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    rows = []
    for year, sample in evaluation.groupby("evaluation_year", sort=True):
        for model in models:
            group = sample.loc[sample["model"].eq(model)]
            rows.append(
                {
                    "scope": "year",
                    "evaluation_year": year,
                    "model": model,
                    **extended_classification_metrics(
                        group["next_range_stress"].to_numpy(),
                        group["probability"].to_numpy(),
                    ),
                    "mean_uncertainty_std": float(
                        group["uncertainty_std"].mean()
                    ),
                    "mean_mutual_information": float(
                        group["mutual_information"].mean()
                    ),
                }
            )
    for model in models:
        group = evaluation.loc[evaluation["model"].eq(model)]
        rows.append(
            {
                "scope": "pooled",
                "evaluation_year": "pooled",
                "model": model,
                **extended_classification_metrics(
                    group["next_range_stress"].to_numpy(),
                    group["probability"].to_numpy(),
                ),
                "mean_uncertainty_std": float(
                    group["uncertainty_std"].mean()
                ),
                "mean_mutual_information": float(
                    group["mutual_information"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def conformal_table(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    models: list[str],
    alphas: list[float],
) -> pd.DataFrame:
    rows = []
    for fold in sorted(evaluation["fold"].unique()):
        for model in models:
            calibration_group = calibration.loc[
                calibration["fold"].eq(fold)
                & calibration["model"].eq(model)
                & calibration["calibration_part"].eq("conformal")
            ]
            evaluation_group = evaluation.loc[
                evaluation["fold"].eq(fold)
                & evaluation["model"].eq(model)
            ]
            for alpha in alphas:
                quantile = conformal_quantile(
                    calibration_group["next_range_stress"].to_numpy(),
                    calibration_group["probability"].to_numpy(),
                    alpha,
                )
                rows.append(
                    {
                        "fold": fold,
                        "evaluation_year": int(
                            evaluation_group["evaluation_year"].iloc[0]
                        ),
                        "model": model,
                        "alpha": alpha,
                        "nominal_coverage": 1 - alpha,
                        **conformal_set_metrics(
                            evaluation_group[
                                "next_range_stress"
                            ].to_numpy(),
                            evaluation_group["probability"].to_numpy(),
                            quantile,
                        ),
                    }
                )
    return pd.DataFrame(rows)


def selective_table(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    models: list[str],
    coverages: list[float],
) -> pd.DataFrame:
    rows = []
    for fold in sorted(evaluation["fold"].unique()):
        for model in models:
            calibration_group = calibration.loc[
                calibration["fold"].eq(fold)
                & calibration["model"].eq(model)
                & calibration["calibration_part"].eq("conformal")
            ]
            evaluation_group = evaluation.loc[
                evaluation["fold"].eq(fold)
                & evaluation["model"].eq(model)
            ]
            for requested in coverages:
                threshold = float(
                    np.quantile(
                        calibration_group["uncertainty_std"],
                        requested,
                    )
                )
                selected = evaluation_group.loc[
                    evaluation_group["uncertainty_std"].le(threshold)
                ]
                if len(selected) < 100 or selected[
                    "next_range_stress"
                ].nunique() < 2:
                    continue
                rows.append(
                    {
                        "fold": fold,
                        "evaluation_year": int(
                            evaluation_group["evaluation_year"].iloc[0]
                        ),
                        "model": model,
                        "requested_coverage": requested,
                        "uncertainty_threshold": threshold,
                        "realized_coverage": len(selected)
                        / len(evaluation_group),
                        "rows": len(selected),
                        **extended_classification_metrics(
                            selected["next_range_stress"].to_numpy(),
                            selected["probability"].to_numpy(),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def reliability_table(
    evaluation: pd.DataFrame,
    bins: int = 10,
) -> pd.DataFrame:
    rows = []
    for keys, sample in evaluation.groupby(
        ["evaluation_year", "model"], sort=True, observed=True
    ):
        ranked = sample.copy()
        ranked["bin"] = pd.qcut(
            ranked["probability"],
            q=bins,
            labels=False,
            duplicates="drop",
        )
        grouped = (
            ranked.groupby("bin", observed=True)
            .agg(
                rows=("next_range_stress", "size"),
                predicted=("probability", "mean"),
                observed=("next_range_stress", "mean"),
            )
            .reset_index()
        )
        grouped["evaluation_year"] = keys[0]
        grouped["model"] = keys[1]
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True)


def add_causal_market_state(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat(
        [
            calibration.assign(_source="calibration"),
            evaluation.assign(_source="evaluation"),
        ],
        ignore_index=True,
    )
    outputs = []
    for fold, sample in combined.groupby("fold", sort=True):
        market = (
            sample[
                ["date", "mkt_return", "mkt_abs_return", "mkt_log_turnover"]
            ]
            .drop_duplicates("date")
            .sort_values("date")
        )
        market["market_trend20"] = (
            market["mkt_return"].rolling(20, min_periods=5).sum()
        )
        market["market_volatility20"] = (
            market["mkt_abs_return"].rolling(20, min_periods=5).mean()
        )
        sample = sample.merge(
            market[["date", "market_trend20", "market_volatility20"]],
            on="date",
            how="left",
            validate="many_to_one",
        )
        outputs.append(sample)
    combined = pd.concat(outputs, ignore_index=True)
    return (
        combined.loc[combined["_source"].eq("calibration")]
        .drop(columns="_source")
        .reset_index(drop=True),
        combined.loc[combined["_source"].eq("evaluation")]
        .drop(columns="_source")
        .reset_index(drop=True),
    )


def regime_table(
    calibration: pd.DataFrame,
    evaluation: pd.DataFrame,
    models: list[str],
) -> pd.DataFrame:
    rows = []
    calibration, evaluation = add_causal_market_state(
        calibration, evaluation
    )
    for fold in sorted(evaluation["fold"].unique()):
        reference = calibration.loc[
            calibration["fold"].eq(fold)
            & calibration["model"].eq("temporal_tabular")
            & calibration["calibration_part"].eq("conformal")
        ]
        thresholds = {
            "volatility": float(reference["market_volatility20"].median()),
            "liquidity_q25": float(reference["log_turnover_t"].quantile(0.25)),
            "liquidity_q75": float(reference["log_turnover_t"].quantile(0.75)),
            "concentration": float(reference["gross_flow_hhi"].median()),
        }
        sample = evaluation.loc[evaluation["fold"].eq(fold)].copy()
        sample["market_trend"] = np.where(
            sample["market_trend20"] >= 0, "up", "down"
        )
        sample["market_volatility"] = np.where(
            sample["market_volatility20"] >= thresholds["volatility"],
            "high",
            "low",
        )
        sample["stock_liquidity"] = np.select(
            [
                sample["log_turnover_t"] <= thresholds["liquidity_q25"],
                sample["log_turnover_t"] >= thresholds["liquidity_q75"],
            ],
            ["low", "high"],
            default="middle",
        )
        sample["broker_concentration"] = np.where(
            sample["gross_flow_hhi"] >= thresholds["concentration"],
            "high",
            "low",
        )
        dimensions = [
            "market_trend",
            "market_volatility",
            "stock_liquidity",
            "broker_concentration",
            "sector",
        ]
        for dimension in dimensions:
            for value, group in sample.groupby(
                dimension, observed=True, sort=True
            ):
                for model in models:
                    model_group = group.loc[group["model"].eq(model)]
                    target = model_group["next_range_stress"]
                    if len(model_group) < 500 or target.sum() < 20:
                        continue
                    rows.append(
                        {
                            "fold": fold,
                            "evaluation_year": int(
                                model_group["evaluation_year"].iloc[0]
                            ),
                            "dimension": dimension,
                            "regime": value,
                            "model": model,
                            "rows": len(model_group),
                            "positives": int(target.sum()),
                            **extended_classification_metrics(
                                target.to_numpy(),
                                model_group["probability"].to_numpy(),
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def bootstrap_index(
    panel: pd.DataFrame,
    block_length: int,
    generator: np.random.Generator,
) -> np.ndarray:
    indices = []
    for _, year in panel.groupby("evaluation_year", sort=True):
        groups = [
            np.asarray(index, dtype="int64")
            for index in year.groupby("date", sort=True).groups.values()
        ]
        count = len(groups)
        positions = []
        while len(positions) < count:
            start = int(generator.integers(0, count))
            positions.extend(
                (start + offset) % count for offset in range(block_length)
            )
        indices.extend(groups[position] for position in positions[:count])
    return np.concatenate(indices)


def comparison_panel(evaluation: pd.DataFrame) -> pd.DataFrame:
    key = [
        "fold",
        "evaluation_year",
        "date",
        "symbol",
        "security_id",
        "next_range_stress",
    ]
    return (
        evaluation.loc[
            evaluation["model"].isin(["temporal_tabular", "lgbm_state"])
        ]
        .pivot(index=key, columns="model", values="probability")
        .reset_index()
        .sort_values(["evaluation_year", "date", "security_id"])
        .reset_index(drop=True)
    )


def bootstrap_comparison(
    evaluation: pd.DataFrame,
    draws: int,
    block_length: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    panel = comparison_panel(evaluation)
    target = panel["next_range_stress"].to_numpy()
    temporal = panel["temporal_tabular"].to_numpy()
    baseline = panel["lgbm_state"].to_numpy()
    generator = np.random.default_rng(seed)
    records = []
    started = time.time()
    for draw in range(draws):
        index = bootstrap_index(panel, block_length, generator)
        y = target[index]
        temporal_probability = temporal[index]
        baseline_probability = baseline[index]
        temporal_top = y[
            temporal_probability >= np.quantile(temporal_probability, 0.90)
        ].mean()
        baseline_top = y[
            baseline_probability >= np.quantile(baseline_probability, 0.90)
        ].mean()
        records.append(
            {
                "draw": draw,
                "pr_auc_difference": (
                    average_precision_score(y, temporal_probability)
                    - average_precision_score(y, baseline_probability)
                ),
                "brier_difference": (
                    brier_score_loss(y, temporal_probability)
                    - brier_score_loss(y, baseline_probability)
                ),
                "top_decile_precision_difference": (
                    temporal_top - baseline_top
                ),
            }
        )
        if (draw + 1) % 25 == 0 or draw + 1 == draws:
            elapsed = time.time() - started
            remaining = elapsed / (draw + 1) * (draws - draw - 1)
            print(
                f"Bootstrap {draw + 1}/{draws}; "
                f"elapsed={elapsed / 60:.1f}m; "
                f"remaining={remaining / 60:.1f}m",
                flush=True,
            )
    frame = pd.DataFrame(records)
    summary = {}
    for column in frame.columns.drop("draw"):
        values = frame[column].to_numpy()
        lower_is_better = column == "brier_difference"
        improvement = values < 0 if lower_is_better else values > 0
        summary[column] = {
            "mean": float(values.mean()),
            "p025": float(np.quantile(values, 0.025)),
            "p975": float(np.quantile(values, 0.975)),
            "probability_improvement": float(improvement.mean()),
        }
    return frame, summary


def make_figures(
    metrics: pd.DataFrame,
    reliability: pd.DataFrame,
    selective: pd.DataFrame,
    output: Path,
) -> None:
    annual = metrics.loc[metrics["scope"].eq("year")]
    models = ["temporal_tabular", "lgbm_state", "logit_state"]
    colors = {
        "temporal_tabular": "#2E8B57",
        "lgbm_state": "#6C8EBF",
        "logit_state": "#D9895B",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for model in models:
        sample = annual.loc[annual["model"].eq(model)]
        axes[0].plot(
            sample["evaluation_year"].astype(int),
            sample["pr_auc"],
            marker="o",
            color=colors[model],
            label=model,
        )
        axes[1].plot(
            sample["evaluation_year"].astype(int),
            sample["brier"],
            marker="o",
            color=colors[model],
            label=model,
        )
    axes[0].set(title="Outer-year discrimination", ylabel="PR-AUC")
    axes[1].set(title="Outer-year calibration", ylabel="Brier score")
    for axis in axes:
        axis.set(
            xlabel="Evaluation year",
            xticks=sorted(annual["evaluation_year"].astype(int).unique()),
        )
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "surveillance_performance.png", dpi=300)
    figure.savefig(output / "surveillance_performance.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for axis, year in zip(
        axes, sorted(reliability["evaluation_year"].unique())
    ):
        for model in ("temporal_tabular", "lgbm_state"):
            sample = reliability.loc[
                reliability["evaluation_year"].eq(year)
                & reliability["model"].eq(model)
            ]
            axis.plot(
                sample["predicted"],
                sample["observed"],
                marker="o",
                label=model,
                color=colors[model],
            )
        axis.plot([0, 0.5], [0, 0.5], "--", color="black")
        axis.set(
            title=f"{year} reliability",
            xlabel="Mean predicted risk",
            ylabel="Observed rate",
        )
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "surveillance_reliability.png", dpi=300)
    figure.savefig(output / "surveillance_reliability.pdf")
    plt.close(figure)

    sample = selective.loc[selective["model"].eq("temporal_tabular")]
    if not sample.empty:
        figure, axis = plt.subplots(figsize=(7, 5))
        for year, group in sample.groupby("evaluation_year", sort=True):
            axis.plot(
                group["realized_coverage"],
                group["brier"],
                marker="o",
                label=str(year),
            )
        axis.set(
            title="Uncertainty-based selective prediction",
            xlabel="Realized coverage",
            ylabel="Brier score",
        )
        axis.legend(title="Evaluation year", frameon=False)
        figure.tight_layout()
        figure.savefig(output / "surveillance_selective_risk.png", dpi=300)
        figure.savefig(output / "surveillance_selective_risk.pdf")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int)
    arguments = parser.parse_args()
    configuration = json.loads(
        arguments.config.read_text(encoding="utf-8")
    )
    temporal_root = Path(str(configuration["temporal_root"]))
    baseline_root = Path(str(configuration["baseline_root"]))
    temporal_evaluation = load_prediction_files(
        temporal_root, "evaluation_predictions.parquet"
    )
    temporal_calibration = load_prediction_files(
        temporal_root, "calibration_predictions.parquet"
    )
    baseline_evaluation = load_prediction_files(
        baseline_root, "evaluation_predictions.parquet"
    )
    baseline_calibration = load_prediction_files(
        baseline_root, "calibration_predictions.parquet"
    )
    evaluation = ensemble_predictions(
        pd.concat(
            [temporal_evaluation, baseline_evaluation], ignore_index=True
        ),
        calibration=False,
    )
    calibration = ensemble_predictions(
        pd.concat(
            [temporal_calibration, baseline_calibration], ignore_index=True
        ),
        calibration=True,
    )
    models = [str(model) for model in configuration["models"]]
    evaluation = evaluation.loc[evaluation["model"].isin(models)].copy()
    calibration = calibration.loc[calibration["model"].isin(models)].copy()
    output = Path(str(configuration["output_root"]))
    output.mkdir(parents=True, exist_ok=True)

    metrics = metrics_table(evaluation, models)
    conformal = conformal_table(
        calibration,
        evaluation,
        models,
        [float(alpha) for alpha in configuration["conformal_alphas"]],
    )
    selective = selective_table(
        calibration,
        evaluation,
        models,
        [
            float(coverage)
            for coverage in configuration["selective_coverages"]
        ],
    )
    reliability = reliability_table(evaluation)
    regimes = regime_table(calibration, evaluation, models)
    draws = (
        arguments.bootstrap_draws
        if arguments.bootstrap_draws is not None
        else int(configuration["bootstrap_draws"])
    )
    bootstrap, bootstrap_summary = bootstrap_comparison(
        evaluation,
        draws,
        int(configuration["block_length"]),
        int(configuration["random_seed"]),
    )

    evaluation.to_parquet(
        output / "ensemble_evaluation_predictions.parquet",
        index=False,
        compression="zstd",
    )
    calibration.to_parquet(
        output / "ensemble_calibration_predictions.parquet",
        index=False,
        compression="zstd",
    )
    metrics.to_csv(output / "surveillance_metrics.csv", index=False)
    conformal.to_csv(output / "conformal_metrics.csv", index=False)
    selective.to_csv(output / "selective_prediction_metrics.csv", index=False)
    reliability.to_csv(output / "reliability_bins.csv", index=False)
    regimes.to_csv(output / "regime_metrics.csv", index=False)
    bootstrap.to_parquet(
        output / "temporal_vs_lgbm_block_bootstrap.parquet",
        index=False,
        compression="zstd",
    )
    (output / "temporal_vs_lgbm_bootstrap_summary.json").write_text(
        json.dumps(bootstrap_summary, indent=2),
        encoding="utf-8",
    )
    make_figures(metrics, reliability, selective, output)
    metadata = {
        "status": "complete",
        "evaluation_rows": len(evaluation),
        "calibration_rows": len(calibration),
        "models": models,
        "bootstrap_draws": draws,
        "block_length": int(configuration["block_length"]),
        "conformal_alphas": configuration["conformal_alphas"],
        "regime_threshold_source": "preceding conformal-calibration half",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(metrics.to_string(index=False), flush=True)
    print(json.dumps(bootstrap_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
