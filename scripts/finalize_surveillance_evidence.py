"""Robustness, calibration, and decision analysis."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from nepse_ai.evaluation import (
    alert_counts,
    alert_metrics,
    circular_block_indices,
    conformal_quantile,
    cost_utility,
    daily_alert_assignments,
    decision_curve,
    interval_summary,
    paired_forecasts,
    paired_metric_differences,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def report_progress(
    stage: str,
    completed: int,
    total: int,
    started: float,
) -> None:
    if completed % 50 and completed != total:
        return
    elapsed = time.time() - started
    remaining = elapsed / completed * (total - completed)
    print(
        f"{stage}: {completed}/{total}; elapsed={elapsed / 60:.1f}m; "
        f"remaining={remaining / 60:.1f}m",
        flush=True,
    )


def yearly_comparison_bootstrap(
    evaluation: pd.DataFrame,
    left_model: str,
    right_model: str,
    draws: int,
    block_length: int,
    seed: int,
    top_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair = paired_forecasts(evaluation, left_model, right_model)
    generator = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    points: list[dict[str, object]] = []
    years = sorted(pair["evaluation_year"].unique())
    total = draws * len(years)
    completed = 0
    started = time.time()
    for year in years:
        sample = pair.loc[pair["evaluation_year"].eq(year)].reset_index(
            drop=True
        )
        points.append(
            {
                "evaluation_year": int(year),
                **paired_metric_differences(
                    sample, left_model, right_model, top_fraction
                ),
            }
        )
        for draw in range(draws):
            index = circular_block_indices(
                sample, block_length, generator
            )
            records.append(
                {
                    "evaluation_year": int(year),
                    "draw": draw,
                    **paired_metric_differences(
                        sample.iloc[index],
                        left_model,
                        right_model,
                        top_fraction,
                    ),
                }
            )
            completed += 1
            report_progress(
                "Year-specific forecast bootstrap",
                completed,
                total,
                started,
            )
    frame = pd.DataFrame(records)
    metrics = [
        "pr_auc_difference",
        "brier_difference",
        "log_loss_difference",
        "top_fraction_precision_difference",
    ]
    summary = interval_summary(
        frame, ["evaluation_year"], metrics
    )
    point_frame = pd.DataFrame(points).melt(
        id_vars="evaluation_year",
        value_vars=metrics,
        var_name="metric",
        value_name="point_estimate",
    )
    return frame, summary.merge(
        point_frame, on=["evaluation_year", "metric"], validate="one_to_one"
    )


def conformal_indicators(
    target: np.ndarray,
    probability: np.ndarray,
    quantile: float,
) -> dict[str, np.ndarray]:
    include_zero = probability <= quantile
    include_one = 1 - probability <= quantile
    covered = np.where(target == 1, include_one, include_zero)
    size = include_zero.astype("int8") + include_one.astype("int8")
    return {
        "coverage": covered.astype("float64"),
        "average_set_size": size.astype("float64"),
        "singleton_rate": (size == 1).astype("float64"),
        "empty_rate": (size == 0).astype("float64"),
    }


def calibration_conformal_bootstrap(
    evaluation: pd.DataFrame,
    calibration: pd.DataFrame,
    models: list[str],
    alphas: list[float],
    draws: int,
    block_length: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generator = np.random.default_rng(seed)
    calibration_draws: list[dict[str, object]] = []
    conformal_draws: list[dict[str, object]] = []
    calibration_points: list[dict[str, object]] = []
    conformal_points: list[dict[str, object]] = []
    years = sorted(evaluation["evaluation_year"].unique())
    total = draws * len(years)
    completed = 0
    started = time.time()
    for year in years:
        model_frames: dict[str, pd.DataFrame] = {}
        indicators: dict[tuple[str, float], dict[str, np.ndarray]] = {}
        quantiles: dict[tuple[str, float], float] = {}
        for model in models:
            sample = (
                evaluation.loc[
                    evaluation["evaluation_year"].eq(year)
                    & evaluation["model"].eq(model)
                ]
                .sort_values(["date", "security_id"])
                .reset_index(drop=True)
            )
            model_frames[model] = sample
            target = sample["next_range_stress"].to_numpy()
            probability = np.clip(
                sample["probability"].to_numpy(), 1e-12, 1 - 1e-12
            )
            calibration_points.append(
                {
                    "evaluation_year": int(year),
                    "model": model,
                    "brier": brier_score_loss(target, probability),
                    "log_loss": log_loss(target, probability),
                    "calibration_gap": float(
                        probability.mean() - target.mean()
                    ),
                }
            )
            reference = calibration.loc[
                calibration["evaluation_year"].eq(year)
                & calibration["model"].eq(model)
                & calibration["calibration_part"].eq("conformal")
            ]
            for alpha in alphas:
                quantile = conformal_quantile(
                    reference["next_range_stress"].to_numpy(),
                    reference["probability"].to_numpy(),
                    alpha,
                )
                key = (model, float(alpha))
                quantiles[key] = quantile
                indicators[key] = conformal_indicators(
                    target, probability, quantile
                )
                conformal_points.append(
                    {
                        "evaluation_year": int(year),
                        "model": model,
                        "alpha": float(alpha),
                        "nominal_coverage": 1 - float(alpha),
                        "conformal_quantile": quantile,
                        **{
                            metric: float(values.mean())
                            for metric, values in indicators[key].items()
                        },
                    }
                )
        template = model_frames[models[0]][
            ["date", "security_id"]
        ].copy()
        for model in models[1:]:
            other = model_frames[model][["date", "security_id"]]
            if not template.equals(other):
                raise ValueError(
                    f"Evaluation support differs for {model} in {year}"
                )
        for draw in range(draws):
            index = circular_block_indices(
                template, block_length, generator
            )
            for model, sample in model_frames.items():
                target = sample["next_range_stress"].to_numpy()[index]
                probability = np.clip(
                    sample["probability"].to_numpy()[index],
                    1e-12,
                    1 - 1e-12,
                )
                calibration_draws.append(
                    {
                        "evaluation_year": int(year),
                        "model": model,
                        "draw": draw,
                        "brier": brier_score_loss(target, probability),
                        "log_loss": log_loss(target, probability),
                        "calibration_gap": float(
                            probability.mean() - target.mean()
                        ),
                    }
                )
                for alpha in alphas:
                    values = indicators[(model, float(alpha))]
                    conformal_draws.append(
                        {
                            "evaluation_year": int(year),
                            "model": model,
                            "alpha": float(alpha),
                            "draw": draw,
                            **{
                                metric: float(array[index].mean())
                                for metric, array in values.items()
                            },
                        }
                    )
            completed += 1
            report_progress(
                "Calibration/conformal bootstrap",
                completed,
                total,
                started,
            )
    calibration_draw_frame = pd.DataFrame(calibration_draws)
    conformal_draw_frame = pd.DataFrame(conformal_draws)
    calibration_summary = interval_summary(
        calibration_draw_frame,
        ["evaluation_year", "model"],
        ["brier", "log_loss", "calibration_gap"],
    )
    calibration_point_frame = pd.DataFrame(calibration_points).melt(
        id_vars=["evaluation_year", "model"],
        var_name="metric",
        value_name="point_estimate",
    )
    calibration_summary = calibration_summary.merge(
        calibration_point_frame,
        on=["evaluation_year", "model", "metric"],
        validate="one_to_one",
    )
    conformal_summary = interval_summary(
        conformal_draw_frame,
        ["evaluation_year", "model", "alpha"],
        ["coverage", "average_set_size", "singleton_rate", "empty_rate"],
    )
    conformal_point_frame = pd.DataFrame(conformal_points).melt(
        id_vars=[
            "evaluation_year",
            "model",
            "alpha",
            "nominal_coverage",
            "conformal_quantile",
        ],
        value_vars=[
            "coverage",
            "average_set_size",
            "singleton_rate",
            "empty_rate",
        ],
        var_name="metric",
        value_name="point_estimate",
    )
    conformal_summary = conformal_summary.merge(
        conformal_point_frame,
        on=["evaluation_year", "model", "alpha", "metric"],
        validate="one_to_one",
    )
    return (
        calibration_draw_frame,
        calibration_summary,
        conformal_draw_frame,
        conformal_summary,
    )


def bootstrap_alert_differences(
    counts: pd.DataFrame,
    left_model: str,
    right_model: str,
    missed_event_costs: list[float],
    draws: int,
    block_length: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    generator = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    cost_records: list[dict[str, object]] = []
    combinations = (
        counts[["evaluation_year", "alert_budget"]]
        .drop_duplicates()
        .sort_values(["evaluation_year", "alert_budget"])
    )
    total = draws * len(combinations)
    completed = 0
    started = time.time()
    for combination in combinations.itertuples(index=False):
        year = int(combination.evaluation_year)
        budget = float(combination.alert_budget)
        sample = counts.loc[
            counts["evaluation_year"].eq(year)
            & counts["alert_budget"].eq(budget)
        ]
        metrics = [
            "rows",
            "positives",
            "alerts",
            "true_positives",
            "false_positives",
            "false_negatives",
        ]
        pair = (
            sample.pivot(index="date", columns="model", values=metrics)
            .sort_index()
        )
        required = [
            (metric, model)
            for metric in metrics
            for model in (left_model, right_model)
        ]
        if pair[required].isna().any().any():
            raise ValueError("Alert counts do not align between models")
        template = pd.DataFrame({"date": pair.index}).reset_index(drop=True)
        pair = pair.reset_index(drop=True)
        for draw in range(draws):
            index = circular_block_indices(
                template, block_length, generator
            )
            totals = {
                model: {
                    metric: float(pair[(metric, model)].to_numpy()[index].sum())
                    for metric in metrics
                }
                for model in (left_model, right_model)
            }
            derived: dict[str, dict[str, float]] = {}
            for model, values in totals.items():
                derived[model] = {
                    "precision": (
                        values["true_positives"] / values["alerts"]
                    ),
                    "recall": (
                        values["true_positives"] / values["positives"]
                    ),
                    "lift": (
                        values["true_positives"] / values["alerts"]
                    )
                    / (values["positives"] / values["rows"]),
                }
            records.append(
                {
                    "evaluation_year": year,
                    "alert_budget": budget,
                    "draw": draw,
                    **{
                        f"{metric}_difference": (
                            derived[left_model][metric]
                            - derived[right_model][metric]
                        )
                        for metric in ("precision", "recall", "lift")
                    },
                }
            )
            for missed_cost in missed_event_costs:
                reductions: dict[str, float] = {}
                for model, values in totals.items():
                    cost = (
                        values["false_positives"]
                        + missed_cost * values["false_negatives"]
                    )
                    no_alert = missed_cost * values["positives"]
                    reductions[model] = (no_alert - cost) / no_alert
                cost_records.append(
                    {
                        "evaluation_year": year,
                        "alert_budget": budget,
                        "missed_event_cost": missed_cost,
                        "draw": draw,
                        "cost_reduction_difference": (
                            reductions[left_model]
                            - reductions[right_model]
                        ),
                    }
                )
            completed += 1
            report_progress(
                "Daily-alert bootstrap",
                completed,
                total,
                started,
            )
    frame = pd.DataFrame(records)
    cost_frame = pd.DataFrame(cost_records)
    summary = interval_summary(
        frame,
        ["evaluation_year", "alert_budget"],
        ["precision_difference", "recall_difference", "lift_difference"],
    )
    return frame, cost_frame, summary


def make_figures(
    comparison: pd.DataFrame,
    alert: pd.DataFrame,
    decision: pd.DataFrame,
    conformal: pd.DataFrame,
    output: Path,
    left_model: str,
    right_model: str,
) -> None:
    colors = {left_model: "#2E8B57", right_model: "#6C8EBF"}
    annual_pr = comparison.loc[
        comparison["metric"].eq("pr_auc_difference")
    ].sort_values("evaluation_year")
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.errorbar(
        annual_pr["point_estimate"],
        annual_pr["evaluation_year"].astype(str),
        xerr=np.vstack(
            [
                annual_pr["point_estimate"] - annual_pr["ci_lower"],
                annual_pr["ci_upper"] - annual_pr["point_estimate"],
            ]
        ),
        fmt="o",
        color="#2E8B57",
        capsize=4,
    )
    axis.axvline(0, color="black", linewidth=1, linestyle="--")
    axis.set(
        title="Year-specific temporal forecast advantage",
        xlabel=f"PR-AUC difference ({left_model} minus {right_model})",
        ylabel="Outer evaluation year",
    )
    figure.tight_layout()
    figure.savefig(output / "year_specific_inference.png", dpi=300)
    figure.savefig(output / "year_specific_inference.pdf")
    plt.close(figure)

    years = sorted(alert["evaluation_year"].unique())
    figure, axes = plt.subplots(len(years), 2, figsize=(12.5, 4.5 * len(years)))
    axes = np.atleast_2d(axes)
    for row, year in enumerate(years):
        for model in (left_model, right_model):
            sample = alert.loc[
                alert["evaluation_year"].eq(year)
                & alert["model"].eq(model)
            ].sort_values("alert_budget")
            axes[row, 0].plot(
                sample["alert_budget"] * 100,
                sample["precision"],
                marker="o",
                color=colors[model],
                label=model,
            )
            curve = decision.loc[
                decision["evaluation_year"].eq(year)
                & decision["model"].eq(model)
            ].sort_values("threshold")
            axes[row, 1].plot(
                curve["threshold"],
                curve["standardized_net_benefit"],
                color=colors[model],
                label=model,
            )
        axes[row, 0].set(
            title=f"{year}: session-level alert precision",
            xlabel="Daily alert budget (%)",
            ylabel="Precision",
        )
        axes[row, 1].set(
            title=f"{year}: calibrated decision utility",
            xlabel="Risk threshold",
            ylabel="Standardized net benefit",
        )
        for axis in axes[row]:
            axis.legend(frameon=False)
            axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "operational_utility.png", dpi=300)
    figure.savefig(output / "operational_utility.pdf")
    plt.close(figure)

    coverage = conformal.loc[conformal["metric"].eq("coverage")].copy()
    coverage["nominal"] = 1 - coverage["alpha"]
    figure, axes = plt.subplots(1, len(years), figsize=(12, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for axis, year in zip(axes, years):
        sample = coverage.loc[
            coverage["evaluation_year"].eq(year)
        ].sort_values(["alpha", "model"])
        positions = np.arange(len(sample))
        axis.errorbar(
            positions,
            sample["point_estimate"],
            yerr=np.vstack(
                [
                    sample["point_estimate"] - sample["ci_lower"],
                    sample["ci_upper"] - sample["point_estimate"],
                ]
            ),
            fmt="o",
            capsize=3,
            color="#2E8B57",
        )
        axis.scatter(
            positions,
            sample["nominal"],
            marker="_",
            s=300,
            color="black",
            label="Nominal",
        )
        axis.set_xticks(
            positions,
            [
                f"{model}\n{int((1-alpha)*100)}%"
                for model, alpha in zip(sample["model"], sample["alpha"])
            ],
            rotation=20,
        )
        axis.set(
            title=f"{year} conformal coverage",
            ylabel="Coverage",
        )
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output / "conformal_coverage_uncertainty.png", dpi=300)
    figure.savefig(output / "conformal_coverage_uncertainty.pdf")
    plt.close(figure)


def build_robustness_summary(
    comparison: pd.DataFrame,
    conformal: pd.DataFrame,
    alert_summary: pd.DataFrame,
    left_model: str,
    right_model: str,
) -> dict[str, object]:
    pr = comparison.loc[
        comparison["metric"].eq("pr_auc_difference")
    ]
    brier = comparison.loc[
        comparison["metric"].eq("brier_difference")
    ]
    coverage = conformal.loc[
        conformal["metric"].eq("coverage")
        & conformal["model"].eq(left_model)
    ].copy()
    coverage["nominal"] = 1 - coverage["alpha"]
    precision = alert_summary.loc[
        alert_summary["metric"].eq("precision_difference")
    ]
    return {
        "status": "complete",
        "comparison": f"{left_model} minus {right_model}",
        "year_specific_pr_auc_ci_positive_all_years": bool(
            pr["ci_lower"].gt(0).all()
        ),
        "year_specific_brier_ci_below_zero_all_years": bool(
            brier["ci_upper"].lt(0).all()
        ),
        "temporal_conformal_nominal_inside_95ci_all_settings": bool(
            (
                coverage["ci_lower"].le(coverage["nominal"])
                & coverage["ci_upper"].ge(coverage["nominal"])
            ).all()
        ),
        "temporal_conformal_absolute_error_within_0_02_all_settings": bool(
            (coverage["point_estimate"] - coverage["nominal"])
            .abs()
            .le(0.02)
            .all()
        ),
        "daily_alert_precision_ci_positive_all_budgets_years": bool(
            precision["ci_lower"].gt(0).all()
        ),
        "daily_alert_precision_ci_positive_settings": int(
            precision["ci_lower"].gt(0).sum()
        ),
        "daily_alert_precision_total_settings": int(len(precision)),
        "interpretation": (
            "Uncertainty is used for escalation rather than abstention because "
            "high ensemble disagreement is concentrated among true stress events."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int)
    arguments = parser.parse_args()
    configuration = json.loads(
        arguments.config.read_text(encoding="utf-8")
    )
    analysis_root = Path(str(configuration["analysis_root"]))
    output = Path(str(configuration["output_root"]))
    output.mkdir(parents=True, exist_ok=True)
    evaluation = pd.read_parquet(
        analysis_root / "ensemble_evaluation_predictions.parquet"
    )
    calibration = pd.read_parquet(
        analysis_root / "ensemble_calibration_predictions.parquet"
    )
    models = [str(model) for model in configuration["models"]]
    evaluation = evaluation.loc[evaluation["model"].isin(models)].copy()
    calibration = calibration.loc[calibration["model"].isin(models)].copy()
    draws = (
        arguments.bootstrap_draws
        if arguments.bootstrap_draws is not None
        else int(configuration["bootstrap_draws"])
    )
    block_length = int(configuration["block_length"])
    seed = int(configuration["random_seed"])
    left_model = str(configuration["left_model"])
    right_model = str(configuration["right_model"])

    comparison_draws, comparison_summary = yearly_comparison_bootstrap(
        evaluation,
        left_model,
        right_model,
        draws,
        block_length,
        seed,
        float(configuration["top_fraction"]),
    )
    (
        calibration_draws,
        calibration_summary,
        conformal_draws,
        conformal_summary,
    ) = calibration_conformal_bootstrap(
        evaluation,
        calibration,
        models,
        [float(value) for value in configuration["conformal_alphas"]],
        draws,
        block_length,
        seed + 1,
    )
    assignments = daily_alert_assignments(
        evaluation,
        [float(value) for value in configuration["alert_budgets"]],
    )
    counts = alert_counts(assignments)
    alert = alert_metrics(counts)
    missed_costs = [
        float(value) for value in configuration["missed_event_costs"]
    ]
    cost = cost_utility(alert, missed_costs)
    decision = decision_curve(
        evaluation,
        [float(value) for value in configuration["decision_thresholds"]],
    )
    (
        alert_draws,
        alert_cost_draws,
        alert_summary,
    ) = bootstrap_alert_differences(
        counts,
        left_model,
        right_model,
        missed_costs,
        draws,
        block_length,
        seed + 2,
    )
    alert_cost_summary = interval_summary(
        alert_cost_draws,
        ["evaluation_year", "alert_budget", "missed_event_cost"],
        ["cost_reduction_difference"],
    )

    parquet_outputs = {
        "year_specific_bootstrap_draws.parquet": comparison_draws,
        "calibration_bootstrap_draws.parquet": calibration_draws,
        "conformal_bootstrap_draws.parquet": conformal_draws,
        "daily_alert_assignments.parquet": assignments,
        "daily_alert_counts.parquet": counts,
        "daily_alert_bootstrap_draws.parquet": alert_draws,
        "daily_alert_cost_bootstrap_draws.parquet": alert_cost_draws,
    }
    for filename, frame in parquet_outputs.items():
        frame.to_parquet(
            output / filename, index=False, compression="zstd"
        )
    csv_outputs = {
        "year_specific_bootstrap_summary.csv": comparison_summary,
        "calibration_bootstrap_summary.csv": calibration_summary,
        "conformal_bootstrap_summary.csv": conformal_summary,
        "daily_alert_metrics.csv": alert,
        "cost_utility.csv": cost,
        "decision_curve.csv": decision,
        "daily_alert_bootstrap_summary.csv": alert_summary,
        "daily_alert_cost_bootstrap_summary.csv": alert_cost_summary,
    }
    for filename, frame in csv_outputs.items():
        frame.to_csv(output / filename, index=False)

    make_figures(
        comparison_summary,
        alert,
        decision,
        conformal_summary,
        output,
        left_model,
        right_model,
    )
    robustness_summary = build_robustness_summary(
        comparison_summary,
        conformal_summary,
        alert_summary,
        left_model,
        right_model,
    )
    (output / "robustness_summary.json").write_text(
        json.dumps(robustness_summary, indent=2), encoding="utf-8"
    )
    metadata = {
        "status": "complete",
        "bootstrap_draws": draws,
        "block_length_sessions": block_length,
        "evaluation_years": sorted(
            int(year) for year in evaluation["evaluation_year"].unique()
        ),
        "models": models,
        "alert_budgets": configuration["alert_budgets"],
        "missed_event_costs": missed_costs,
        "outer_predictions_reused_without_retraining": True,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(robustness_summary, indent=2), flush=True)
    print(comparison_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
