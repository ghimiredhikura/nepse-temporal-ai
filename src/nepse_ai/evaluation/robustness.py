"""Session-aware robustness utilities for rare-event forecasts."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
)


PAIR_KEYS = [
    "fold",
    "evaluation_year",
    "date",
    "symbol",
    "security_id",
    "next_range_stress",
]


def paired_forecasts(
    evaluation: pd.DataFrame,
    left_model: str,
    right_model: str,
) -> pd.DataFrame:
    """Align two models on the same stock-session outcomes."""
    pair = (
        evaluation.loc[evaluation["model"].isin([left_model, right_model])]
        .pivot(index=PAIR_KEYS, columns="model", values="probability")
        .reset_index()
        .sort_values(["evaluation_year", "date", "security_id"])
        .reset_index(drop=True)
    )
    if pair[[left_model, right_model]].isna().any().any():
        raise ValueError("Models do not have identical evaluation support")
    return pair


def circular_block_indices(
    frame: pd.DataFrame,
    block_length: int,
    generator: np.random.Generator,
    stratify: str | None = None,
) -> np.ndarray:
    """Resample complete sessions in circular moving blocks."""
    if stratify is None:
        strata: Iterable[tuple[object, pd.DataFrame]] = [("all", frame)]
    else:
        strata = frame.groupby(stratify, sort=True, observed=True)
    sampled: list[np.ndarray] = []
    for _, stratum in strata:
        groups = [
            np.asarray(index, dtype="int64")
            for index in stratum.groupby("date", sort=True).groups.values()
        ]
        session_count = len(groups)
        positions: list[int] = []
        while len(positions) < session_count:
            start = int(generator.integers(0, session_count))
            positions.extend(
                (start + offset) % session_count
                for offset in range(block_length)
            )
        sampled.extend(groups[position] for position in positions[:session_count])
    return np.concatenate(sampled)


def top_fraction_precision(
    target: np.ndarray,
    probability: np.ndarray,
    fraction: float,
) -> float:
    """Precision among the globally highest forecast fraction."""
    count = max(1, int(np.ceil(len(probability) * fraction)))
    selected = np.argpartition(probability, -count)[-count:]
    return float(target[selected].mean())


def paired_metric_differences(
    frame: pd.DataFrame,
    left_model: str,
    right_model: str,
    top_fraction: float = 0.10,
) -> dict[str, float]:
    """Left-minus-right forecast differences; negative favors Brier/log loss."""
    target = frame["next_range_stress"].to_numpy()
    left = np.clip(frame[left_model].to_numpy(), 1e-12, 1 - 1e-12)
    right = np.clip(frame[right_model].to_numpy(), 1e-12, 1 - 1e-12)
    return {
        "pr_auc_difference": float(
            average_precision_score(target, left)
            - average_precision_score(target, right)
        ),
        "brier_difference": float(
            brier_score_loss(target, left)
            - brier_score_loss(target, right)
        ),
        "log_loss_difference": float(
            log_loss(target, left) - log_loss(target, right)
        ),
        "top_fraction_precision_difference": float(
            top_fraction_precision(target, left, top_fraction)
            - top_fraction_precision(target, right, top_fraction)
        ),
    }


def interval_summary(
    draws: pd.DataFrame,
    identifiers: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    """Convert bootstrap draws to percentile interval summaries."""
    rows: list[dict[str, object]] = []
    grouped = draws.groupby(identifiers, sort=True, observed=True)
    for keys, sample in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(identifiers, keys))
        for metric in metric_columns:
            values = sample[metric].dropna().to_numpy()
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "bootstrap_mean": float(values.mean()),
                    "ci_lower": float(np.quantile(values, 0.025)),
                    "ci_upper": float(np.quantile(values, 0.975)),
                    "probability_above_zero": float((values > 0).mean()),
                    "draws": len(values),
                }
            )
    return pd.DataFrame(rows)


def daily_alert_assignments(
    evaluation: pd.DataFrame,
    budgets: Iterable[float],
) -> pd.DataFrame:
    """Create deterministic within-session top-budget alert assignments."""
    records: list[pd.DataFrame] = []
    ordered = evaluation.sort_values(
        ["evaluation_year", "model", "date", "probability", "security_id"],
        ascending=[True, True, True, False, True],
    )
    for budget in budgets:
        sample = ordered.copy()
        sample["_rank"] = sample.groupby(
            ["evaluation_year", "model", "date"], observed=True
        ).cumcount()
        sample["_session_rows"] = sample.groupby(
            ["evaluation_year", "model", "date"], observed=True
        )["security_id"].transform("size")
        sample["_alert_count"] = np.ceil(
            sample["_session_rows"] * float(budget)
        ).clip(lower=1)
        sample["alert"] = sample["_rank"] < sample["_alert_count"]
        sample["alert_budget"] = float(budget)
        records.append(
            sample[
                [
                    "fold",
                    "evaluation_year",
                    "date",
                    "symbol",
                    "security_id",
                    "next_range_stress",
                    "model",
                    "probability",
                    "alert_budget",
                    "alert",
                ]
            ]
        )
    return pd.concat(records, ignore_index=True)


def alert_counts(assignments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate operational alert outcomes by session."""
    sample = assignments.assign(
        tp=lambda x: x["alert"] & x["next_range_stress"].eq(1),
        fp=lambda x: x["alert"] & x["next_range_stress"].eq(0),
        fn=lambda x: ~x["alert"] & x["next_range_stress"].eq(1),
    )
    return (
        sample.groupby(
            ["evaluation_year", "date", "model", "alert_budget"],
            as_index=False,
            observed=True,
        )
        .agg(
            rows=("next_range_stress", "size"),
            positives=("next_range_stress", "sum"),
            alerts=("alert", "sum"),
            true_positives=("tp", "sum"),
            false_positives=("fp", "sum"),
            false_negatives=("fn", "sum"),
        )
    )


def alert_metrics(counts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate session counts into resource-constrained utility metrics."""
    counts = counts.assign(
        stress_session=counts["positives"].gt(0),
        detected_stress_session=counts["true_positives"].gt(0),
    )
    grouped = (
        counts.groupby(
            ["evaluation_year", "model", "alert_budget"],
            as_index=False,
            observed=True,
        )
        .agg(
            sessions=("date", "size"),
            rows=("rows", "sum"),
            positives=("positives", "sum"),
            alerts=("alerts", "sum"),
            true_positives=("true_positives", "sum"),
            false_positives=("false_positives", "sum"),
            false_negatives=("false_negatives", "sum"),
            stress_sessions=("stress_session", "sum"),
            detected_stress_sessions=("detected_stress_session", "sum"),
        )
    )
    grouped["precision"] = (
        grouped["true_positives"] / grouped["alerts"].clip(lower=1)
    )
    grouped["recall"] = (
        grouped["true_positives"] / grouped["positives"].clip(lower=1)
    )
    grouped["prevalence"] = grouped["positives"] / grouped["rows"]
    grouped["lift"] = grouped["precision"] / grouped["prevalence"]
    grouped["alerts_per_session"] = grouped["alerts"] / grouped["sessions"]
    grouped["stress_session_detection_rate"] = (
        grouped["detected_stress_sessions"]
        / grouped["stress_sessions"].clip(lower=1)
    )
    return grouped


def cost_utility(
    metrics: pd.DataFrame,
    missed_event_costs: Iterable[float],
) -> pd.DataFrame:
    """Evaluate false-alert cost 1 versus a grid of missed-event costs."""
    rows: list[pd.DataFrame] = []
    for missed_cost in missed_event_costs:
        sample = metrics.copy()
        sample["missed_event_cost"] = float(missed_cost)
        sample["total_cost"] = (
            sample["false_positives"]
            + float(missed_cost) * sample["false_negatives"]
        )
        sample["no_alert_cost"] = float(missed_cost) * sample["positives"]
        sample["normalized_cost_reduction"] = (
            sample["no_alert_cost"] - sample["total_cost"]
        ) / sample["no_alert_cost"].clip(lower=np.finfo("float64").eps)
        rows.append(sample)
    return pd.concat(rows, ignore_index=True)


def decision_curve(
    evaluation: pd.DataFrame,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    """Classical decision-curve net benefit at calibrated risk thresholds."""
    rows: list[dict[str, object]] = []
    for (year, model), sample in evaluation.groupby(
        ["evaluation_year", "model"], sort=True, observed=True
    ):
        target = sample["next_range_stress"].to_numpy()
        probability = sample["probability"].to_numpy()
        prevalence = float(target.mean())
        total = len(sample)
        for threshold in thresholds:
            selected = probability >= float(threshold)
            tp = int(np.sum(selected & (target == 1)))
            fp = int(np.sum(selected & (target == 0)))
            odds = float(threshold) / (1 - float(threshold))
            net_benefit = tp / total - fp / total * odds
            treat_all = prevalence - (1 - prevalence) * odds
            rows.append(
                {
                    "evaluation_year": int(year),
                    "model": model,
                    "threshold": float(threshold),
                    "prevalence": prevalence,
                    "alerts": int(selected.sum()),
                    "net_benefit": float(net_benefit),
                    "standardized_net_benefit": float(
                        net_benefit / prevalence
                    ),
                    "treat_all_net_benefit": float(treat_all),
                    "treat_none_net_benefit": 0.0,
                }
            )
    return pd.DataFrame(rows)
