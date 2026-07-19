"""Rare-event discrimination and calibration metrics."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


def classification_metrics(
    target: np.ndarray, probability: np.ndarray
) -> dict[str, float]:
    return {
        "pr_auc": average_precision_score(target, probability),
        "roc_auc": roc_auc_score(target, probability),
        "brier": brier_score_loss(target, probability),
        "log_loss": log_loss(target, probability),
    }


def expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    """Equal-frequency expected calibration error."""
    frame = np.rec.fromarrays(
        [target.astype("float64"), probability.astype("float64")],
        names=("target", "probability"),
    )
    order = np.argsort(frame.probability, kind="stable")
    groups = np.array_split(order, bins)
    total = len(target)
    return float(
        sum(
            len(index)
            / total
            * abs(
                frame.target[index].mean()
                - frame.probability[index].mean()
            )
            for index in groups
            if len(index)
        )
    )


def calibration_slope_intercept(
    target: np.ndarray,
    probability: np.ndarray,
) -> tuple[float, float]:
    """Fit outcome ~ intercept + slope * prediction log-odds."""
    epsilon = np.finfo("float64").eps
    clipped = np.clip(probability, epsilon, 1 - epsilon)
    log_odds = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs")
    model.fit(log_odds, target)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def extended_classification_metrics(
    target: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    """Discrimination, proper scores, alert utility, and calibration."""
    result = classification_metrics(target, probability)
    threshold = np.quantile(probability, 0.90)
    selected = probability >= threshold
    precision = float(target[selected].mean())
    prevalence = float(target.mean())
    slope, intercept = calibration_slope_intercept(target, probability)
    result.update(
        prevalence=prevalence,
        ece_10=expected_calibration_error(target, probability, bins=10),
        calibration_slope=slope,
        calibration_intercept=intercept,
        top_decile_precision=precision,
        top_decile_lift=precision / prevalence,
    )
    return result


def conformal_quantile(
    target: np.ndarray,
    probability: np.ndarray,
    alpha: float,
) -> float:
    """Finite-sample split-conformal quantile for binary label sets."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    scores = np.where(target == 1, 1 - probability, probability)
    level = min(1.0, np.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
    return float(np.quantile(scores, level, method="higher"))


def conformal_set_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    quantile: float,
) -> dict[str, float]:
    """Coverage and efficiency of binary split-conformal prediction sets."""
    include_zero = probability <= quantile
    include_one = 1 - probability <= quantile
    covered = np.where(target == 1, include_one, include_zero)
    size = include_zero.astype("int8") + include_one.astype("int8")
    positive_singleton = include_one & ~include_zero
    return {
        "conformal_quantile": float(quantile),
        "conformal_coverage": float(covered.mean()),
        "conformal_average_set_size": float(size.mean()),
        "conformal_singleton_rate": float((size == 1).mean()),
        "conformal_empty_rate": float((size == 0).mean()),
        "conformal_positive_singleton_rate": float(
            positive_singleton.mean()
        ),
        "conformal_positive_singleton_precision": float(
            target[positive_singleton].mean()
            if positive_singleton.any()
            else np.nan
        ),
    }
