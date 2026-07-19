"""Run frozen transparent and tree baselines on chronological outer folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nepse_ai.utils import mirrored_console
from prepare_graph_benchmark import (
    BASE_FEATURES,
    MARKET_STATE_FEATURES,
    STOCK_BROKER_FEATURES,
)
from transaction_graph_loader import BENCHMARK


FEATURE_SETS = {
    "logit_state": [
        *BASE_FEATURES,
        *STOCK_BROKER_FEATURES,
        *MARKET_STATE_FEATURES,
    ],
    "lgbm_price_liquidity": BASE_FEATURES,
    "lgbm_state": [
        *BASE_FEATURES,
        *STOCK_BROKER_FEATURES,
        *MARKET_STATE_FEATURES,
    ],
}
IDENTITY = [
    "date",
    "symbol",
    "security_id",
    "next_range_stress",
]


def make_model(name: str, seed: int) -> object:
    if name == "logit_state":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.3,
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def platt_calibrate(
    calibration_target: np.ndarray,
    calibration_probability: np.ndarray,
    evaluation_probability: np.ndarray,
) -> tuple[np.ndarray, LogisticRegression]:
    epsilon = np.finfo("float64").eps

    def logit(probability: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability, epsilon, 1 - epsilon)
        return np.log(clipped / (1 - clipped)).reshape(-1, 1)

    calibrator = LogisticRegression(C=1e6, solver="lbfgs")
    calibrator.fit(logit(calibration_probability), calibration_target)
    calibrated = calibrator.predict_proba(logit(evaluation_probability))[:, 1]
    return calibrated, calibrator


def scores(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    threshold = np.quantile(probability, 0.90)
    selected = probability >= threshold
    precision = float(target[selected].mean())
    prevalence = float(target.mean())
    return {
        "pr_auc": float(average_precision_score(target, probability)),
        "roc_auc": float(roc_auc_score(target, probability)),
        "brier": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability)),
        "top_decile_precision": precision,
        "top_decile_lift": precision / prevalence,
    }


def fit_fold(
    panel: pd.DataFrame,
    fold: dict[str, object],
    configuration: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development_end = pd.Timestamp(str(fold["development_end"]))
    calibration_year = int(fold["calibration_year"])
    evaluation_year = int(fold["evaluation_year"])
    train_start = pd.Timestamp(str(configuration["train_start"]))
    development = panel["date"].between(train_start, development_end)
    calibration = panel["date"].dt.year.eq(calibration_year)
    evaluation = panel["date"].dt.year.eq(evaluation_year)
    if not (development.any() and calibration.any() and evaluation.any()):
        raise ValueError(f"Empty chronological partition for {fold}")

    fold_name = str(fold["fold"])
    artifact_root = (
        Path(str(configuration["model_artifact_root"])) / fold_name
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    target_development = panel.loc[
        development, "next_range_stress"
    ].to_numpy()
    target_calibration = panel.loc[
        calibration, "next_range_stress"
    ].to_numpy()
    target_evaluation = panel.loc[
        evaluation, "next_range_stress"
    ].to_numpy()

    for name in configuration["models"]:
        features = FEATURE_SETS[str(name)]
        for seed in configuration["seeds"]:
            seed = int(seed)
            model = make_model(str(name), seed)
            model.fit(panel.loc[development, features], target_development)
            calibration_probability = model.predict_proba(
                panel.loc[calibration, features]
            )[:, 1]
            evaluation_raw = model.predict_proba(
                panel.loc[evaluation, features]
            )[:, 1]
            evaluation_probability, calibrator = platt_calibrate(
                target_calibration,
                calibration_probability,
                evaluation_raw,
            )
            output = panel.loc[evaluation, IDENTITY].copy()
            output["probability_raw"] = evaluation_raw
            output["probability_calibrated"] = evaluation_probability
            output["model"] = str(name)
            output["seed"] = seed
            prediction_frames.append(output)
            metric_rows.append(
                {
                    "fold": fold_name,
                    "evaluation_year": evaluation_year,
                    "model": str(name),
                    "seed": seed,
                    **scores(target_evaluation, evaluation_probability),
                }
            )
            checkpoint = artifact_root / f"{name}_seed{seed}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "calibrator": calibrator,
                    "features": features,
                    "development_end": development_end.date().isoformat(),
                    "calibration_year": calibration_year,
                    "evaluation_year": evaluation_year,
                },
                checkpoint,
                compress=3,
            )
            print(
                json.dumps(metric_rows[-1], sort_keys=True),
                flush=True,
            )

    persistence_calibration = panel.loc[
        calibration, "range_t"
    ].to_numpy(dtype="float64")
    persistence_evaluation = panel.loc[
        evaluation, "range_t"
    ].to_numpy(dtype="float64")
    persistence_probability, persistence_calibrator = platt_calibrate(
        target_calibration,
        1 / (1 + np.exp(-persistence_calibration)),
        1 / (1 + np.exp(-persistence_evaluation)),
    )
    output = panel.loc[evaluation, IDENTITY].copy()
    output["probability_raw"] = persistence_evaluation
    output["probability_calibrated"] = persistence_probability
    output["model"] = "range_persistence"
    output["seed"] = 0
    prediction_frames.append(output)
    metric_rows.append(
        {
            "fold": fold_name,
            "evaluation_year": evaluation_year,
            "model": "range_persistence",
            "seed": 0,
            **scores(target_evaluation, persistence_probability),
        }
    )
    joblib.dump(
        {"calibrator": persistence_calibrator, "feature": "range_t"},
        artifact_root / "range_persistence.joblib",
        compress=3,
    )
    print(json.dumps(metric_rows[-1], sort_keys=True), flush=True)
    return (
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame(metric_rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    configuration = json.loads(
        arguments.config.read_text(encoding="utf-8")
    )
    unknown = set(configuration["models"]) - set(FEATURE_SETS)
    if unknown:
        raise ValueError(f"Unknown baseline models: {sorted(unknown)}")
    panel = pd.read_parquet(BENCHMARK / "stock_stress_labels.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    required = set(IDENTITY)
    for features in FEATURE_SETS.values():
        required.update(features)
    missing = required - set(panel)
    if missing:
        raise ValueError(f"Missing baseline columns: {sorted(missing)}")

    result_root = Path(str(configuration["result_root"]))
    result_root.mkdir(parents=True, exist_ok=True)
    all_predictions: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    with mirrored_console(Path(str(configuration["log_file"]))):
        print(f"Baseline configuration: {arguments.config.resolve()}")
        for fold in configuration["folds"]:
            fold_name = str(fold["fold"])
            fold_root = result_root / fold_name
            prediction_path = fold_root / "baseline_predictions.parquet"
            metric_path = fold_root / "baseline_metrics.csv"
            if (
                arguments.resume
                and prediction_path.exists()
                and metric_path.exists()
            ):
                print(f"Skipping completed fold {fold_name}", flush=True)
                all_predictions.append(pd.read_parquet(prediction_path))
                all_metrics.append(pd.read_csv(metric_path))
                continue
            print(f"Starting baseline fold {fold_name}", flush=True)
            predictions, metrics = fit_fold(panel, fold, configuration)
            fold_root.mkdir(parents=True, exist_ok=True)
            predictions.to_parquet(
                prediction_path,
                index=False,
                compression="zstd",
            )
            metrics.to_csv(metric_path, index=False)
            all_predictions.append(predictions)
            all_metrics.append(metrics)
            print(f"Completed baseline fold {fold_name}", flush=True)

        pd.concat(all_predictions, ignore_index=True).to_parquet(
            result_root / "rolling_baseline_predictions.parquet",
            index=False,
            compression="zstd",
        )
        pd.concat(all_metrics, ignore_index=True).to_csv(
            result_root / "rolling_baseline_metrics.csv",
            index=False,
        )
        metadata = {
            "status": "complete",
            "protocol": "fixed_hyperparameters_rolling_origin",
            "models": configuration["models"],
            "seeds": configuration["seeds"],
            "folds": configuration["folds"],
            "feature_sets": FEATURE_SETS,
            "calibration": "Platt scaling fitted only on calibration year",
        }
        (result_root / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        print("All rolling baseline folds completed.", flush=True)


if __name__ == "__main__":
    main()
