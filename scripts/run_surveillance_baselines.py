"""Frozen four-stage baselines for the 2024-2025 surveillance folds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nepse_ai.evaluation import extended_classification_metrics
from nepse_ai.utils import mirrored_console
from run_rolling_baselines import FEATURE_SETS, make_model
from transaction_graph_loader import BENCHMARK


CONTEXT = [
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
]


def probability_logit(probability: np.ndarray) -> np.ndarray:
    epsilon = np.finfo("float64").eps
    clipped = np.clip(probability, epsilon, 1 - epsilon)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def split_masks(
    panel: pd.DataFrame,
    fold: dict[str, object],
    train_start: str,
) -> dict[str, pd.Series]:
    date = panel["date"]
    development = date.between(
        pd.Timestamp(train_start),
        pd.Timestamp(str(fold["development_end"])),
    )
    selection = date.dt.year.eq(int(fold["selection_year"]))
    calibration_dates = sorted(
        date.loc[
            date.dt.year.eq(int(fold["calibration_year"]))
        ].drop_duplicates()
    )
    split = len(calibration_dates) // 2
    masks = {
        "refit": development | selection,
        "platt": date.isin(set(calibration_dates[:split])),
        "conformal": date.isin(set(calibration_dates[split:])),
        "evaluation": date.dt.year.eq(int(fold["evaluation_year"])),
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError(f"Empty baseline partition for {fold}")
    return masks


def calibrated_frames(
    panel: pd.DataFrame,
    masks: dict[str, pd.Series],
    raw_probability: dict[str, np.ndarray],
    model_name: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, LogisticRegression]:
    calibrator = LogisticRegression(C=1e6, solver="lbfgs")
    calibrator.fit(
        probability_logit(raw_probability["platt"]),
        panel.loc[masks["platt"], "next_range_stress"].to_numpy(),
    )
    calibration_parts = []
    for part in ("platt", "conformal"):
        frame = panel.loc[masks[part], CONTEXT].copy()
        frame["probability_raw"] = raw_probability[part]
        frame["probability_calibrated"] = calibrator.predict_proba(
            probability_logit(raw_probability[part])
        )[:, 1]
        frame["calibration_part"] = part
        calibration_parts.append(frame)
    evaluation = panel.loc[masks["evaluation"], CONTEXT].copy()
    evaluation["probability_raw"] = raw_probability["evaluation"]
    evaluation["probability_calibrated"] = calibrator.predict_proba(
        probability_logit(raw_probability["evaluation"])
    )[:, 1]
    calibration = pd.concat(calibration_parts, ignore_index=True)
    for frame in (calibration, evaluation):
        frame["model"] = model_name
        frame["seed"] = seed
    return calibration, evaluation, calibrator


def run_fold(
    panel: pd.DataFrame,
    fold: dict[str, object],
    configuration: dict[str, object],
) -> None:
    fold_name = str(fold["fold"])
    result_root = Path(str(configuration["result_root"])) / fold_name
    artifact_root = (
        Path(str(configuration["model_artifact_root"])) / fold_name
    )
    result_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    masks = split_masks(panel, fold, str(configuration["train_start"]))
    target_refit = panel.loc[
        masks["refit"], "next_range_stress"
    ].to_numpy()
    calibration_frames = []
    evaluation_frames = []
    metric_rows = []
    started = time.time()
    for name in configuration["models"]:
        features = FEATURE_SETS[str(name)]
        for configured_seed in configuration["seeds"]:
            seed = int(configured_seed)
            model = make_model(str(name), seed)
            model.fit(panel.loc[masks["refit"], features], target_refit)
            raw = {
                part: model.predict_proba(
                    panel.loc[masks[part], features]
                )[:, 1]
                for part in ("platt", "conformal", "evaluation")
            }
            calibration, evaluation, calibrator = calibrated_frames(
                panel, masks, raw, str(name), seed
            )
            calibration_frames.append(calibration)
            evaluation_frames.append(evaluation)
            metrics = {
                "fold": fold_name,
                "evaluation_year": int(fold["evaluation_year"]),
                "model": str(name),
                "seed": seed,
                **extended_classification_metrics(
                    evaluation["next_range_stress"].to_numpy(),
                    evaluation["probability_calibrated"].to_numpy(),
                ),
            }
            metric_rows.append(metrics)
            joblib.dump(
                {
                    "model": model,
                    "calibrator": calibrator,
                    "features": features,
                    "fold": fold,
                },
                artifact_root / f"{name}_seed{seed}.joblib",
                compress=3,
            )
            print(json.dumps(metrics, sort_keys=True), flush=True)

    raw_score = {
        part: panel.loc[masks[part], "range_t"].to_numpy(
            dtype="float64"
        )
        for part in ("platt", "conformal", "evaluation")
    }
    raw_probability = {
        part: 1 / (1 + np.exp(-np.clip(score, -40, 40)))
        for part, score in raw_score.items()
    }
    calibration, evaluation, calibrator = calibrated_frames(
        panel, masks, raw_probability, "range_persistence", 0
    )
    calibration_frames.append(calibration)
    evaluation_frames.append(evaluation)
    metrics = {
        "fold": fold_name,
        "evaluation_year": int(fold["evaluation_year"]),
        "model": "range_persistence",
        "seed": 0,
        **extended_classification_metrics(
            evaluation["next_range_stress"].to_numpy(),
            evaluation["probability_calibrated"].to_numpy(),
        ),
    }
    metric_rows.append(metrics)
    joblib.dump(
        {"calibrator": calibrator, "feature": "range_t", "fold": fold},
        artifact_root / "range_persistence.joblib",
        compress=3,
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)

    pd.concat(calibration_frames, ignore_index=True).to_parquet(
        result_root / "calibration_predictions.parquet",
        index=False,
        compression="zstd",
    )
    pd.concat(evaluation_frames, ignore_index=True).to_parquet(
        result_root / "evaluation_predictions.parquet",
        index=False,
        compression="zstd",
    )
    pd.DataFrame(metric_rows).to_csv(
        result_root / "evaluation_metrics.csv", index=False
    )
    metadata = {
        "status": "complete",
        "protocol": "refit_platt_conformal_evaluation",
        "fold": fold,
        "train_start": configuration["train_start"],
        "calibration_split": "chronological halves",
        "models": configuration["models"],
        "seeds": configuration["seeds"],
        "feature_sets": {
            model: FEATURE_SETS[model] for model in configuration["models"]
        },
        "model_artifact_root": str(artifact_root),
        "elapsed_seconds": time.time() - started,
    }
    (result_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
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
    panel = panel.sort_values(["date", "security_id"]).reset_index(drop=True)
    Path(str(configuration["result_root"])).mkdir(parents=True, exist_ok=True)
    with mirrored_console(Path(str(configuration["log_file"]))):
        print(f"Surveillance baseline config: {arguments.config.resolve()}")
        for position, fold in enumerate(configuration["folds"], start=1):
            result_root = (
                Path(str(configuration["result_root"])) / str(fold["fold"])
            )
            complete = (
                (result_root / "metadata.json").exists()
                and (result_root / "evaluation_predictions.parquet").exists()
                and (
                    result_root / "calibration_predictions.parquet"
                ).exists()
            )
            if arguments.resume and complete:
                print(f"Skipping completed fold {fold['fold']}", flush=True)
                continue
            print(
                f"Starting baseline fold {position}/"
                f"{len(configuration['folds'])}: {fold}",
                flush=True,
            )
            run_fold(panel, fold, configuration)
            print(f"Completed baseline fold {fold['fold']}", flush=True)
        print("All surveillance baseline folds completed.", flush=True)


if __name__ == "__main__":
    main()
