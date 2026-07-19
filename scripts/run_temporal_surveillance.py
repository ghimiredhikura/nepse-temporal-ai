"""Four-stage rolling temporal-GRU surveillance experiment.

Each outer fold uses development for epoch search, a subsequent selection
year for early stopping, refits on development plus selection, splits the
calibration year chronologically into Platt and conformal halves, and reads
the outer evaluation year only after all model choices are frozen.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from nepse_ai.evaluation import extended_classification_metrics
from nepse_ai.models import TemporalStressModel
from nepse_ai.training import EarlyStopping, WarmupCosineScheduler
from nepse_ai.utils import mirrored_console, save_local_checkpoint
from prepare_graph_benchmark import (
    BASE_FEATURES,
    MARKET_STATE_FEATURES,
    STOCK_BROKER_FEATURES,
)
from transaction_graph_loader import BENCHMARK


FEATURES = BASE_FEATURES + STOCK_BROKER_FEATURES + MARKET_STATE_FEATURES
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def chronological_batches(
    panel: pd.DataFrame,
    mask: pd.Series,
) -> list[pd.DataFrame]:
    sample = panel.loc[mask].sort_values(["date", "security_id"])
    return [group for _, group in sample.groupby("date", sort=True)]


def standardized_values(
    panel: pd.DataFrame,
    fit_mask: pd.Series,
) -> tuple[np.ndarray, pd.DataFrame]:
    means = panel.loc[fit_mask, FEATURES].mean()
    standard_deviations = (
        panel.loc[fit_mask, FEATURES].std(ddof=0).replace(0, 1)
    )
    values = (
        (
            panel[FEATURES].to_numpy(dtype="float64")
            - means.to_numpy()
        )
        / standard_deviations.to_numpy()
    )
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype("float32")
    scaling = pd.DataFrame(
        {
            "feature": FEATURES,
            "mean": means.to_numpy(),
            "std": standard_deviations.to_numpy(),
        }
    )
    return values, scaling


def session_tensors(
    batch: pd.DataFrame,
    values: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tabular = torch.as_tensor(
        values[batch["_row"].to_numpy()], device=device
    )
    security = torch.as_tensor(
        batch["security_id"].to_numpy(dtype="int64"), device=device
    )
    target = torch.as_tensor(
        batch["next_range_stress"].to_numpy(dtype="float32"),
        device=device,
    )
    return tabular, security, target


def make_model(
    hidden_dimension: int,
    dropout: float,
    device: torch.device,
) -> TemporalStressModel:
    return TemporalStressModel(
        "temporal_tabular",
        len(FEATURES),
        broker_count=1,
        hidden_dimension=hidden_dimension,
        dropout=dropout,
    ).to(device)


def train_epoch(
    model: TemporalStressModel,
    batches: list[pd.DataFrame],
    values: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: torch.nn.Module,
    device: torch.device,
    security_count: int,
    tbptt: int,
    gradient_clip_norm: float,
) -> float:
    model.train()
    hidden = torch.zeros(
        (security_count, model.hidden_dimension), device=device
    )
    optimizer.zero_grad(set_to_none=True)
    window_loss = None
    total_loss = 0.0
    window_steps = 0
    for step, batch in enumerate(batches, start=1):
        tabular, security, target = session_tensors(batch, values, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits, hidden = model.forward_session(
                tabular,
                security,
                hidden,
                events=None,
                security_count=security_count,
            )
            loss = criterion(logits, target)
        window_loss = loss if window_loss is None else window_loss + loss
        window_steps += 1
        total_loss += float(loss.detach())
        if step % tbptt == 0 or step == len(batches):
            assert window_loss is not None
            scaler.scale(window_loss / window_steps).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            hidden = hidden.detach()
            window_loss = None
            window_steps = 0
    return total_loss / len(batches)


@torch.no_grad()
def replay(
    model: TemporalStressModel,
    batches: list[pd.DataFrame],
    values: np.ndarray,
    device: torch.device,
    security_count: int,
    hidden: torch.Tensor | None = None,
    collect: bool = False,
) -> tuple[torch.Tensor, pd.DataFrame | None]:
    model.eval()
    if hidden is None:
        hidden = torch.zeros(
            (security_count, model.hidden_dimension), device=device
        )
    records = []
    for batch in batches:
        tabular, security, _ = session_tensors(batch, values, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits, hidden = model.forward_session(
                tabular,
                security,
                hidden,
                events=None,
                security_count=security_count,
            )
        if collect:
            output = batch[CONTEXT].copy()
            output["logit"] = logits.float().cpu().numpy()
            records.append(output)
    frame = pd.concat(records, ignore_index=True) if collect else None
    return hidden, frame


def select_epoch(
    seed: int,
    development_batches: list[pd.DataFrame],
    selection_batches: list[pd.DataFrame],
    values: np.ndarray,
    configuration: dict[str, object],
    device: torch.device,
    security_count: int,
) -> tuple[int, list[dict[str, object]]]:
    seed_everything(seed)
    model = make_model(
        int(configuration["hidden_dimension"]),
        float(configuration["dropout"]),
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configuration["learning_rate"]),
        weight_decay=float(configuration["weight_decay"]),
    )
    maximum_epochs = int(configuration["maximum_epochs"])
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_epochs=maximum_epochs,
        warmup_epochs=int(configuration["warmup_epochs"]),
        minimum_learning_rate=float(configuration["minimum_learning_rate"]),
    )
    stopper = EarlyStopping(
        patience=int(configuration["early_stopping_patience"]),
        minimum_epochs=int(configuration["minimum_epochs"]),
        minimum_delta=float(
            configuration["early_stopping_minimum_delta"]
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    best_epoch = 1
    best_score = -np.inf
    for epoch in range(1, maximum_epochs + 1):
        learning_rate = scheduler.step(epoch)[0]
        development_loss = train_epoch(
            model,
            development_batches,
            values,
            optimizer,
            scaler,
            criterion,
            device,
            security_count,
            int(configuration["tbptt_sessions"]),
            float(configuration["gradient_clip_norm"]),
        )
        hidden, _ = replay(
            model,
            development_batches,
            values,
            device,
            security_count,
        )
        _, selection = replay(
            model,
            selection_batches,
            values,
            device,
            security_count,
            hidden=hidden,
            collect=True,
        )
        assert selection is not None
        probability = 1 / (
            1 + np.exp(-np.clip(selection["logit"].to_numpy(), -40, 40))
        )
        score = float(
            average_precision_score(
                selection["next_range_stress"].to_numpy(),
                probability,
            )
        )
        record = {
            "phase": "selection",
            "seed": seed,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "development_loss": development_loss,
            "selection_pr_auc": score,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
        if stopper.update(epoch, score):
            break
    print(
        json.dumps(
            {
                "phase": "selection_complete",
                "seed": seed,
                "best_epoch": best_epoch,
                "best_selection_pr_auc": best_score,
            }
        ),
        flush=True,
    )
    return best_epoch, history


def refit_model(
    seed: int,
    epochs: int,
    schedule_epochs: int,
    batches: list[pd.DataFrame],
    values: np.ndarray,
    configuration: dict[str, object],
    device: torch.device,
    security_count: int,
) -> tuple[TemporalStressModel, list[dict[str, object]]]:
    seed_everything(seed)
    model = make_model(
        int(configuration["hidden_dimension"]),
        float(configuration["dropout"]),
        device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configuration["learning_rate"]),
        weight_decay=float(configuration["weight_decay"]),
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        total_epochs=schedule_epochs,
        warmup_epochs=int(configuration["warmup_epochs"]),
        minimum_learning_rate=float(configuration["minimum_learning_rate"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = torch.nn.BCEWithLogitsLoss()
    history = []
    for epoch in range(1, epochs + 1):
        learning_rate = scheduler.step(epoch)[0]
        loss = train_epoch(
            model,
            batches,
            values,
            optimizer,
            scaler,
            criterion,
            device,
            security_count,
            int(configuration["tbptt_sessions"]),
            float(configuration["gradient_clip_norm"]),
        )
        record = {
            "phase": "refit",
            "seed": seed,
            "epoch": epoch,
            "learning_rate": learning_rate,
            "refit_loss": loss,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
    return model, history


def fit_seed(
    seed: int,
    panel: pd.DataFrame,
    masks: dict[str, pd.Series],
    selection_values: np.ndarray,
    refit_values: np.ndarray,
    configuration: dict[str, object],
    device: torch.device,
    security_count: int,
    artifact_root: Path,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    batches = {
        name: chronological_batches(panel, mask)
        for name, mask in masks.items()
    }
    best_epoch, selection_history = select_epoch(
        seed,
        batches["development"],
        batches["selection"],
        selection_values,
        configuration,
        device,
        security_count,
    )
    model, refit_history = refit_model(
        seed,
        best_epoch,
        int(configuration["maximum_epochs"]),
        batches["refit"],
        refit_values,
        configuration,
        device,
        security_count,
    )
    hidden, _ = replay(
        model,
        batches["refit"],
        refit_values,
        device,
        security_count,
    )
    hidden, platt = replay(
        model,
        batches["platt"],
        refit_values,
        device,
        security_count,
        hidden=hidden,
        collect=True,
    )
    hidden, conformal = replay(
        model,
        batches["conformal"],
        refit_values,
        device,
        security_count,
        hidden=hidden,
        collect=True,
    )
    _, evaluation = replay(
        model,
        batches["evaluation"],
        refit_values,
        device,
        security_count,
        hidden=hidden,
        collect=True,
    )
    assert platt is not None and conformal is not None and evaluation is not None
    calibrator = LogisticRegression(C=1e6, solver="lbfgs")
    calibrator.fit(
        platt[["logit"]].to_numpy(),
        platt["next_range_stress"].to_numpy(),
    )
    calibration = pd.concat(
        [
            platt.assign(calibration_part="platt"),
            conformal.assign(calibration_part="conformal"),
        ],
        ignore_index=True,
    )
    for frame in (calibration, evaluation):
        frame["probability_raw"] = 1 / (
            1 + np.exp(-np.clip(frame["logit"].to_numpy(), -40, 40))
        )
        frame["probability_calibrated"] = calibrator.predict_proba(
            frame[["logit"]].to_numpy()
        )[:, 1]
        frame["model"] = "temporal_tabular"
        frame["seed"] = seed

    artifact_root.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    checkpoint = artifact_root / f"temporal_tabular_seed{seed}.pt"
    digest = save_local_checkpoint(
        {
            "model_state_dict": state,
            "model_type": "temporal_tabular",
            "seed": seed,
            "best_epoch": best_epoch,
            "tabular_features": FEATURES,
            "hidden_dimension": int(configuration["hidden_dimension"]),
            "dropout": float(configuration["dropout"]),
            "calibration_coefficient": calibrator.coef_.tolist(),
            "calibration_intercept": calibrator.intercept_.tolist(),
        },
        checkpoint,
    )
    target = evaluation["next_range_stress"].to_numpy()
    result = {
        "model": "temporal_tabular",
        "seed": seed,
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        **extended_classification_metrics(
            target,
            evaluation["probability_calibrated"].to_numpy(),
        ),
    }
    return (
        selection_history,
        refit_history,
        calibration,
        evaluation,
        result,
    )


def fold_masks(
    panel: pd.DataFrame,
    fold: dict[str, object],
    train_start: str,
) -> dict[str, pd.Series]:
    date = panel["date"]
    development_end = pd.Timestamp(str(fold["development_end"]))
    selection_year = int(fold["selection_year"])
    calibration_year = int(fold["calibration_year"])
    evaluation_year = int(fold["evaluation_year"])
    development = date.between(pd.Timestamp(train_start), development_end)
    selection = date.dt.year.eq(selection_year)
    calibration_dates = sorted(
        date.loc[date.dt.year.eq(calibration_year)].drop_duplicates()
    )
    if len(calibration_dates) < 20:
        raise ValueError(f"Insufficient calibration sessions for {fold}")
    split = len(calibration_dates) // 2
    platt_dates = set(calibration_dates[:split])
    conformal_dates = set(calibration_dates[split:])
    masks = {
        "development": development,
        "selection": selection,
        "refit": development | selection,
        "platt": date.isin(platt_dates),
        "conformal": date.isin(conformal_dates),
        "evaluation": date.dt.year.eq(evaluation_year),
    }
    if any(not mask.any() for mask in masks.values()):
        raise ValueError(f"Empty temporal partition for {fold}")
    return masks


def run_fold(
    panel: pd.DataFrame,
    fold: dict[str, object],
    configuration: dict[str, object],
    device: torch.device,
    security_count: int,
) -> None:
    fold_name = str(fold["fold"])
    result_root = Path(str(configuration["result_root"])) / fold_name
    artifact_root = Path(str(configuration["model_artifact_root"])) / fold_name
    result_root.mkdir(parents=True, exist_ok=True)
    masks = fold_masks(panel, fold, str(configuration["train_start"]))
    selection_values, selection_scaling = standardized_values(
        panel, masks["development"]
    )
    refit_values, refit_scaling = standardized_values(panel, masks["refit"])
    selection_scaling.to_csv(
        result_root / "selection_scaling.csv", index=False
    )
    refit_scaling.to_csv(result_root / "refit_scaling.csv", index=False)

    selection_histories = []
    refit_histories = []
    calibration_predictions = []
    evaluation_predictions = []
    results = []
    started = time.time()
    for seed in configuration["seeds"]:
        outputs = fit_seed(
            int(seed),
            panel,
            masks,
            selection_values,
            refit_values,
            configuration,
            device,
            security_count,
            artifact_root,
        )
        selection_histories.extend(outputs[0])
        refit_histories.extend(outputs[1])
        calibration_predictions.append(outputs[2])
        evaluation_predictions.append(outputs[3])
        results.append(outputs[4])

    pd.DataFrame(selection_histories).to_csv(
        result_root / "selection_history.csv", index=False
    )
    pd.DataFrame(refit_histories).to_csv(
        result_root / "refit_history.csv", index=False
    )
    pd.concat(calibration_predictions, ignore_index=True).to_parquet(
        result_root / "calibration_predictions.parquet",
        index=False,
        compression="zstd",
    )
    pd.concat(evaluation_predictions, ignore_index=True).to_parquet(
        result_root / "evaluation_predictions.parquet",
        index=False,
        compression="zstd",
    )
    pd.DataFrame(results).to_csv(
        result_root / "evaluation_metrics.csv", index=False
    )
    metadata = {
        "status": "complete",
        "protocol": "development_selection_refit_platt_conformal_evaluation",
        "fold": fold,
        "train_start": configuration["train_start"],
        "calibration_split": "chronological halves",
        "features": FEATURES,
        "seeds": configuration["seeds"],
        "maximum_epochs": configuration["maximum_epochs"],
        "model_artifact_root": str(artifact_root),
        "torch": torch.__version__,
        "device": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "cpu"
        ),
        "elapsed_seconds": time.time() - started,
    }
    (result_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(results).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    configuration = json.loads(
        arguments.config.read_text(encoding="utf-8")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    panel = pd.read_parquet(BENCHMARK / "stock_stress_labels.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    missing = set(FEATURES + CONTEXT) - set(panel)
    if missing:
        raise ValueError(f"Missing surveillance columns: {sorted(missing)}")
    panel = panel.sort_values(["date", "security_id"]).reset_index(drop=True)
    panel["_row"] = np.arange(len(panel))
    security_count = int(panel["security_id"].max()) + 1
    Path(str(configuration["result_root"])).mkdir(parents=True, exist_ok=True)
    with mirrored_console(Path(str(configuration["log_file"]))):
        print(f"Temporal surveillance config: {arguments.config.resolve()}")
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
                f"Starting temporal fold {position}/"
                f"{len(configuration['folds'])}: {fold}",
                flush=True,
            )
            run_fold(
                panel,
                fold,
                configuration,
                device,
                security_count,
            )
            print(f"Completed temporal fold {fold['fold']}", flush=True)
        print("All temporal surveillance folds completed.", flush=True)


if __name__ == "__main__":
    main()
