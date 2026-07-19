"""Sequence-aware GRU occlusion and matched LightGBM TreeSHAP."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss

from prepare_graph_benchmark import (
    BASE_FEATURES,
    MARKET_STATE_FEATURES,
    STOCK_BROKER_FEATURES,
)
from run_temporal_surveillance import (
    FEATURES,
    chronological_batches,
    fold_masks,
    make_model,
    replay,
)
from transaction_graph_loader import BENCHMARK

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


GROUPS = {
    "group_price_liquidity": BASE_FEATURES,
    "group_broker_state": STOCK_BROKER_FEATURES,
    "group_market_state": MARKET_STATE_FEATURES,
}


def values_from_scaling(
    panel: pd.DataFrame,
    path: Path,
) -> np.ndarray:
    scaling = pd.read_csv(path).set_index("feature").loc[FEATURES]
    values = (
        panel[FEATURES].to_numpy(dtype="float64")
        - scaling["mean"].to_numpy()
    ) / scaling["std"].replace(0, 1).to_numpy()
    return np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype("float32")


def path_probabilities(
    model: torch.nn.Module,
    batches: dict[str, list[pd.DataFrame]],
    values: np.ndarray,
    device: torch.device,
    security_count: int,
    coefficient: float,
    intercept: float,
) -> pd.DataFrame:
    hidden, _ = replay(
        model,
        batches["refit"],
        values,
        device,
        security_count,
    )
    hidden, _ = replay(
        model,
        batches["platt"],
        values,
        device,
        security_count,
        hidden=hidden,
    )
    hidden, _ = replay(
        model,
        batches["conformal"],
        values,
        device,
        security_count,
        hidden=hidden,
    )
    _, evaluation = replay(
        model,
        batches["evaluation"],
        values,
        device,
        security_count,
        hidden=hidden,
        collect=True,
    )
    assert evaluation is not None
    calibrated_logit = (
        coefficient * evaluation["logit"].to_numpy() + intercept
    )
    evaluation["probability"] = 1 / (
        1 + np.exp(-np.clip(calibrated_logit, -40, 40))
    )
    return evaluation


def temporal_occlusion(
    configuration: dict[str, object],
    temporal_configuration: dict[str, object],
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = torch.device("cuda")
    security_count = int(panel["security_id"].max()) + 1
    records = []
    integrity = []
    for fold in temporal_configuration["folds"]:
        fold_name = str(fold["fold"])
        result_root = Path(str(configuration["temporal_root"])) / fold_name
        artifact_root = (
            Path(str(configuration["temporal_artifact_root"])) / fold_name
        )
        masks = fold_masks(
            panel,
            fold,
            str(temporal_configuration["train_start"]),
        )
        batches = {
            name: chronological_batches(panel, mask)
            for name, mask in masks.items()
            if name in {"refit", "platt", "conformal", "evaluation"}
        }
        values = values_from_scaling(
            panel, result_root / "refit_scaling.csv"
        )
        saved = pd.read_parquet(
            result_root / "evaluation_predictions.parquet"
        )
        seeds = list(temporal_configuration["seeds"])[
            : int(
                configuration.get(
                    "seed_limit", len(temporal_configuration["seeds"])
                )
            )
        ]
        feature_limit = int(
            configuration.get("feature_limit", len(FEATURES))
        )
        for seed in seeds:
            checkpoint_path = (
                artifact_root / f"temporal_tabular_seed{seed}.pt"
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )
            model = make_model(
                int(checkpoint["hidden_dimension"]),
                float(checkpoint["dropout"]),
                device,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            coefficient = float(
                checkpoint["calibration_coefficient"][0][0]
            )
            intercept = float(checkpoint["calibration_intercept"][0])
            reference = path_probabilities(
                model,
                batches,
                values,
                device,
                security_count,
                coefficient,
                intercept,
            )
            saved_seed = (
                saved.loc[saved["seed"].eq(seed)]
                .sort_values(["date", "security_id"])
                .reset_index(drop=True)
            )
            reference = reference.sort_values(
                ["date", "security_id"]
            ).reset_index(drop=True)
            maximum_difference = float(
                np.max(
                    np.abs(
                        reference["probability"].to_numpy()
                        - saved_seed[
                            "probability_calibrated"
                        ].to_numpy()
                    )
                )
            )
            integrity.append(
                {
                    "fold": fold_name,
                    "seed": seed,
                    "maximum_probability_replay_difference": (
                        maximum_difference
                    ),
                }
            )
            target = reference["next_range_stress"].to_numpy()
            reference_probability = reference["probability"].to_numpy()
            reference_pr = average_precision_score(
                target, reference_probability
            )
            reference_brier = brier_score_loss(
                target, reference_probability
            )
            reference_top = target[
                reference_probability
                >= np.quantile(reference_probability, 0.90)
            ].mean()
            ablations = {
                **{
                    feature: [feature]
                    for feature in FEATURES[:feature_limit]
                },
                **(GROUPS if feature_limit >= len(FEATURES) else {}),
            }
            for name, removed_features in ablations.items():
                ablated_values = values.copy()
                columns = [FEATURES.index(feature) for feature in removed_features]
                ablated_values[:, columns] = 0
                ablated = path_probabilities(
                    model,
                    batches,
                    ablated_values,
                    device,
                    security_count,
                    coefficient,
                    intercept,
                )
                probability = ablated["probability"].to_numpy()
                ablated_pr = average_precision_score(target, probability)
                ablated_brier = brier_score_loss(target, probability)
                ablated_top = target[
                    probability >= np.quantile(probability, 0.90)
                ].mean()
                records.append(
                    {
                        "fold": fold_name,
                        "evaluation_year": int(fold["evaluation_year"]),
                        "seed": seed,
                        "ablation": name,
                        "ablation_type": (
                            "group"
                            if name.startswith("group_")
                            else "feature"
                        ),
                        "removed_feature_count": len(removed_features),
                        "reference_pr_auc": reference_pr,
                        "ablated_pr_auc": ablated_pr,
                        "pr_auc_drop": reference_pr - ablated_pr,
                        "reference_brier": reference_brier,
                        "ablated_brier": ablated_brier,
                        "brier_increase": ablated_brier - reference_brier,
                        "top_decile_precision_drop": (
                            reference_top - ablated_top
                        ),
                        "mean_absolute_probability_change": float(
                            np.mean(np.abs(probability - reference_probability))
                        ),
                    }
                )
                print(
                    f"{fold_name} seed={seed} ablation={name} "
                    f"PR-drop={reference_pr - ablated_pr:+.6f}",
                    flush=True,
                )
    return pd.DataFrame(records), pd.DataFrame(integrity)


def tree_shap_importance(
    configuration: dict[str, object],
    panel: pd.DataFrame,
    temporal_configuration: dict[str, object],
) -> pd.DataFrame:
    rows = []
    sample_size = int(configuration["tree_shap_sample_per_year"])
    random_seed = int(configuration["random_seed"])
    for fold in temporal_configuration["folds"]:
        fold_name = str(fold["fold"])
        evaluation = panel.loc[
            panel["date"].dt.year.eq(int(fold["evaluation_year"]))
        ]
        sample = evaluation.sample(
            n=min(sample_size, len(evaluation)),
            random_state=random_seed + int(fold["evaluation_year"]),
        )
        artifact_root = (
            Path(str(configuration["baseline_artifact_root"])) / fold_name
        )
        seeds = list(temporal_configuration["seeds"])[
            : int(
                configuration.get(
                    "seed_limit", len(temporal_configuration["seeds"])
                )
            )
        ]
        for seed in seeds:
            payload = joblib.load(
                artifact_root / f"lgbm_state_seed{seed}.joblib"
            )
            model = payload["model"]
            features = payload["features"]
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(sample[features])
            if isinstance(values, list):
                values = values[-1]
            values = np.asarray(values)
            for feature, importance in zip(
                features, np.abs(values).mean(axis=0)
            ):
                rows.append(
                    {
                        "fold": fold_name,
                        "evaluation_year": int(fold["evaluation_year"]),
                        "seed": seed,
                        "feature": feature,
                        "mean_absolute_shap": float(importance),
                        "sample_rows": len(sample),
                    }
                )
    return pd.DataFrame(rows)


def stability_summary(
    occlusion: pd.DataFrame,
    shap_importance: pd.DataFrame,
) -> dict[str, object]:
    feature_occlusion = (
        occlusion.loc[occlusion["ablation_type"].eq("feature")]
        .groupby(["evaluation_year", "ablation"], as_index=False)[
            "mean_absolute_probability_change"
        ]
        .mean()
    )
    years = [
        int(year)
        for year in sorted(feature_occlusion["evaluation_year"].unique())
    ]
    if len(years) < 2:
        return {
            "years": years,
            "temporal_occlusion_rank_spearman": None,
            "temporal_occlusion_top10_jaccard": None,
            "tree_shap_rank_spearman": None,
        }
    first = feature_occlusion.loc[
        feature_occlusion["evaluation_year"].eq(years[0])
    ].set_index("ablation")["mean_absolute_probability_change"]
    second = feature_occlusion.loc[
        feature_occlusion["evaluation_year"].eq(years[1])
    ].set_index("ablation")["mean_absolute_probability_change"]
    common = first.index.intersection(second.index)
    temporal_spearman = float(spearmanr(first[common], second[common]).statistic)
    temporal_top = {
        year: set(
            feature_occlusion.loc[
                feature_occlusion["evaluation_year"].eq(year)
            ]
            .nlargest(10, "mean_absolute_probability_change")["ablation"]
        )
        for year in years
    }
    shap_grouped = (
        shap_importance.groupby(
            ["evaluation_year", "feature"], as_index=False
        )["mean_absolute_shap"]
        .mean()
    )
    shap_first = shap_grouped.loc[
        shap_grouped["evaluation_year"].eq(years[0])
    ].set_index("feature")["mean_absolute_shap"]
    shap_second = shap_grouped.loc[
        shap_grouped["evaluation_year"].eq(years[1])
    ].set_index("feature")["mean_absolute_shap"]
    shap_common = shap_first.index.intersection(shap_second.index)
    shap_spearman = float(
        spearmanr(
            shap_first[shap_common],
            shap_second[shap_common],
        ).statistic
    )
    return {
        "years": years,
        "temporal_occlusion_rank_spearman": temporal_spearman,
        "temporal_occlusion_top10_jaccard": (
            len(temporal_top[years[0]] & temporal_top[years[1]])
            / len(temporal_top[years[0]] | temporal_top[years[1]])
        ),
        "tree_shap_rank_spearman": shap_spearman,
    }


def seed_stability(occlusion: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample = occlusion.loc[occlusion["ablation_type"].eq("feature")]
    for year, group in sample.groupby("evaluation_year", sort=True):
        pivot = group.pivot(
            index="ablation",
            columns="seed",
            values="mean_absolute_probability_change",
        )
        for left, right in combinations(pivot.columns, 2):
            rows.append(
                {
                    "evaluation_year": year,
                    "seed_left": left,
                    "seed_right": right,
                    "rank_spearman": float(
                        spearmanr(pivot[left], pivot[right]).statistic
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_figure(
    occlusion: pd.DataFrame,
    shap_importance: pd.DataFrame,
    output: Path,
) -> None:
    temporal = (
        occlusion.loc[occlusion["ablation_type"].eq("feature")]
        .groupby(["evaluation_year", "ablation"], as_index=False)[
            "mean_absolute_probability_change"
        ]
        .mean()
    )
    shap_grouped = (
        shap_importance.groupby(
            ["evaluation_year", "feature"], as_index=False
        )["mean_absolute_shap"]
        .mean()
    )
    years = sorted(temporal["evaluation_year"].unique())
    figure, axes = plt.subplots(2, 2, figsize=(13, 10))
    for column, year in enumerate(years):
        temporal_sample = temporal.loc[
            temporal["evaluation_year"].eq(year)
        ].nlargest(12, "mean_absolute_probability_change")
        axes[0, column].barh(
            temporal_sample["ablation"][::-1],
            temporal_sample["mean_absolute_probability_change"][::-1],
            color="#2E8B57",
        )
        axes[0, column].set(
            title=f"{year} sequence-aware GRU occlusion",
            xlabel="Mean absolute probability change",
        )
        shap_sample = shap_grouped.loc[
            shap_grouped["evaluation_year"].eq(year)
        ].nlargest(12, "mean_absolute_shap")
        axes[1, column].barh(
            shap_sample["feature"][::-1],
            shap_sample["mean_absolute_shap"][::-1],
            color="#6C8EBF",
        )
        axes[1, column].set(
            title=f"{year} LightGBM TreeSHAP",
            xlabel="Mean |SHAP value|",
        )
    figure.tight_layout()
    figure.savefig(output / "surveillance_explanations.png", dpi=300)
    figure.savefig(output / "surveillance_explanations.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    configuration = json.loads(
        arguments.config.read_text(encoding="utf-8")
    )
    temporal_configuration = json.loads(
        Path(str(configuration["temporal_config"])).read_text(
            encoding="utf-8"
        )
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required for temporal occlusion")
    panel = pd.read_parquet(BENCHMARK / "stock_stress_labels.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "security_id"]).reset_index(drop=True)
    panel["_row"] = np.arange(len(panel))
    output = Path(str(configuration["output_root"]))
    output.mkdir(parents=True, exist_ok=True)

    occlusion, integrity = temporal_occlusion(
        configuration, temporal_configuration, panel
    )
    shap_importance = tree_shap_importance(
        configuration, panel, temporal_configuration
    )
    stability = stability_summary(occlusion, shap_importance)
    seed_rank_stability = seed_stability(occlusion)
    occlusion.to_parquet(
        output / "temporal_occlusion.parquet",
        index=False,
        compression="zstd",
    )
    (
        occlusion.groupby(
            ["evaluation_year", "ablation", "ablation_type"],
            as_index=False,
        )
        .agg(
            pr_auc_drop_mean=("pr_auc_drop", "mean"),
            pr_auc_drop_std=("pr_auc_drop", "std"),
            brier_increase_mean=("brier_increase", "mean"),
            probability_change_mean=(
                "mean_absolute_probability_change",
                "mean",
            ),
            probability_change_std=(
                "mean_absolute_probability_change",
                "std",
            ),
        )
        .to_csv(output / "temporal_occlusion_summary.csv", index=False)
    )
    integrity.to_csv(output / "replay_integrity.csv", index=False)
    shap_importance.to_parquet(
        output / "lgbm_tree_shap.parquet",
        index=False,
        compression="zstd",
    )
    (
        shap_importance.groupby(
            ["evaluation_year", "feature"], as_index=False
        )
        .agg(
            mean_absolute_shap=("mean_absolute_shap", "mean"),
            seed_std=("mean_absolute_shap", "std"),
        )
        .to_csv(output / "lgbm_tree_shap_summary.csv", index=False)
    )
    seed_rank_stability.to_csv(
        output / "temporal_seed_rank_stability.csv", index=False
    )
    (output / "explanation_stability.json").write_text(
        json.dumps(stability, indent=2), encoding="utf-8"
    )
    make_figure(occlusion, shap_importance, output)
    metadata = {
        "status": "complete",
        "temporal_explanation": (
            "Entire historical sequence replayed with standardized feature "
            "set to its refit-development mean."
        ),
        "tree_explanation": "TreeSHAP on fixed LightGBM outer-year samples",
        "tree_shap_sample_per_year": int(
            configuration["tree_shap_sample_per_year"]
        ),
        "stability": stability,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(stability, indent=2), flush=True)


if __name__ == "__main__":
    main()
