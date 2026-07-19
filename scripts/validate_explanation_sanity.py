"""Model-parameter randomization sanity checks for temporal occlusion."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from explain_temporal_surveillance import (
    path_probabilities,
    values_from_scaling,
)
from run_temporal_surveillance import (
    FEATURES,
    chronological_batches,
    fold_masks,
    make_model,
)
from transaction_graph_loader import BENCHMARK

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def reset_modules(
    model: torch.nn.Module,
    names: list[str],
    seed: int,
) -> torch.nn.Module:
    """Randomize selected trained submodules deterministically."""
    randomized = copy.deepcopy(model)
    torch.manual_seed(seed)
    for name in names:
        module = getattr(randomized, name)
        for child in module.modules():
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()
    return randomized


def occlusion_importance(
    model: torch.nn.Module,
    batches: dict[str, list[pd.DataFrame]],
    values: np.ndarray,
    device: torch.device,
    security_count: int,
    fold_name: str,
    evaluation_year: int,
    stage: str,
    randomization_seed: int | None,
) -> list[dict[str, object]]:
    """Replay the complete path and measure raw-probability sensitivity."""
    reference = path_probabilities(
        model,
        batches,
        values,
        device,
        security_count,
        coefficient=1.0,
        intercept=0.0,
    )["probability"].to_numpy()
    records: list[dict[str, object]] = []
    for position, feature in enumerate(FEATURES):
        ablated_values = values.copy()
        ablated_values[:, position] = 0
        ablated = path_probabilities(
            model,
            batches,
            ablated_values,
            device,
            security_count,
            coefficient=1.0,
            intercept=0.0,
        )["probability"].to_numpy()
        importance = float(np.mean(np.abs(ablated - reference)))
        records.append(
            {
                "fold": fold_name,
                "evaluation_year": evaluation_year,
                "stage": stage,
                "randomization_seed": randomization_seed,
                "feature": feature,
                "mean_absolute_raw_probability_change": importance,
            }
        )
        print(
            f"{fold_name} stage={stage} seed={randomization_seed} "
            f"feature={feature} importance={importance:.8f}",
            flush=True,
        )
    return records


def compare_with_trained(importance: pd.DataFrame) -> pd.DataFrame:
    """Compare trained rankings with each randomized-model explanation."""
    rows: list[dict[str, object]] = []
    for year, sample in importance.groupby(
        "evaluation_year", sort=True, observed=True
    ):
        trained = (
            sample.loc[sample["stage"].eq("trained")]
            .set_index("feature")[
                "mean_absolute_raw_probability_change"
            ]
        )
        trained_top = set(trained.nlargest(10).index)
        controls = sample.loc[~sample["stage"].eq("trained")]
        for (stage, seed), control in controls.groupby(
            ["stage", "randomization_seed"],
            sort=True,
            observed=True,
        ):
            control = control.set_index("feature")[
                "mean_absolute_raw_probability_change"
            ]
            common = trained.index.intersection(control.index)
            control_top = set(control.nlargest(10).index)
            rows.append(
                {
                    "evaluation_year": int(year),
                    "stage": stage,
                    "randomization_seed": int(seed),
                    "rank_spearman_with_trained": float(
                        spearmanr(trained[common], control[common]).statistic
                    ),
                    "top10_jaccard_with_trained": (
                        len(trained_top & control_top)
                        / len(trained_top | control_top)
                    ),
                    "control_to_trained_mean_signal_ratio": float(
                        control.mean() / trained.mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def make_figure(
    importance: pd.DataFrame,
    output: Path,
) -> None:
    years = sorted(importance["evaluation_year"].unique())
    figure, axes = plt.subplots(
        1, len(years), figsize=(6.2 * len(years), 5.2)
    )
    axes = np.atleast_1d(axes)
    for axis, year in zip(axes, years):
        sample = importance.loc[
            importance["evaluation_year"].eq(year)
        ]
        pivot = sample.pivot_table(
            index="feature",
            columns="stage",
            values="mean_absolute_raw_probability_change",
            aggfunc="mean",
        )
        for stage, color in (
            ("predictor_randomized", "#D9895B"),
            ("recurrent_predictor_randomized", "#6C8EBF"),
        ):
            axis.scatter(
                pivot["trained"],
                pivot[stage],
                alpha=0.75,
                label=stage,
                color=color,
            )
        axis.set(
            title=f"{year} parameter-randomization sanity check",
            xlabel="Trained-model occlusion importance",
            ylabel="Randomized-model occlusion importance",
        )
        axis.legend(frameon=False)
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "explanation_randomization_sanity.png", dpi=300)
    figure.savefig(output / "explanation_randomization_sanity.pdf")
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
        raise RuntimeError("CUDA PyTorch is required for this sanity check")
    device = torch.device("cuda")
    panel = pd.read_parquet(BENCHMARK / "stock_stress_labels.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["date", "security_id"]).reset_index(drop=True)
    panel["_row"] = np.arange(len(panel))
    security_count = int(panel["security_id"].max()) + 1
    output = Path(str(configuration["output_root"]))
    output.mkdir(parents=True, exist_ok=True)
    representative_seed = int(configuration["representative_model_seed"])
    randomization_seeds = [
        int(seed) for seed in configuration["randomization_seeds"]
    ]
    records: list[dict[str, object]] = []

    for fold in temporal_configuration["folds"]:
        fold_name = str(fold["fold"])
        year = int(fold["evaluation_year"])
        result_root = (
            Path(str(configuration["temporal_root"])) / fold_name
        )
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
        checkpoint = torch.load(
            artifact_root
            / f"temporal_tabular_seed{representative_seed}.pt",
            map_location=device,
            weights_only=False,
        )
        trained = make_model(
            int(checkpoint["hidden_dimension"]),
            float(checkpoint["dropout"]),
            device,
        )
        trained.load_state_dict(checkpoint["model_state_dict"])
        records.extend(
            occlusion_importance(
                trained,
                batches,
                values,
                device,
                security_count,
                fold_name,
                year,
                "trained",
                None,
            )
        )
        for seed in randomization_seeds:
            predictor_randomized = reset_modules(
                trained, ["predictor"], seed
            )
            records.extend(
                occlusion_importance(
                    predictor_randomized,
                    batches,
                    values,
                    device,
                    security_count,
                    fold_name,
                    year,
                    "predictor_randomized",
                    seed,
                )
            )
            recurrent_predictor_randomized = reset_modules(
                trained, ["recurrent", "predictor"], seed
            )
            records.extend(
                occlusion_importance(
                    recurrent_predictor_randomized,
                    batches,
                    values,
                    device,
                    security_count,
                    fold_name,
                    year,
                    "recurrent_predictor_randomized",
                    seed,
                )
            )
    importance = pd.DataFrame(records)
    comparisons = compare_with_trained(importance)
    importance.to_parquet(
        output / "randomization_occlusion_importance.parquet",
        index=False,
        compression="zstd",
    )
    comparisons.to_csv(
        output / "randomization_sanity_summary.csv", index=False
    )
    make_figure(importance, output)
    full = comparisons.loc[
        comparisons["stage"].eq("recurrent_predictor_randomized")
    ]
    summary = {
        "status": "complete",
        "representative_model_seed": representative_seed,
        "randomization_seeds": randomization_seeds,
        "median_full_randomization_rank_spearman": float(
            full["rank_spearman_with_trained"].median()
        ),
        "maximum_full_randomization_rank_spearman": float(
            full["rank_spearman_with_trained"].max()
        ),
        "median_full_randomization_top10_jaccard": float(
            full["top10_jaccard_with_trained"].median()
        ),
        "sanity_gate_pass": bool(
            full["rank_spearman_with_trained"].abs().median() < 0.50
            and full["top10_jaccard_with_trained"].median() < 0.50
        ),
        "gate_definition": (
            "Median absolute trained-vs-fully-randomized rank Spearman < "
            "0.50 and median top-10 Jaccard < 0.50."
        ),
    }
    (output / "explanation_sanity_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                **summary,
                "importance_measure": (
                    "Mean absolute raw-probability change after complete "
                    "historical replay with one standardized feature set "
                    "to its refit-development mean."
                ),
                "trained_outer_predictions_unchanged": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
