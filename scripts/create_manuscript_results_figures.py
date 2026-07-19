"""Create journal-ready multipanel figures from locked NEPSE results.

The script reads only completed outer-evaluation artifacts. It does not refit,
recalibrate, or otherwise alter any forecast.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import shap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "paper"
OUTPUT = ROOT / "results" / "figures"
BENCHMARK = (
    ROOT
    / "data"
    / "processed"
    / "transaction_hypergraph"
    / "benchmark"
    / "stock_stress_labels.parquet"
)
BASELINE_MODELS = ROOT / "models" / "surveillance_baselines_2024_2025"
SHAP_SAMPLE_SIZE = 10_000
SHAP_RANDOM_SEED = 20_260_718

TEMPORAL = "temporal_tabular"
TREE = "lgbm_state"
COLORS = {TEMPORAL: "#0072B2", TREE: "#D55E00"}
LABELS = {TEMPORAL: "Temporal GRU", TREE: "Full-state LightGBM"}

FEATURE_LABELS = {
    "abs_return_mean20": "Mean absolute return (20)",
    "abs_return_std20": "Absolute-return SD (20)",
    "abs_vwap_return_t": "Absolute VWAP return",
    "aggregate_broker_imbalance": "Broker imbalance",
    "aggregate_broker_imbalance_z20": "Broker-imbalance deviation (20)",
    "broker_count": "Broker count",
    "broker_count_z20": "Broker-count deviation (20)",
    "broker_flow_entropy": "Broker-flow entropy",
    "buyer_hhi": "Buyer concentration",
    "gross_flow_hhi": "Gross-flow concentration",
    "gross_flow_hhi_z20": "Concentration deviation (20)",
    "illiquidity_t": "Log illiquidity",
    "log_trade_count_t": "Log stock trades",
    "log_turnover_t": "Log stock turnover",
    "log_volume_t": "Log stock volume",
    "mkt_abs_return": "Market absolute return",
    "mkt_log_trades": "Log market trades",
    "mkt_log_turnover": "Log market turnover",
    "mkt_return": "Market return",
    "range_mean20": "Mean range (20)",
    "range_std20": "Range SD (20)",
    "range_t": "Current range",
    "same_broker_value_share": "Same-broker value share",
    "seller_hhi": "Seller concentration",
    "top_broker_flow_share": "Top-broker flow share",
    "top_broker_flow_share_z20": "Top-share deviation (20)",
    "turnover_z20": "Turnover deviation (20)",
    "vwap_return_t": "VWAP return",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman No9 L",
                "Liberation Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.7,
            "savefig.dpi": 300,
        }
    )


def panel_label(
    axis: plt.Axes,
    label: str,
    y_position: float = -0.24,
) -> None:
    axis.text(
        0.5,
        y_position,
        label,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontweight="normal",
        clip_on=False,
    )


def save(
    figure: plt.Figure,
    stem: str,
    pad_inches: float = 0.1,
) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=pad_inches,
        metadata={"Creator": "NEPSE manuscript figure pipeline"},
    )
    figure.savefig(
        OUTPUT / f"{stem}.png",
        bbox_inches="tight",
        pad_inches=pad_inches,
        facecolor="white",
    )
    plt.close(figure)


def reliability_figure() -> None:
    path = (
        RESULTS
        / "temporal_surveillance_analysis"
        / "reliability_bins.csv"
    )
    frame = pd.read_csv(path)
    frame = frame.loc[frame["model"].isin([TEMPORAL, TREE])]

    figure, axes = plt.subplots(
        1, 2, figsize=(7.25, 3.5), sharex=True, sharey=True
    )
    for axis, year, letter in zip(axes, [2024, 2025], ["a", "b"]):
        axis.plot(
            [0, 0.42],
            [0, 0.42],
            color="#777777",
            linestyle="--",
            linewidth=1,
            label="Perfect reliability",
            zorder=1,
        )
        for model, marker in [(TEMPORAL, "o"), (TREE, "s")]:
            sample = frame.loc[
                frame["evaluation_year"].eq(year)
                & frame["model"].eq(model)
            ].sort_values("bin")
            axis.plot(
                sample["predicted"],
                sample["observed"],
                color=COLORS[model],
                marker=marker,
                markersize=4,
                label=LABELS[model],
                zorder=2,
            )
        axis.set(
            xlim=(0, 0.35),
            ylim=(0, 0.42),
            xlabel="Mean predicted probability",
        )
        axis.set_title(str(year), pad=5)
        axis.grid(alpha=0.18, linewidth=0.6)
        panel_label(axis, f"({letter})", y_position=-0.25)
    axes[0].set_ylabel("Observed event frequency")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        labels,
        loc="upper left",
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
        borderaxespad=0.6,
    )
    figure.subplots_adjust(bottom=0.22, wspace=0.12)
    save(figure, "fig_results_reliability")


def operational_figure() -> None:
    alert = pd.read_csv(
        RESULTS
        / "surveillance_journal_robustness"
        / "daily_alert_metrics.csv"
    )
    decision = pd.read_csv(
        RESULTS
        / "surveillance_journal_robustness"
        / "decision_curve.csv"
    )
    alert = alert.loc[alert["model"].isin([TEMPORAL, TREE])]
    decision = decision.loc[decision["model"].isin([TEMPORAL, TREE])]

    figure, axes = plt.subplots(
        2, 2, figsize=(7.25, 6.0), sharex="row", sharey="row"
    )
    for column, year in enumerate([2024, 2025]):
        for model, marker in [(TEMPORAL, "o"), (TREE, "s")]:
            sample = alert.loc[
                alert["evaluation_year"].eq(year)
                & alert["model"].eq(model)
            ].sort_values("alert_budget")
            axes[0, column].plot(
                100 * sample["alert_budget"],
                sample["precision"],
                color=COLORS[model],
                marker=marker,
                markersize=4,
                label=LABELS[model],
            )
            curve = decision.loc[
                decision["evaluation_year"].eq(year)
                & decision["model"].eq(model)
            ].sort_values("threshold")
            axes[1, column].plot(
                curve["threshold"],
                curve["standardized_net_benefit"],
                color=COLORS[model],
                marker=marker,
                markersize=3,
                label=LABELS[model],
            )
        axes[0, column].set_xlabel("Daily review budget (%)")
        axes[1, column].set_xlabel("Calibrated probability threshold")
        for row in range(2):
            axes[row, column].set_title(str(year), pad=4)
            axes[row, column].grid(alpha=0.18, linewidth=0.6)
    axes[0, 0].set_ylabel("Alert precision")
    axes[1, 0].set_ylabel("Standardized net benefit")
    panel_label(axes[0, 0], "(a)", y_position=-0.29)
    panel_label(axes[0, 1], "(b)", y_position=-0.29)
    panel_label(axes[1, 0], "(c)", y_position=-0.29)
    panel_label(axes[1, 1], "(d)", y_position=-0.29)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.subplots_adjust(
        bottom=0.17, hspace=0.62, wspace=0.14
    )
    save(figure, "fig_results_operational")


def explanation_figure() -> None:
    explanation_root = RESULTS / "surveillance_explainability"
    summary = pd.read_csv(
        explanation_root / "temporal_occlusion_summary.csv"
    )
    features = summary.loc[summary["ablation_type"].eq("feature")].copy()
    group = summary.loc[summary["ablation_type"].eq("group")].copy()

    average = (
        features.groupby("ablation", as_index=False)[
            "probability_change_mean"
        ]
        .mean()
        .nlargest(12, "probability_change_mean")
    )
    selected = average["ablation"].tolist()
    feature_panel = (
        features.loc[features["ablation"].isin(selected)]
        .pivot(
            index="ablation",
            columns="evaluation_year",
            values="probability_change_mean",
        )
        .loc[selected]
        .sort_values(2024)
    )

    group_labels = {
        "group_price_liquidity": "Price/liquidity",
        "group_market_state": "Market state",
        "group_broker_state": "Broker state",
    }
    group_panel = group.pivot(
        index="ablation",
        columns="evaluation_year",
        values="pr_auc_drop_mean",
    ).loc[list(group_labels)]

    seed = pd.read_csv(
        explanation_root / "temporal_seed_rank_stability.csv"
    )
    randomization = pd.read_csv(
        RESULTS
        / "surveillance_explanation_sanity"
        / "randomization_sanity_summary.csv"
    )
    full_randomization = randomization.loc[
        randomization["stage"].eq("recurrent_predictor_randomized"),
        "rank_spearman_with_trained",
    ].to_numpy()

    figure = plt.figure(figsize=(7.25, 6.5))
    # Explicit coordinates guarantee that the left panel and the combined
    # right column have identical top and bottom boundaries after rendering.
    ax_feature = figure.add_axes([0.24, 0.10, 0.39, 0.82])
    ax_group = figure.add_axes([0.70, 0.59, 0.27, 0.33])
    ax_sanity = figure.add_axes([0.70, 0.10, 0.27, 0.33])

    positions = np.arange(len(feature_panel))
    ax_feature.scatter(
        feature_panel[2024],
        positions - 0.13,
        color="#0072B2",
        marker="o",
        s=28,
        label="2024",
    )
    ax_feature.scatter(
        feature_panel[2025],
        positions + 0.13,
        color="#009E73",
        marker="s",
        s=26,
        label="2025",
    )
    for position, row in enumerate(feature_panel.itertuples()):
        ax_feature.plot(
            [getattr(row, "_1"), getattr(row, "_2")],
            [position - 0.13, position + 0.13],
            color="#B8B8B8",
            linewidth=0.7,
            zorder=0,
        )
    ax_feature.set_yticks(positions)
    ax_feature.set_yticklabels(
        [FEATURE_LABELS.get(value, value) for value in feature_panel.index]
    )
    ax_feature.set_xlabel("Mean absolute probability change")
    ax_feature.set_title("Temporal path sensitivity", pad=5)
    ax_feature.grid(axis="x", alpha=0.18, linewidth=0.6)
    ax_feature.legend(frameon=False, loc="lower right")

    x = np.arange(len(group_panel))
    width = 0.36
    ax_group.bar(
        x - width / 2,
        group_panel[2024],
        width,
        color="#0072B2",
        label="2024",
    )
    ax_group.bar(
        x + width / 2,
        group_panel[2025],
        width,
        color="#009E73",
        label="2025",
    )
    ax_group.axhline(0, color="#777777", linewidth=0.8)
    ax_group.set_xticks(x)
    ax_group.set_xticklabels(
        [group_labels[value] for value in group_panel.index],
        rotation=20,
        ha="right",
    )
    ax_group.set_ylabel("PR-AUC loss after occlusion")
    ax_group.set_title("Feature-group occlusion", pad=5)
    ax_group.grid(axis="y", alpha=0.18, linewidth=0.6)

    generator = np.random.default_rng(20260718)
    categories = [
        ("Cross-year", np.asarray([0.9288451012588943])),
        ("Across seeds", seed["rank_spearman"].to_numpy()),
        ("Randomized", full_randomization),
    ]
    sanity_colors = ["#0072B2", "#56B4E9", "#D55E00"]
    for position, ((_, values), color) in enumerate(
        zip(categories, sanity_colors)
    ):
        jitter = generator.uniform(-0.09, 0.09, size=len(values))
        ax_sanity.scatter(
            np.full(len(values), position) + jitter,
            values,
            color=color,
            s=25,
            alpha=0.86,
            edgecolor="white",
            linewidth=0.4,
        )
        ax_sanity.plot(
            [position - 0.16, position + 0.16],
            [np.median(values), np.median(values)],
            color="black",
            linewidth=1.2,
        )
    ax_sanity.axhline(
        0.5, color="#777777", linestyle="--", linewidth=0.9
    )
    ax_sanity.set_xticks(range(len(categories)))
    ax_sanity.set_xticklabels(
        [item[0] for item in categories], rotation=18, ha="right"
    )
    ax_sanity.set_ylim(-0.02, 1.02)
    ax_sanity.set_ylabel("Spearman rank correlation")
    ax_sanity.set_title("Explanation stability", pad=5)
    ax_sanity.grid(axis="y", alpha=0.18, linewidth=0.6)
    figure.text(0.435, 0.045, "(a)", ha="center", va="top")
    figure.text(0.835, 0.505, "(b)", ha="center", va="top")
    figure.text(0.835, 0.045, "(c)", ha="center", va="top")
    save(figure, "fig_results_explainability")


def tree_shap_beeswarm_figure() -> None:
    """Plot directional TreeSHAP distributions for the locked tree benchmark."""
    panel = pd.read_parquet(BENCHMARK)
    panel["date"] = pd.to_datetime(panel["date"])

    years = [2024, 2025]
    fold_names = {2024: "eval_2024", 2025: "eval_2025"}
    shap_by_year: dict[int, np.ndarray] = {}
    inputs_by_year: dict[int, pd.DataFrame] = {}
    feature_names: list[str] | None = None

    for year in years:
        sample = (
            panel.loc[panel["date"].dt.year.eq(year)]
            .sample(
                n=min(
                    SHAP_SAMPLE_SIZE,
                    int(panel["date"].dt.year.eq(year).sum()),
                ),
                random_state=SHAP_RANDOM_SEED + year,
            )
            .sort_index()
        )
        fold_root = BASELINE_MODELS / fold_names[year]
        model_paths = sorted(fold_root.glob("lgbm_state_seed*.joblib"))
        if not model_paths:
            raise FileNotFoundError(
                f"No fitted LightGBM artifacts found in {fold_root}"
            )

        seed_values: list[np.ndarray] = []
        for model_path in model_paths:
            payload = joblib.load(model_path)
            current_features = list(payload["features"])
            if feature_names is None:
                feature_names = current_features
            elif feature_names != current_features:
                raise ValueError(
                    "LightGBM feature order differs across locked artifacts"
                )
            inputs = sample[current_features]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                values = shap.TreeExplainer(
                    payload["model"]
                ).shap_values(inputs, check_additivity=False)
            if isinstance(values, list):
                values = values[-1]
            seed_values.append(np.asarray(values, dtype="float64"))

        shap_by_year[year] = np.mean(
            np.stack(seed_values, axis=0),
            axis=0,
        )
        inputs_by_year[year] = sample[feature_names].copy()

    if feature_names is None:
        raise RuntimeError("TreeSHAP feature names were not recovered")

    pooled_importance = np.mean(
        [
            np.abs(shap_by_year[year]).mean(axis=0)
            for year in years
        ],
        axis=0,
    )
    selected = np.argsort(pooled_importance)[::-1][:10]
    selected_labels = [
        FEATURE_LABELS.get(feature_names[index], feature_names[index])
        for index in selected
    ]

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.25, 4.75),
        sharex=True,
    )
    for axis, year, letter in zip(axes, years, ["a", "b"]):
        explanation = shap.Explanation(
            values=shap_by_year[year][:, selected],
            data=inputs_by_year[year].iloc[:, selected].to_numpy(),
            feature_names=selected_labels,
        )
        shap.plots.beeswarm(
            explanation,
            max_display=len(selected),
            order=np.arange(len(selected)),
            color=shap.plots.colors.red_blue,
            color_bar=False,
            alpha=0.72,
            ax=axis,
            show=False,
            plot_size=None,
            s=9,
            group_remaining_features=False,
        )
        for collection in axis.collections:
            collection.set_rasterized(True)
        axis.axvline(0, color="#777777", linewidth=0.7, zorder=0)
        axis.set_xlabel(
            "SHAP value (log-odds contribution)",
            fontsize=9,
        )
        axis.tick_params(axis="y", labelsize=8.2, length=0, pad=-3)
        axis.tick_params(axis="x", labelsize=8)
        panel_label(
            axis,
            f"({letter}) Outer evaluation: {year}",
            y_position=-0.145,
        )

    color_map = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmin=0, vmax=1),
        cmap=shap.plots.colors.red_blue,
    )
    color_bar = figure.colorbar(
        color_map,
        cax=figure.add_axes([0.34, 0.945, 0.32, 0.018]),
        orientation="horizontal",
    )
    color_bar.set_ticks([0, 1])
    color_bar.set_ticklabels(
        ["Low feature value", "High feature value"]
    )
    color_bar.outline.set_visible(False)
    color_bar.ax.tick_params(length=0, labelsize=8.2, pad=2)
    figure.subplots_adjust(
        left=0.18,
        right=0.995,
        bottom=0.17,
        top=0.88,
        wspace=0.64,
    )
    save(figure, "fig_results_treeshap", pad_inches=0)


def main() -> None:
    style()
    reliability_figure()
    operational_figure()
    explanation_figure()
    tree_shap_beeswarm_figure()
    print(f"Saved manuscript result figures to {OUTPUT}")


if __name__ == "__main__":
    main()
