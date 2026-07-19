"""Prepare labels, snapshot indexes, and train-only scaling for graph models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "data" / "processed" / "transaction_hypergraph"
BENCHMARK = GRAPH / "benchmark"
LABEL_SOURCE = ROOT / "research" / "ai_feasibility" / "stock_stress_panel.parquet"
TOP_K = 32
HYPEREDGES = (
    GRAPH / f"hyperedges_top{TOP_K}" / "year=*" / "*.parquet"
).as_posix()
REGISTRY = GRAPH / "nodes" / "security_registry.parquet"
TRAIN_END = "2022-12-31"

BASE_FEATURES = [
    "range_t",
    "vwap_return_t",
    "abs_vwap_return_t",
    "log_turnover_t",
    "log_volume_t",
    "log_trade_count_t",
    "range_mean20",
    "range_std20",
    "abs_return_mean20",
    "abs_return_std20",
    "turnover_z20",
    "illiquidity_t",
]

STOCK_BROKER_FEATURES = [
    "broker_count",
    "buyer_hhi",
    "seller_hhi",
    "gross_flow_hhi",
    "top_broker_flow_share",
    "broker_flow_entropy",
    "aggregate_broker_imbalance",
    "same_broker_value_share",
    "broker_count_z20",
    "gross_flow_hhi_z20",
    "top_broker_flow_share_z20",
    "aggregate_broker_imbalance_z20",
]

MARKET_STATE_FEATURES = [
    "mkt_return",
    "mkt_abs_return",
    "mkt_log_turnover",
    "mkt_log_trades",
]

MARKET_NETWORK_FEATURES = [
    "mkt_gross_flow_hhi",
    "mkt_top_broker_flow_share",
    "mkt_broker_flow_entropy",
    "mkt_aggregate_broker_imbalance",
    "mkt_mean_broker_imbalance",
    "mkt_edge_value_hhi",
    "mkt_top_edge_value_share",
    "mkt_edge_value_entropy",
    "mkt_weighted_reciprocity",
    "mkt_directed_density",
    "mkt_same_broker_value_share",
]

GLOBAL_FEATURES = [
    "nifty_ret_l1",
    "sp500_ret_l1",
    "vix_change_l1",
    "brent_ret_l1",
    "gold_ret_l1",
    "usd_inr_ret_l1",
]


def artifact_name(stem: str, extension: str, representation: str) -> str:
    """Keep the original top32 names while suffixing lossless artifacts."""
    suffix = "" if representation == "top32" else "_full"
    return f"{stem}{suffix}.{extension}"


def write_labels() -> pd.DataFrame:
    labels = pd.read_parquet(LABEL_SOURCE)
    registry = pd.read_parquet(REGISTRY)[
        ["security_id", "symbol", "matched_current_master"]
    ]
    labels["date"] = pd.to_datetime(labels["date"]).astype("datetime64[ns]")
    labels["next_date"] = pd.to_datetime(labels["next_date"]).astype(
        "datetime64[ns]"
    )
    labels = labels.merge(
        registry,
        on="symbol",
        how="left",
        validate="many_to_one",
    )
    if labels["security_id"].isna().any():
        missing = labels.loc[labels["security_id"].isna(), "symbol"].unique()
        raise ValueError(f"Labels missing registry IDs: {missing[:10]}")
    labels["security_id"] = labels["security_id"].astype("int32")
    labels = labels.sort_values(["date", "security_id"]).reset_index(drop=True)
    labels.to_parquet(
        BENCHMARK / "stock_stress_labels.parquet",
        index=False,
        compression="zstd",
    )
    return labels


def build_snapshot_index(
    connection: duckdb.DuckDBPyConnection, representation: str
) -> pd.DataFrame:
    snapshot = connection.execute(
        f"""
        SELECT
            date,
            count(*) AS hyperedge_count,
            count(DISTINCT buyer_id) AS buyer_count,
            count(DISTINCT seller_id) AS seller_count,
            count(DISTINCT security_id) AS security_count,
            sum(trade_count) AS represented_trades,
            sum(value) AS total_value,
            sum(quantity) AS total_quantity,
            sum(CASE WHEN same_broker THEN value ELSE 0 END) / sum(value)
                AS same_broker_value_share,
            sum(missing_quantity_trades)::DOUBLE / sum(trade_count)
                AS missing_quantity_trade_share,
            sum(missing_rate_trades)::DOUBLE / sum(trade_count)
                AS missing_rate_trade_share,
            quantile_cont(value, 0.50) AS median_hyperedge_value,
            quantile_cont(value, 0.90) AS p90_hyperedge_value,
            quantile_cont(value, 0.99) AS p99_hyperedge_value
        FROM read_parquet('{HYPEREDGES}', hive_partitioning=true)
        GROUP BY date
        ORDER BY date
        """
    ).fetchdf()
    snapshot["date"] = pd.to_datetime(snapshot["date"])
    snapshot["snapshot_id"] = range(len(snapshot))
    snapshot.to_parquet(
        BENCHMARK
        / artifact_name("snapshot_index", "parquet", representation),
        index=False,
        compression="zstd",
    )
    return snapshot


def train_hyperedge_scaling(
    connection: duckdb.DuckDBPyConnection,
    representation: str,
) -> pd.DataFrame:
    # Log transforms are defined here and must be reproduced exactly by the
    # dataset loader.  Parameters are fitted only on pre-2023 hyperedges.
    expressions = {
        "log1p_trade_count": "ln(1 + trade_count)",
        "log1p_quantity": "ln(1 + coalesce(quantity, 0))",
        "log1p_value": "ln(1 + value)",
        "log_vwap": "ln(vwap)",
        "log_rate_range": "ln(max_rate / nullif(min_rate, 0))",
        "missing_quantity_share": (
            "missing_quantity_trades::DOUBLE / trade_count"
        ),
        "missing_rate_share": "missing_rate_trades::DOUBLE / trade_count",
    }
    rows: list[dict[str, object]] = []
    for feature, expression in expressions.items():
        result = connection.execute(
            f"""
            WITH transformed AS (
                SELECT {expression} AS x
                FROM read_parquet('{HYPEREDGES}', hive_partitioning=true)
                WHERE date <= DATE '{TRAIN_END}'
            )
            SELECT
                count(x), avg(x), stddev_pop(x),
                quantile_cont(x, 0.01), quantile_cont(x, 0.50),
                quantile_cont(x, 0.99), min(x), max(x)
            FROM transformed
            WHERE isfinite(x)
            """
        ).fetchone()
        rows.append(
            {
                "feature": feature,
                "expression": expression,
                "train_count": result[0],
                "train_mean": result[1],
                "train_std": result[2],
                "train_p01": result[3],
                "train_median": result[4],
                "train_p99": result[5],
                "train_min": result[6],
                "train_max": result[7],
            }
        )
    scaling = pd.DataFrame(rows)
    scaling.to_csv(
        BENCHMARK
        / artifact_name(
            "hyperedge_train_scaling", "csv", representation
        ),
        index=False,
    )
    return scaling


def validate_coverage(
    connection: duckdb.DuckDBPyConnection,
    labels: pd.DataFrame,
    representation: str,
) -> dict[str, object]:
    connection.register(
        "benchmark_labels",
        labels[["date", "security_id", "split", "next_range_stress"]],
    )
    coverage = connection.execute(
        f"""
        WITH observed AS (
            SELECT DISTINCT date, security_id
            FROM read_parquet('{HYPEREDGES}', hive_partitioning=true)
        )
        SELECT
            l.split,
            count(*) AS labels,
            sum(o.security_id IS NOT NULL) AS labels_with_current_hyperedges,
            avg((o.security_id IS NOT NULL)::INTEGER) AS coverage
        FROM benchmark_labels l
        LEFT JOIN observed o USING (date, security_id)
        GROUP BY l.split
        ORDER BY l.split
        """
    ).fetchdf()
    coverage.to_csv(
        BENCHMARK
        / artifact_name(
            "label_hyperedge_coverage", "csv", representation
        ),
        index=False,
    )
    return {
        row["split"]: {
            "labels": int(row["labels"]),
            "labels_with_current_hyperedges": int(
                row["labels_with_current_hyperedges"]
            ),
            "coverage": float(row["coverage"]),
        }
        for _, row in coverage.iterrows()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representation",
        choices=("top32", "full"),
        default="top32",
    )
    arguments = parser.parse_args()
    representation = arguments.representation
    global HYPEREDGES
    folder = (
        "hyperedges_full"
        if representation == "full"
        else f"hyperedges_top{TOP_K}"
    )
    HYPEREDGES = (GRAPH / folder / "year=*" / "*.parquet").as_posix()
    if not list((GRAPH / folder).glob("year=*/*.parquet")):
        raise FileNotFoundError(f"No hyperedges found under {GRAPH / folder}")

    BENCHMARK.mkdir(parents=True, exist_ok=True)
    labels = write_labels()

    connection = duckdb.connect()
    connection.execute("PRAGMA threads=4")
    connection.execute("PRAGMA memory_limit='4GB'")
    snapshot = build_snapshot_index(connection, representation)
    scaling = train_hyperedge_scaling(connection, representation)
    coverage = validate_coverage(connection, labels, representation)

    split_summary = (
        labels.groupby("split", as_index=False)
        .agg(
            rows=("next_range_stress", "size"),
            sessions=("date", "nunique"),
            securities=("security_id", "nunique"),
            event_rate=("next_range_stress", "mean"),
            first_date=("date", "min"),
            last_date=("date", "max"),
        )
    )
    split_summary.to_csv(BENCHMARK / "split_summary.csv", index=False)

    manifest = {
        "representation": representation,
        "target": (
            "next observed security-session range exceeds the rolling "
            "60-session 90th percentile known at t"
        ),
        "train_end": TRAIN_END,
        "label_rows": len(labels),
        "snapshot_rows": len(snapshot),
        "snapshot_first_date": snapshot["date"].min().date().isoformat(),
        "snapshot_last_date": snapshot["date"].max().date().isoformat(),
        "feature_groups": {
            "base_security_state": BASE_FEATURES,
            "stock_broker_summaries": STOCK_BROKER_FEATURES,
            "market_state": MARKET_STATE_FEATURES,
            "market_network_summaries": MARKET_NETWORK_FEATURES,
            "international_lagged": GLOBAL_FEATURES,
        },
        "hyperedge_attributes": scaling["feature"].tolist()
        + ["same_broker", "value_rank"],
        "retained_top_k_per_security_session": (
            TOP_K if representation == "top32" else None
        ),
        "pairwise_projection_policy": (
            "Generate broker-broker and signed broker-security projections "
            "from the same retained hyperedges inside each data loader."
        ),
        "coverage": coverage,
        "evaluation_status": (
            "The benchmark split is retained for historical comparison; "
            "rolling-origin outer evaluations provide the final evidence."
        ),
        "leakage_rules": [
            "A prediction for session t+1 may use hyperedges only through t.",
            "Scaling parameters are fitted only within each training window.",
            "Outer evaluation years are not used for model selection.",
            "No future neighbor, full-sample centrality, or future listing state.",
            "Final evidence uses chronological rolling-origin evaluation.",
        ],
    }
    (
        BENCHMARK
        / artifact_name("benchmark_manifest", "json", representation)
    ).write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(split_summary.to_string(index=False))
    print(json.dumps(coverage, indent=2))
    print(snapshot.describe().to_string())


if __name__ == "__main__":
    main()
