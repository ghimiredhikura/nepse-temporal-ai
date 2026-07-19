"""Load one NEPSE session into matched native and projected graph views.

The loader is deliberately framework-neutral: it returns NumPy arrays that can
be converted to PyTorch, PyG, DGL, or JAX tensors without changing the data
definition.  Pairwise views are derived from the same native hyperedges in
memory, avoiding duplicated graph files and representation drift.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GRAPH = ROOT / "data" / "processed" / "transaction_hypergraph"
BENCHMARK = GRAPH / "benchmark"
VALID_REPRESENTATIONS = {"top32", "full"}

FEATURES = {
    "log1p_trade_count": lambda x: np.log1p(x["trade_count"]),
    "log1p_quantity": lambda x: np.log1p(x["quantity"].fillna(0)),
    "log1p_value": lambda x: np.log1p(x["value"]),
    "log_vwap": lambda x: np.log(x["vwap"]),
    "log_rate_range": lambda x: np.log(
        x["max_rate"] / x["min_rate"].replace(0, np.nan)
    ),
    "missing_quantity_share": lambda x: (
        x["missing_quantity_trades"] / x["trade_count"]
    ),
    "missing_rate_share": lambda x: (
        x["missing_rate_trades"] / x["trade_count"]
    ),
}


@dataclass(frozen=True)
class NativeHypergraph:
    """A role-aware buyer-seller-security transaction hypergraph."""

    date: pd.Timestamp
    incidence_index: np.ndarray
    attributes: np.ndarray
    feature_names: tuple[str, ...]
    value: np.ndarray
    trade_count: np.ndarray
    retained_rank: np.ndarray
    same_broker: np.ndarray


@dataclass(frozen=True)
class PairwiseGraph:
    """A directed broker-to-broker projection aggregated across securities."""

    date: pd.Timestamp
    edge_index: np.ndarray
    attributes: np.ndarray
    feature_names: tuple[str, ...]


@dataclass(frozen=True)
class BrokerSecurityGraph:
    """A role-preserving broker-security projection.

    role is +1 for buyer participation and -1 for seller participation.  The
    two legs preserve gross value; therefore total leg value is exactly twice
    the native hyperedge value.
    """

    date: pd.Timestamp
    edge_index: np.ndarray
    role: np.ndarray
    attributes: np.ndarray
    feature_names: tuple[str, ...]


class TransactionGraphLoader:
    """Read graph snapshots from the compact exact or top-k representation."""

    def __init__(
        self,
        representation: str = "top32",
        *,
        clip_to_train_quantiles: bool = True,
    ) -> None:
        if representation not in VALID_REPRESENTATIONS:
            raise ValueError(
                f"representation must be one of {VALID_REPRESENTATIONS}"
            )
        folder = (
            "hyperedges_full"
            if representation == "full"
            else "hyperedges_top32"
        )
        self.representation = representation
        self.path = (GRAPH / folder / "year=*" / "*.parquet").as_posix()
        if not list((GRAPH / folder).glob("year=*/*.parquet")):
            raise FileNotFoundError(
                f"No {representation} hyperedges found under {GRAPH / folder}"
            )
        self.clip_to_train_quantiles = clip_to_train_quantiles
        scaling_name = (
            "hyperedge_train_scaling_full.csv"
            if representation == "full"
            else "hyperedge_train_scaling.csv"
        )
        scaling = pd.read_csv(BENCHMARK / scaling_name)
        self.scaling = scaling.set_index("feature")
        missing = set(FEATURES) - set(self.scaling.index)
        if missing:
            raise ValueError(f"Missing train scaling for: {sorted(missing)}")
        self.connection = duckdb.connect()
        self.connection.execute("PRAGMA threads=2")

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "TransactionGraphLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _read(self, date: str | pd.Timestamp) -> pd.DataFrame:
        timestamp = pd.Timestamp(date).normalize()
        frame = self.connection.execute(
            f"""
            SELECT * EXCLUDE (year)
            FROM read_parquet('{self.path}', hive_partitioning=true)
            WHERE date = ?
            ORDER BY security_id, value_rank
            """,
            [timestamp.date()],
        ).fetchdf()
        if frame.empty:
            raise KeyError(
                f"No {self.representation} hyperedges for "
                f"{timestamp.date().isoformat()}"
            )
        frame["date"] = pd.to_datetime(frame["date"])
        return frame

    def read_frame(self, date: str | pd.Timestamp) -> pd.DataFrame:
        """Return the canonical event frame for custom framework adapters."""
        return self._read(date)

    def _scaled_features(self, frame: pd.DataFrame) -> np.ndarray:
        columns: list[np.ndarray] = []
        for feature, transform in FEATURES.items():
            values = transform(frame).to_numpy(dtype="float64")
            parameters = self.scaling.loc[feature]
            if self.clip_to_train_quantiles:
                values = np.clip(
                    values,
                    float(parameters["train_p01"]),
                    float(parameters["train_p99"]),
                )
            standard_deviation = float(parameters["train_std"])
            if not np.isfinite(standard_deviation) or standard_deviation <= 0:
                standard_deviation = 1.0
            values = (
                values - float(parameters["train_mean"])
            ) / standard_deviation
            columns.append(np.nan_to_num(values, copy=False))
        return np.column_stack(columns).astype("float32", copy=False)

    def native(
        self, date: str | pd.Timestamp, frame: pd.DataFrame | None = None
    ) -> NativeHypergraph:
        frame = self._read(date) if frame is None else frame
        incidence = frame[
            ["buyer_id", "seller_id", "security_id"]
        ].to_numpy(dtype="int64").T
        return NativeHypergraph(
            date=pd.Timestamp(frame["date"].iloc[0]),
            incidence_index=incidence,
            attributes=self._scaled_features(frame),
            feature_names=tuple(FEATURES),
            value=frame["value"].to_numpy(dtype="float64"),
            trade_count=frame["trade_count"].to_numpy(dtype="int64"),
            retained_rank=frame["value_rank"].to_numpy(dtype="int64"),
            same_broker=frame["same_broker"].to_numpy(dtype="bool"),
        )

    def broker_pairwise(
        self, date: str | pd.Timestamp, frame: pd.DataFrame | None = None
    ) -> PairwiseGraph:
        frame = self._read(date) if frame is None else frame
        grouped = (
            frame.groupby(["buyer_id", "seller_id"], sort=True, observed=True)
            .agg(
                value=("value", "sum"),
                trade_count=("trade_count", "sum"),
                quantity=("quantity", "sum"),
                security_count=("security_id", "nunique"),
                hyperedge_count=("security_id", "size"),
            )
            .reset_index()
        )
        grouped["log1p_value"] = np.log1p(grouped["value"])
        grouped["log1p_trade_count"] = np.log1p(grouped["trade_count"])
        grouped["log1p_quantity"] = np.log1p(grouped["quantity"].fillna(0))
        names = (
            "log1p_value",
            "log1p_trade_count",
            "log1p_quantity",
            "security_count",
            "hyperedge_count",
        )
        return PairwiseGraph(
            date=pd.Timestamp(frame["date"].iloc[0]),
            edge_index=grouped[
                ["buyer_id", "seller_id"]
            ].to_numpy(dtype="int64").T,
            attributes=grouped[list(names)].to_numpy(dtype="float32"),
            feature_names=names,
        )

    def broker_security(
        self, date: str | pd.Timestamp, frame: pd.DataFrame | None = None
    ) -> BrokerSecurityGraph:
        frame = self._read(date) if frame is None else frame
        common = [
            "security_id",
            "value",
            "trade_count",
            "quantity",
        ]
        buy = frame[["buyer_id", *common]].rename(
            columns={"buyer_id": "broker_id"}
        )
        buy["role"] = 1
        sell = frame[["seller_id", *common]].rename(
            columns={"seller_id": "broker_id"}
        )
        sell["role"] = -1
        legs = pd.concat([buy, sell], ignore_index=True)
        grouped = (
            legs.groupby(
                ["broker_id", "security_id", "role"],
                sort=True,
                observed=True,
            )
            .agg(
                value=("value", "sum"),
                trade_count=("trade_count", "sum"),
                quantity=("quantity", "sum"),
                hyperedge_count=("security_id", "size"),
            )
            .reset_index()
        )
        grouped["log1p_value"] = np.log1p(grouped["value"])
        grouped["log1p_trade_count"] = np.log1p(grouped["trade_count"])
        grouped["log1p_quantity"] = np.log1p(grouped["quantity"].fillna(0))
        names = (
            "log1p_value",
            "log1p_trade_count",
            "log1p_quantity",
            "hyperedge_count",
        )
        return BrokerSecurityGraph(
            date=pd.Timestamp(frame["date"].iloc[0]),
            edge_index=grouped[
                ["broker_id", "security_id"]
            ].to_numpy(dtype="int64").T,
            role=grouped["role"].to_numpy(dtype="int8"),
            attributes=grouped[list(names)].to_numpy(dtype="float32"),
            feature_names=names,
        )

    def matched_views(
        self, date: str | pd.Timestamp
    ) -> tuple[NativeHypergraph, PairwiseGraph, BrokerSecurityGraph]:
        frame = self._read(date)
        return (
            self.native(date, frame),
            self.broker_pairwise(date, frame),
            self.broker_security(date, frame),
        )


def audit_snapshot(
    representation: str, date: str | pd.Timestamp
) -> dict[str, object]:
    with TransactionGraphLoader(representation) as loader:
        native, pairwise, broker_security = loader.matched_views(date)
    pair_value = np.expm1(
        pairwise.attributes[
            :, pairwise.feature_names.index("log1p_value")
        ].astype("float64")
    ).sum()
    leg_value = np.expm1(
        broker_security.attributes[
            :, broker_security.feature_names.index("log1p_value")
        ].astype("float64")
    ).sum()
    native_value = native.value.sum()
    tolerance = max(1.0, native_value * 1e-6)
    checks = {
        "pairwise_value_equals_native": bool(
            abs(pair_value - native_value) <= tolerance
        ),
        "broker_security_gross_value_equals_twice_native": bool(
            abs(leg_value - 2 * native_value) <= 2 * tolerance
        ),
        "native_attributes_finite": bool(
            np.isfinite(native.attributes).all()
        ),
    }
    return {
        "representation": representation,
        "date": native.date.date().isoformat(),
        "native_hyperedges": native.incidence_index.shape[1],
        "broker_pair_edges": pairwise.edge_index.shape[1],
        "broker_security_role_edges": broker_security.edge_index.shape[1],
        "native_value": float(native_value),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=VALID_REPRESENTATIONS)
    parser.add_argument("--date", required=True)
    arguments = parser.parse_args()
    print(audit_snapshot(arguments.representation, arguments.date))


if __name__ == "__main__":
    main()
