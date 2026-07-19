# Processed sample data

`processed/transaction_hypergraph/benchmark/stock_stress_labels.parquet` is an
illustrative subset of the final modelling panel. It contains 27,476
stock–session rows for 24 securities from 2021 through 2025 and retains all 56
analysis columns.

The securities were selected for high session coverage in every sample year.
This subset is suitable for inspecting the schema, checking code, and running
small examples. It is **not statistically representative** and cannot reproduce
the paper’s estimates. No raw floorsheets or personal investor data are
included.

`sample_manifest.json` records the selection rule, dimensions, security IDs,
and SHA-256 checksum. `sample_overview.csv` gives yearly counts and event rates.
