# Data availability

## Research input

The frozen v1.0 results were generated from a private standardized product-level Chinese commodity-futures dataset covering 44 commodities from 2018 through 2025. The research instructor supplied the dataset and constructed the continuous futures price series from market data obtained through the instructor's CTP interface using a custom dominant-contract stitching process.

Permission to publish the research results is separate from redistribution rights for the underlying market data. The full historical input file, `warehouse_receipts_standardized.csv`, is therefore intentionally **not redistributed** in this public repository.

The private input used for the frozen run is described by the following immutable provenance metadata:

| Property | Value |
|---|---|
| Rows | 69,108 |
| Commodities | 44 |
| Date range | 2018-01-02 to 2025-12-31 |
| SHA256 | `b448dc7daae786eca901572f1e4bb36ac511797b88a2e2f0af689652d9fe5256` |
| Columns | `id`, `datetime`, `close`, `on_warrant` |

Required schema:

| Column | Meaning |
|---|---|
| `id` | Commodity symbol |
| `datetime` | Observation date in YYYY-MM-DD form |
| `close` | Research close from a continuous futures construction |
| `on_warrant` | Standardized registered warehouse-receipt quantity |

Dominant-contract switching can differ by roughly one or two trading days across providers. A licensed equivalent dataset can run through the same code after mapping to this schema, but it should not be expected to reproduce the frozen artifacts byte-for-byte unless its contract stitching and data construction match the original research input.

During the provenance audit, sampled warehouse-receipt observations were reconciled against exchange-published registered-warrant totals. This check supports the economic interpretation of the field but does not establish that every observation was scraped directly from an exchange.

## Public repository data

`symbol_mapping.csv` is a clean UTF-8 symbol/name/exchange mapping containing only the research-relevant fields. The source workbook is excluded to avoid publishing Office metadata and unrelated workbook structure.

Synthetic observations are generated in `tests/conftest.py` solely for tests and CI. They are not sampled from the private historical dataset, are not empirical evidence, and are never substituted into the formal backtest.

No additional example CSV is included because the existing synthetic fixtures already demonstrate the required schema and causal wiring without duplicating test data.

## Research evidence

The files under `results/` are frozen artifacts from the original private-data research run. They include formal daily returns, metrics, NAV, strict-training selection evidence and robustness diagnostics. Their hashes and generating scripts are recorded in `results/artifact_manifest.json`.

See `manifest.json` for the private input fingerprint and public mapping metadata.
