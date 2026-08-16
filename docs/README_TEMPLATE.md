# Commodity Warehouse Receipt Signals

A reproducible cross-sectional commodity-futures study of whether warehouse-receipt scarcity and shocks carry return information after causal execution and conservative cost stress.

> **Research question.** Can standardized warehouse receipts—an observable proxy for deliverable inventory tightness or abundance—support a diversified long/short futures signal?
>
> **Economic logic.** Falling or unusually scarce registered inventory can indicate tighter deliverable supply; rising or abundant inventory can indicate looser conditions. The strategy combines a slow scarcity sleeve with a faster receipt-shock sleeve while keeping price data out of the signal itself.
>
> **Main limitation.** This is a historical backtest on standardized close/receipt data without a full point-in-time timestamp audit; rolling is represented by a monthly turnover proxy rather than contract-level execution.

**Data availability.** Strategy code, tests, configuration, methodology, robustness evidence and frozen research outputs are public. The full historical market-data input is not redistributed because underlying market-data redistribution rights are separate from permission to publish the research results.

## Formal results

These are frozen empirical results generated from the private real research dataset—not from synthetic fixtures. The headline convention charges **5bp on strategy turnover plus a monthly roll-turnover proxy**.

| Period | Annual return | Annual volatility | Sharpe | Max drawdown | Final NAV |
|---|---:|---:|---:|---:|---:|
| Full sample ({{FULL_START}}–{{FULL_END}}) | {{FULL_RETURN}} | {{FULL_VOL}} | **{{FULL_SHARPE}}** | {{FULL_MDD}} | {{FULL_NAV}} |
| Frozen validation / out-of-sample relative to selection ({{VAL_START}}–{{VAL_END}}) | {{VAL_RETURN}} | {{VAL_VOL}} | **{{VAL_SHARPE}}** | {{VAL_MDD}} | {{VAL_NAV}} |

![Formal 55/45 strategy NAV](results/formal_v1/final_dual_combo_nav.png)

The full-sample row includes the 2018–2021 in-sample selection interval; only 2022–2025 is frozen validation relative to the documented parameter-selection process.

The v1.0 contract is explicit: **lag-1 execution**, **2018–2021 strict-endpoint training**, **55% slow V4 / 45% diagnostic**, and no validation-based re-selection. The 55/45 mix is selected on gross training Sharpe; the later **5bp + roll-proxy weight plateau** is post-freeze robustness evidence.

## Robustness at a glance

- 55/45 is training rank **{{PLATEAU_RANK}}** under the official-cost plateau; neighboring 50/50, 55/45 and 60/40 validation Sharpes are {{W50_VAL}}, {{W55_VAL}} and {{W60_VAL}}.
- All {{PHASE_COUNT}} rebalance-phase combinations have positive validation Sharpe; the range is {{PHASE_MIN}}–{{PHASE_MAX}}.
- V4/diagnostic return correlation is {{CORR_FULL}} full sample and {{CORR_VAL}} in validation.
- {{POSITIVE_SYMBOLS}} of {{TOTAL_SYMBOLS}} symbols contribute positively on the arithmetic gross attribution; the top five account for {{TOP5_SHARE}} of positive contribution.

See [methodology](docs/methodology.md), [reproducibility](docs/reproducibility.md), [limitations](docs/limitations.md), the [formal tables](results/formal_v1/), and the [robustness artifacts](results/robustness_v1/).

## Public verification

Python 3.11 is the supported environment. A public clone can run the synthetic causal suite without private data:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The suite checks lag-1 execution mapping, the strict training endpoint, validation invariance of selection, future-data invariance of historical state/P&L, the frozen parameter contract, portfolio constraints, and causal rebalance forward-fill. CI never reads private data or rewrites `results/`.

Synthetic fixtures exist only for tests and CI. They are not formal research data and do not support any headline result.

## Full empirical reproduction

Users with an appropriately licensed equivalent dataset can map it to:

```text
id
datetime
close
on_warrant
```

Place the file at `data/warehouse_receipts_standardized.csv`, then run:

```bash
python scripts/run_formal.py
python scripts/run_weight_selection.py
python scripts/run_robustness.py
python scripts/build_readme.py
python scripts/build_manifests.py
```

The original dominant continuous series was instructor-constructed from CTP market data. Dominant-contract switching may differ by roughly one or two trading days across providers, so differently constructed licensed inputs should not be expected to reproduce the frozen artifacts byte-for-byte. See [data availability](data/README.md) for the private input fingerprint and schema.

## Project map

```text
configs/                 frozen v1.0 research contract
data/                    public mapping + private-input manifest/documentation
docs/                    methodology, reproducibility and limitations
results/formal_v1/       frozen real-data daily returns, metrics, NAV and period tables
results/robustness_v1/   frozen real-data plateau, phase, attribution and correlation evidence
scripts/                 formal command-line entry points and documentation builders
src/                     formal strategy, parameter provenance and robustness
tests/                   synthetic-only causal and contract tests
```

## Evidence trace

Core claims follow `claim → result artifact → generating script → configs/v1.0.json → private data fingerprint`. Artifact digests and exact generators are recorded in `results/artifact_manifest.json`. This README is generated by `scripts/build_readme.py`; performance fields are read from the formal CSVs rather than recomputed or typed by hand.

## Scope

This repository is a research portfolio project, not investment advice or a claim of live performance. See [limitations](docs/limitations.md) before interpreting the backtest. The public repository intentionally separates reproducible code and synthetic contract tests from frozen empirical evidence generated with the private research input.
