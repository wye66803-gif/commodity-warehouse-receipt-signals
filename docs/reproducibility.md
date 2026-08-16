# Reproducibility and evidence map

## Environment

The project targets Python 3.11. Runtime dependencies are pinned in `requirements.txt`; test dependencies are pinned separately in `requirements-dev.txt`. Project paths are derived from `pathlib.Path(__file__)`, so the code does not depend on the original private research repository or a machine-specific `PYTHONPATH`.

## Public reproducibility

A public clone can reproduce the environment and all synthetic causal checks without private data:

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

This verifies imports, lag-1 execution, the strict training endpoint, validation invariance of parameter selection, future-data invariance of historical state and P&L, the frozen parameter contract, portfolio constraints, and causal rebalance forward-fill. CI performs only this public synthetic workflow and never invokes the full empirical backtest or rewrites `results/`.

The frozen result artifacts, their summaries and their hashes can also be inspected without the private input.

## Full empirical reproduction

Full empirical regeneration requires an appropriately licensed dataset at:

```text
data/warehouse_receipts_standardized.csv
```

It must map to the documented schema:

```text
id
datetime
close
on_warrant
```

After providing that input, run:

```bash
python scripts/run_formal.py
python scripts/run_weight_selection.py
python scripts/run_robustness.py
python scripts/build_readme.py
python scripts/build_manifests.py
```

Run the formal strategy first because robustness checks reconcile against its daily return artifact. The weight-selection run is independent of legacy outputs and rebuilds both sleeves from the formal implementation. `build_readme.py` reads headline metrics from formal result CSVs rather than recomputing or manually encoding them.

The original continuous futures series was constructed by the research instructor from CTP market data using a custom dominant-contract stitching process. Provider definitions can switch dominant contracts one or two trading days apart, so an equivalent licensed input may reproduce the method without reproducing every frozen byte unless the original data construction is matched.

The repository never silently substitutes synthetic observations when a formal command is requested.

## Claim-to-artifact chain

| Claim | Direct evidence | Generator / code | Frozen config | Data version |
|---|---|---|---|---|
| Full 5bp+roll Sharpe 1.9402 and validation Sharpe 1.4301 | `results/formal_v1/final_dual_combo_train_validation.csv` | `scripts/run_formal.py` → `src/formal_strategy.py` | `configs/v1.0.json` | private fingerprint in `data/manifest.json` |
| Formal NAV and daily returns | `results/formal_v1/final_dual_combo_nav.png`, `final_dual_combo_daily.csv` | same formal entry point | same | same |
| 55/45 selected by strict-train gross Sharpe on a 5% grid | `results/weight_selection_v1/dual_combo_summary.json`, `dual_combo_train_validation.csv` | `scripts/run_weight_selection.py` → `src/weight_selection.py` | same | same |
| Last strict-train return start is 2021-12-30 and no endpoint crosses into 2022 | formal period table plus `results/robustness_v1/robustness_summary.json:self_checks` | formal/robustness entry points and `formal_strategy.strict_training_mask` | same | same |
| Lag-1 execution | formal summary `execution_lag`, `tests/test_execution_timing.py` | `formal_strategy.strategy_daily` | same | same |
| Official 5bp + monthly roll proxy | formal period table `cost_label=5bp_with_roll_proxy` | `formal_strategy.add_complete_costs` | same | same |
| Plateau, phase, correlation and contribution conclusions | `results/robustness_v1/robustness_summary.json` plus supporting CSV/PNG files | `scripts/run_robustness.py` → `src/robustness.py` | same | same |

`results/artifact_manifest.json` records the SHA256 of every curated result artifact and its producing code/config/private-data fingerprint. `data/manifest.json` describes the excluded private input and the included public mapping.

## Weight-selection artifact scope

`dual_combo_summary.json` and `dual_combo_train_validation.csv` are the direct frozen-selection evidence. Files named `dual_combo_leaderboard`, `dual_combo_best_daily` and `dual_combo_nav_compare` are retained full-sample exploratory diagnostics. Their “best” result is not used by `select_training_weight`, does not update `configs/v1.0.json`, and is not the official portfolio.
