"""Render README headline fields directly from frozen result artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "results" / "formal_v1" / "final_dual_combo_train_validation.csv"
ROBUSTNESS = ROOT / "results" / "robustness_v1" / "robustness_summary.json"
TEMPLATE = ROOT / "docs" / "README_TEMPLATE.md"
OUTPUT = ROOT / "README.md"


def load_period(period: str) -> dict[str, str]:
    with FORMAL.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        matches = [
            row
            for row in rows
            if row["period"] == period
            and row["cost_label"] == "5bp_with_roll_proxy"
            and row["strategy"] == "Combo train-selected 55/45"
        ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one formal row for {period}, found {len(matches)}")
    return matches[0]


def pct(value: str) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> None:
    full = load_period("full_sample")
    validation = load_period("validation_2022_2025")
    if round(float(full["sharpe"]), 4) != 1.9402:
        raise RuntimeError("Frozen full-sample Sharpe contract changed")
    if round(float(validation["sharpe"]), 4) != 1.4301:
        raise RuntimeError("Frozen validation Sharpe contract changed")

    robustness = json.loads(ROBUSTNESS.read_text(encoding="utf-8"))
    plateau = robustness["weight_plateau"]
    neighbors = plateau["neighboring_weights_results"]
    phase = robustness["rebalance_phase"]
    contribution = robustness["symbol_contribution"]
    correlation = robustness["sleeve_correlation"]

    values = {
        "FULL_START": full["start"],
        "FULL_END": full["end"],
        "FULL_RETURN": pct(full["annual_return"]),
        "FULL_VOL": pct(full["annual_vol"]),
        "FULL_SHARPE": f"{float(full['sharpe']):.4f}",
        "FULL_MDD": pct(full["max_drawdown"]),
        "FULL_NAV": f"{float(full['final_nav']):.4f}",
        "VAL_START": validation["start"],
        "VAL_END": validation["end"],
        "VAL_RETURN": pct(validation["annual_return"]),
        "VAL_VOL": pct(validation["annual_vol"]),
        "VAL_SHARPE": f"{float(validation['sharpe']):.4f}",
        "VAL_MDD": pct(validation["max_drawdown"]),
        "VAL_NAV": f"{float(validation['final_nav']):.4f}",
        "PLATEAU_RANK": str(plateau["baseline_train_rank"]),
        "W50_VAL": f"{neighbors['0.50']['validation_2022_2025']:.3f}",
        "W55_VAL": f"{neighbors['0.55']['validation_2022_2025']:.3f}",
        "W60_VAL": f"{neighbors['0.60']['validation_2022_2025']:.3f}",
        "PHASE_COUNT": str(robustness["self_checks"]["phase_combination_count"]),
        "PHASE_MIN": f"{phase['validation_min_sharpe']:.3f}",
        "PHASE_MAX": f"{phase['validation_max_sharpe']:.3f}",
        "CORR_FULL": f"{correlation['full_pearson']:.3f}",
        "CORR_VAL": f"{correlation['validation_pearson']:.3f}",
        "POSITIVE_SYMBOLS": str(contribution["positive_symbols"]),
        "TOTAL_SYMBOLS": str(
            contribution["positive_symbols"]
            + contribution["negative_symbols"]
            + contribution["zero_symbols"]
        ),
        "TOP5_SHARE": f"{100.0 * contribution['top5_share']:.2f}%",
    }

    rendered = TEMPLATE.read_text(encoding="utf-8")
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    if "{{" in rendered or "}}" in rendered:
        raise RuntimeError("Unresolved README template placeholder")
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
