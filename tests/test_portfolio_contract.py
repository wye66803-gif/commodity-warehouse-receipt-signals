from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import formal_strategy as formal
import weight_selection as provenance
import robustness


def test_frozen_parameter_contract() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "v1.0.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert config["version"] == "v1.0"
    assert config["status"] == "frozen"
    assert config["execution_lag_days"] == formal.EXECUTION_LAG == 1
    assert config["portfolio"]["v4_weight"] == formal.MAIN_WEIGHTS[0]
    assert config["portfolio"]["diagnostic_weight"] == formal.MAIN_WEIGHTS[1]
    assert config["selection"]["metric"] == provenance.SELECTION_METRIC
    assert config["selection"]["return_label"] == provenance.SELECTION_RETURN_LABEL
    assert config["official_reporting_cost"] == "5bp_with_roll_proxy"

    assert formal.EXECUTION_LAG == 1
    assert formal.MAIN_WEIGHTS == (0.55, 0.45)
    assert formal.Q == 0.40
    assert formal.WEIGHT_REBALANCE_DAYS == 20
    assert formal.SCALE_REBALANCE_DAYS == 63
    assert formal.CAP_ABS_WEIGHT == 0.08
    assert provenance.SELECTION_METRIC == "gross_sharpe"
    assert provenance.SELECTION_RETURN_LABEL == "gross"
    assert provenance.FROZEN_V4_WEIGHT == robustness.BASELINE_LONG_WEIGHT == 0.55
    assert (
        provenance.FROZEN_DIAGNOSTIC_WEIGHT
        == robustness.BASELINE_SHORT_WEIGHT
        == 0.45
    )
    assert np.isclose(sum(formal.MAIN_WEIGHTS), 1.0)


def test_portfolio_weight_constraints() -> None:
    index = pd.DatetimeIndex(["2024-01-02", "2024-01-03"])
    columns = [f"S{i:02d}" for i in range(14)]
    long_mask = pd.DataFrame(False, index=index, columns=columns)
    short_mask = long_mask.copy()
    long_mask.iloc[:, :7] = True
    short_mask.iloc[:, 7:] = True

    long_weights = formal.allocate_side(long_mask, 0.5, formal.CAP_ABS_WEIGHT)
    short_weights = formal.allocate_side(short_mask, -0.5, formal.CAP_ABS_WEIGHT)
    combined = long_weights + short_weights

    assert (long_weights.abs() <= formal.CAP_ABS_WEIGHT + 1e-12).all().all()
    assert (short_weights.abs() <= formal.CAP_ABS_WEIGHT + 1e-12).all().all()
    np.testing.assert_allclose(long_weights.sum(axis=1), 0.5)
    np.testing.assert_allclose(short_weights.sum(axis=1), -0.5)
    np.testing.assert_allclose(combined.sum(axis=1), 0.0)
    np.testing.assert_allclose(combined.abs().sum(axis=1), 1.0)


def test_rebalance_is_causal_forward_fill() -> None:
    index = pd.bdate_range("2024-01-02", periods=6)
    source = pd.DataFrame({"A": [1.0, 99.0, 98.0, 4.0, 97.0, 96.0]}, index=index)
    result = formal.apply_rebalance(source, days=3)
    np.testing.assert_allclose(result["A"], [1.0, 1.0, 1.0, 4.0, 4.0, 4.0])

    changed_non_rebalance = source.copy()
    changed_non_rebalance.iloc[[1, 2, 4, 5], 0] = -1000.0
    pd.testing.assert_frame_equal(
        formal.apply_rebalance(changed_non_rebalance, days=3), result
    )

    offset_result = robustness.apply_rebalance_offset(source, days=3, offset=2)
    np.testing.assert_allclose(
        offset_result["A"], [0.0, 0.0, 98.0, 98.0, 98.0, 96.0]
    )
