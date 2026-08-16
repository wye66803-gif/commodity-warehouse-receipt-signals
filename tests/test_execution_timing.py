from __future__ import annotations

import numpy as np
import pandas as pd

import formal_strategy as formal


def test_lag1_execution_mapping() -> None:
    index = pd.bdate_range("2024-01-02", periods=4)
    signal_weights = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0]}, index=index)
    return_arrays = {
        "fwd_ret": np.array([[0.10], [0.20], [0.30], [0.40]]),
        "valid_day": np.array([True, True, True, True]),
        "roll_day": np.array([False, False, False, False]),
        "index": index,
        "columns": pd.Index(["A"]),
    }

    daily = formal.strategy_daily(signal_weights, return_arrays)

    np.testing.assert_allclose(daily["ret_gross"], [0.0, 0.20, 0.0, 0.0])
    np.testing.assert_allclose(daily["gross_exposure"], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(daily["strategy_turnover"], [0.0, 1.0, 1.0, 0.0])


def test_strict_train_boundary_excludes_cross_boundary_endpoint() -> None:
    index = pd.DatetimeIndex(
        [
            "2021-12-28",
            "2021-12-29",
            "2021-12-30",
            "2021-12-31",
            "2022-01-04",
        ]
    )
    mask = formal.strict_training_mask(index)
    assert mask.tolist() == [True, True, True, False, False]

    daily = pd.DataFrame(
        {
            "ret_gross": [0.001, -0.002, 0.003, 0.50, 0.004],
            "ret_2bp_strategy_only": [0.0008, -0.0022, 0.0028, 0.49, 0.0038],
            "ret_5bp_with_roll_proxy": [0.0005, -0.0025, 0.0025, 0.48, 0.0035],
            "strategy_turnover": [0.1] * 5,
            "roll_turnover": [0.0] * 5,
        },
        index=index,
    )
    table = formal.train_validation_table(daily)
    strict_gross = table.loc[
        (table["period"] == "train_2018_2021_strict")
        & (table["cost_label"] == "gross")
    ].iloc[0]

    assert strict_gross["end"] == "2021-12-30"
    assert strict_gross["n_days"] == 3
