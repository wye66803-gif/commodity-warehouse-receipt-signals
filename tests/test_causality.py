from __future__ import annotations

import numpy as np
import pandas as pd

import formal_strategy as formal


def _build_weights_and_daily(
    close: pd.DataFrame, warrant: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fwd_ret = close.shift(-1) / close - 1
    arrays = formal.prepare_return_arrays(close)
    v4_weights = formal.build_v4_weights(np.log1p(warrant))
    diagnostic_weights, _ = formal.build_diagnostic_weights(warrant, fwd_ret)
    v4_daily = formal.strategy_daily(v4_weights, arrays)
    diagnostic_daily = formal.strategy_daily(diagnostic_weights, arrays)
    return v4_weights, diagnostic_weights, v4_daily, diagnostic_daily


def test_future_data_perturbation_does_not_change_historical_state_or_pnl(
    synthetic_market: tuple[pd.DataFrame, pd.DataFrame],
) -> None:
    close, warrant = synthetic_market
    cutoff = pd.Timestamp("2022-03-01")
    future = close.index > cutoff

    changed_close = close.copy()
    changed_warrant = warrant.copy()
    changed_close.loc[future] = changed_close.loc[future].mul(
        np.linspace(1.2, 1.8, future.sum()), axis=0
    )
    changed_warrant.loc[future] = changed_warrant.loc[future] + 10000.0

    original = _build_weights_and_daily(close, warrant)
    changed = _build_weights_and_daily(changed_close, changed_warrant)

    original_level_signal = formal.rolling_z(
        np.log1p(warrant), 252, min_periods=84
    )
    changed_level_signal = formal.rolling_z(
        np.log1p(changed_warrant), 252, min_periods=84
    )
    pd.testing.assert_frame_equal(
        original_level_signal.loc[:cutoff],
        changed_level_signal.loc[:cutoff],
    )
    pd.testing.assert_frame_equal(original[0].loc[:cutoff], changed[0].loc[:cutoff])
    pd.testing.assert_frame_equal(original[1].loc[:cutoff], changed[1].loc[:cutoff])

    cutoff_position = close.index.get_loc(cutoff)
    last_completed_return_start = close.index[cutoff_position - 1]
    pd.testing.assert_frame_equal(
        original[2].loc[:last_completed_return_start],
        changed[2].loc[:last_completed_return_start],
    )
    pd.testing.assert_frame_equal(
        original[3].loc[:last_completed_return_start],
        changed[3].loc[:last_completed_return_start],
    )
