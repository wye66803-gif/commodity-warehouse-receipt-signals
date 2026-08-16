from __future__ import annotations

import numpy as np
import pandas as pd

import formal_strategy as formal
import weight_selection as provenance


def _selection_fixture() -> pd.DataFrame:
    index = pd.bdate_range("2018-01-02", "2022-12-30")
    pair = pd.DataFrame(0.0, index=index, columns=["v4", "diagnostic"])
    train_mask = formal.strict_training_mask(index)
    n_train = int(train_mask.sum())

    x = np.linspace(-1.0, 1.0, n_train)
    x = (x - x.mean()) / x.std(ddof=1)
    y = np.sin(np.arange(n_train, dtype=float))
    y = y - y.mean()
    y = y - x * np.dot(x, y) / np.dot(x, x)
    y = y / y.std(ddof=1)

    pair.loc[train_mask, "v4"] = 0.001 + 0.010 * x
    pair.loc[train_mask, "diagnostic"] = (
        0.001 + 0.010 * np.sqrt(0.55 / 0.45) * y
    )
    pair.loc[~train_mask, "v4"] = 0.002
    pair.loc[~train_mask, "diagnostic"] = -0.001
    return pair


def test_validation_perturbation_does_not_alter_frozen_training_selection() -> None:
    baseline = _selection_fixture()
    selected, sharpes = provenance.select_training_weight(baseline)
    assert selected == provenance.FROZEN_V4_WEIGHT == 0.55

    perturbed = baseline.copy()
    validation = formal.validation_mask(perturbed.index)
    perturbed.loc[validation, "v4"] = np.linspace(-5.0, 5.0, validation.sum())
    perturbed.loc[validation, "diagnostic"] = np.linspace(
        7.0, -7.0, validation.sum()
    )
    selected_after, sharpes_after = provenance.select_training_weight(perturbed)

    assert selected_after == selected
    pd.testing.assert_series_equal(sharpes_after, sharpes)
