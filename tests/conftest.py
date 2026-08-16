from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def synthetic_market() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic public fixture with no teacher-provided observations."""
    index = pd.bdate_range("2018-01-02", "2022-06-30")
    symbols = [f"S{i:02d}" for i in range(12)]
    time = np.arange(len(index), dtype=float)[:, None]
    cross_section = np.arange(len(symbols), dtype=float)[None, :]

    warrant_values = (
        1000.0
        + 8.0 * cross_section
        + 0.15 * time
        + 35.0 * np.sin(time / 19.0 + cross_section / 3.0)
        + 12.0 * np.cos(time / 47.0 - cross_section / 5.0)
    )
    warrant = pd.DataFrame(warrant_values, index=index, columns=symbols)

    daily_returns = (
        0.00015
        + 0.0015 * np.sin(time / 13.0 + cross_section / 4.0)
        + 0.0008 * np.cos(time / 31.0 - cross_section / 6.0)
    )
    close_values = 100.0 * np.exp(np.cumsum(daily_returns, axis=0))
    close = pd.DataFrame(close_values, index=index, columns=symbols)
    return close, warrant
