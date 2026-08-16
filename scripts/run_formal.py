"""Reproduce the frozen v1.0 formal backtest artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import formal_strategy


if __name__ == "__main__":
    formal_strategy.main()
