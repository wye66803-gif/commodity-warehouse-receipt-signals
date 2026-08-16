"""Reproduce the frozen v1.0 robustness artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import robustness


if __name__ == "__main__":
    robustness.main()
