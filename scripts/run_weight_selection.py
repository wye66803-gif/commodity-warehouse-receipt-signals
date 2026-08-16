"""Reproduce the strict-training 55/45 selection provenance artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import weight_selection


if __name__ == "__main__":
    weight_selection.main()
