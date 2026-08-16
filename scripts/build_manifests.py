"""Build public data and result manifests with SHA256 provenance."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CONFIG = "configs/v1.0.json"
PRIVATE_DATA_SHA256 = (
    "b448dc7daae786eca901572f1e4bb36ac511797b88a2e2f0af689652d9fe5256"
)
PRIVATE_DATA_ROWS = 69108
PRIVATE_DATA_SYMBOLS = 44
PRIVATE_DATA_START = "2018-01-02"
PRIVATE_DATA_END = "2025-12-31"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_formal_data(path: Path) -> tuple[int, int, str, str]:
    row_count = 0
    symbols: set[str] = set()
    date_min: str | None = None
    date_max: str | None = None
    with path.open(encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        if rows.fieldnames != ["id", "datetime", "close", "on_warrant"]:
            raise RuntimeError(f"Unexpected formal data schema: {rows.fieldnames}")
        for row in rows:
            row_count += 1
            symbols.add(row["id"])
            date = row["datetime"][:10]
            date_min = date if date_min is None or date < date_min else date_min
            date_max = date if date_max is None or date > date_max else date_max
    if date_min is None or date_max is None:
        raise RuntimeError("Formal data file is empty")
    return row_count, len(symbols), date_min, date_max


def inspect_mapping(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or list(rows[0]) != ["symbol", "symbol_name", "exchange"]:
        raise RuntimeError("Unexpected public mapping schema")
    if len({row["symbol"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate symbols in public mapping")
    return len(rows)


def build_data_manifest() -> dict[str, object]:
    data = DATA_DIR / "warehouse_receipts_standardized.csv"
    mapping = DATA_DIR / "symbol_mapping.csv"
    mapping_rows = inspect_mapping(mapping)
    if mapping_rows != PRIVATE_DATA_SYMBOLS:
        raise RuntimeError("Mapping does not cover every formal-data symbol")
    if data.exists():
        rows, symbols, start, end = inspect_formal_data(data)
        if (
            sha256(data) != PRIVATE_DATA_SHA256
            or rows != PRIVATE_DATA_ROWS
            or symbols != PRIVATE_DATA_SYMBOLS
            or start != PRIVATE_DATA_START
            or end != PRIVATE_DATA_END
        ):
            raise RuntimeError("Private research input differs from frozen metadata")
    return {
        "manifest_version": 1,
        "research_results_publication_status": "Confirmed by the supplying research instructor.",
        "input_data_redistribution_status": "NOT REDISTRIBUTED",
        "license_scope_note": "Permission to publish the research results is separate from redistribution rights for the underlying market data, so the full historical input is intentionally excluded.",
        "formal_research_data": True,
        "files": {
            "warehouse_receipts_standardized.csv": {
                "description": "Private standardized warehouse-receipt and close observations used to generate frozen v1.0 results.",
                "availability": "private research input; not included in the public repository",
                "included_in_public_repository": False,
                "source_note": "Supplied by the research instructor. The instructor constructed the continuous futures price series from CTP market data using a custom dominant-contract stitching process.",
                "sha256": PRIVATE_DATA_SHA256,
                "rows": PRIVATE_DATA_ROWS,
                "symbols": PRIVATE_DATA_SYMBOLS,
                "date_start": PRIVATE_DATA_START,
                "date_end": PRIVATE_DATA_END,
                "schema": {
                    "id": "commodity symbol string",
                    "datetime": "observation date, YYYY-MM-DD",
                    "close": "standardized research close, numeric",
                    "on_warrant": "standardized warehouse receipt quantity, numeric"
                }
            },
            "symbol_mapping.csv": {
                "description": "Clean UTF-8 symbol/name/exchange mapping.",
                "source_public_name": "approved source workbook (not redistributed)",
                "transformation": "selected three research fields; trimmed whitespace; removed Office container metadata",
                "sha256": sha256(mapping),
                "rows": mapping_rows,
                "schema": {
                    "symbol": "commodity symbol string",
                    "symbol_name": "commodity name",
                    "exchange": "listing exchange"
                }
            }
        },
        "synthetic_fixture_note": "Synthetic observations are generated only in tests/conftest.py for tests and CI; they are not formal research data or evidence for the frozen results."
    }


def build_artifact_manifest(data_sha256: str) -> dict[str, object]:
    groups = {
        "formal_v1": "scripts/run_formal.py -> src/formal_strategy.py",
        "weight_selection_v1": "scripts/run_weight_selection.py -> src/weight_selection.py",
        "robustness_v1": "scripts/run_robustness.py -> src/robustness.py",
    }
    artifacts: list[dict[str, object]] = []
    for group, generator in groups.items():
        directory = RESULTS_DIR / group
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.iterdir() if item.is_file()):
            artifacts.append({
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "generator": generator,
                "config": CONFIG,
                "data_sha256": data_sha256,
            })
    return {
        "manifest_version": 1,
        "frozen_research_version": "v1.0",
        "artifacts": artifacts,
    }


def main() -> None:
    data_manifest = build_data_manifest()
    (DATA_DIR / "manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact_manifest = build_artifact_manifest(PRIVATE_DATA_SHA256)
    (RESULTS_DIR / "artifact_manifest.json").write_text(
        json.dumps(artifact_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Data files: {len(data_manifest['files'])}")
    print(f"Result artifacts: {len(artifact_manifest['artifacts'])}")


if __name__ == "__main__":
    main()
