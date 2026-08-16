"""Warehouse Strategy v1.0 robustness and attribution module.

This file is NOT the formal strategy entry point.  It is used only for the
frozen v1.0 robustness and attribution checks.  The formal strategy remains
``formal_strategy.py``.

Formal definitions are reused directly where possible.  The phase-aware
helpers below reproduce the logic of ``apply_rebalance``, ``build_v4_weights``
and ``build_diagnostic_weights`` from the formal standalone, adding only an
explicit rebalance offset for the requested diagnostic exercise.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import formal_strategy as formal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "robustness_v1"
FORMAL_DAILY = (
    PROJECT_ROOT
    / "results"
    / "formal_v1"
    / "final_dual_combo_daily.csv"
)

OFFICIAL_RETURN = "ret_5bp_with_roll_proxy"
FORMAL_DAILY_RETURN = "combo_55_45_ret_5bp_with_roll_proxy"
BASELINE_LONG_WEIGHT = 0.55
BASELINE_SHORT_WEIGHT = 0.45
WEIGHT_GRID = np.round(np.arange(0.30, 0.7001, 0.05), 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def apply_rebalance_offset(
    frame: pd.DataFrame | pd.Series, days: int, offset: int
) -> pd.DataFrame | pd.Series:
    """Formal ``apply_rebalance`` logic with an explicit phase offset."""
    if days <= 1:
        return frame.fillna(0.0)
    if not 0 <= offset < days:
        raise ValueError(f"offset must be in [0, {days - 1}]")
    result = frame.copy()
    rebalance_day = np.arange(len(result)) % days == offset
    result.loc[~rebalance_day] = np.nan
    return result.ffill().fillna(0.0)


def period_masks(index: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Return strict train, validation and full-sample masks.

    The strict training mask follows the corrected forward-return boundary:
    the return endpoint, not only its index/start date, must be no later than
    2021-12-31.
    """
    return {
        "train_2018_2021_strict": formal.strict_training_mask(index),
        "validation_2022_2025": formal.validation_mask(index),
        "full_sample": pd.Series(True, index=index),
    }


def metrics_rows(
    daily: pd.DataFrame, identifying: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period, mask in period_masks(daily.index).items():
        stats = formal.calc_metrics(daily.loc[mask, OFFICIAL_RETURN])
        rows.append(
            {
                **identifying,
                "period": period,
                "cost_label": "5bp_with_roll_proxy",
                **stats,
            }
        )
    return rows


def prepare_v4_phase_components(
    log_warrant: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce formal ``build_v4_weights`` before rebalancing."""
    level_z_252 = formal.rolling_z(log_warrant, 252, min_periods=84)
    level_pct_252 = log_warrant.rolling(
        252, min_periods=84
    ).apply(formal.percentile_last, raw=True)

    scarcity = np.tanh(-level_z_252 / 2)
    tail_low = ((0.2 - level_pct_252) / 0.2).where(
        level_pct_252 <= 0.2, 0.0
    )
    persistence = (level_pct_252 <= 0.3).rolling(
        60, min_periods=30
    ).mean()
    score = (
        0.5 * scarcity.rank(axis=1, pct=True)
        + 0.3 * tail_low.rank(axis=1, pct=True)
        + 0.2 * persistence.rank(axis=1, pct=True)
    )

    rank = score.rank(axis=1, pct=True, method="first")
    enough_names = score.count(axis=1) >= 10
    long_mask = rank.ge(1 - formal.Q).mul(enough_names, axis=0)
    short_mask = rank.le(formal.Q).mul(enough_names, axis=0)
    raw_weights = formal.allocate_side(
        long_mask, 0.5, formal.CAP_ABS_WEIGHT
    ) + formal.allocate_side(
        short_mask, -0.5, formal.CAP_ABS_WEIGHT
    )

    tight_count = (level_pct_252 <= 0.3).sum(axis=1)
    loose_count = (level_pct_252 >= 0.7).sum(axis=1)
    state_counts = pd.concat([tight_count, loose_count], axis=1)
    balance = state_counts.min(axis=1) / state_counts.max(
        axis=1
    ).replace(0, np.nan)
    gate = pd.Series(
        np.where(balance >= 0.3, 1.0, 0.0), index=score.index
    )
    return raw_weights, gate


def build_v4_weights_phase(
    raw_weights: pd.DataFrame, gate: pd.Series, offset: int
) -> pd.DataFrame:
    """Formal V4 portfolio with only the 20-day phase changed."""
    base = apply_rebalance_offset(
        raw_weights, formal.WEIGHT_REBALANCE_DAYS, offset
    )
    weight_gate = apply_rebalance_offset(
        gate, formal.WEIGHT_REBALANCE_DAYS, offset
    )
    # The independent 63-day scale rebalance is a frozen formal parameter.
    state_gate = formal.apply_rebalance(gate, formal.SCALE_REBALANCE_DAYS)
    return base.mul(weight_gate * state_gate, axis=0)


def prepare_diagnostic_phase_components(
    warrant: pd.DataFrame, fwd_ret: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    """Reproduce formal ``build_diagnostic_weights`` before rebalancing."""
    processed = formal.causal_stale_mask(warrant.clip(lower=0), days=60)
    log_warrant = np.log1p(processed)
    change_10 = log_warrant - log_warrant.shift(10)
    score = formal.rolling_z(change_10, 60, min_periods=30)
    selected, _ = formal.direction_train_symbols(score, fwd_ret)
    subset = score[selected]
    long_mask = subset > 1.0
    short_mask = subset < -1.0
    raw_weights = formal.allocate_side(
        long_mask, 0.5, 0.15
    ) + formal.allocate_side(short_mask, -0.5, 0.15)
    raw_weights = raw_weights.reindex(
        columns=score.columns, fill_value=0.0
    )
    return raw_weights, selected


def build_diagnostic_weights_phase(
    raw_weights: pd.DataFrame, offset: int
) -> pd.DataFrame:
    """Formal Diagnostic portfolio with only the 3-day phase changed."""
    rebalanced = apply_rebalance_offset(raw_weights, 3, offset)
    return formal.smooth_with_band(rebalanced, alpha=0.3, band=0.0)


def max_abs_frame_difference(
    left: pd.DataFrame, right: pd.DataFrame
) -> float:
    a, b = left.align(right, join="outer", axis=None)
    return float(np.nanmax(np.abs(a.to_numpy() - b.to_numpy())))


def make_baseline(
    close: pd.DataFrame, warrant: pd.DataFrame
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    list[str],
    dict[str, Any],
]:
    fwd_ret = close.shift(-1) / close - 1
    return_arrays = formal.prepare_return_arrays(close)
    v4_weights = formal.build_v4_weights(np.log1p(warrant))
    diagnostic_weights, selected = formal.build_diagnostic_weights(
        warrant, fwd_ret
    )
    v4_daily = formal.add_complete_costs(
        formal.strategy_daily(v4_weights, return_arrays)
    )
    diagnostic_daily = formal.add_complete_costs(
        formal.strategy_daily(diagnostic_weights, return_arrays)
    )
    combo = formal.combine_sleeves(
        v4_daily,
        diagnostic_daily,
        BASELINE_LONG_WEIGHT,
        BASELINE_SHORT_WEIGHT,
    )

    official = pd.read_csv(
        FORMAL_DAILY, parse_dates=["datetime"]
    ).set_index("datetime")
    reference = official[FORMAL_DAILY_RETURN]
    reproduced = combo[OFFICIAL_RETURN]
    dates_match = reproduced.index.equals(reference.index)
    rows_match = len(reproduced) == len(reference)
    if dates_match and rows_match:
        max_abs_diff = float(
            np.max(np.abs(reproduced.to_numpy() - reference.to_numpy()))
        )
    else:
        max_abs_diff = float("inf")

    consistency = {
        "rows": int(len(reproduced)),
        "reference_rows": int(len(reference)),
        "start_date": reproduced.index.min().date().isoformat(),
        "end_date": reproduced.index.max().date().isoformat(),
        "dates_match": bool(dates_match),
        "rows_match": bool(rows_match),
        "max_abs_diff": max_abs_diff,
        "tolerance": 1e-12,
        "passed": bool(dates_match and rows_match and max_abs_diff < 1e-12),
        "formal_daily_sha256": sha256(FORMAL_DAILY),
    }
    return (
        v4_weights,
        diagnostic_weights,
        v4_daily,
        diagnostic_daily,
        selected,
        consistency,
    )


def draw_line_chart(
    x_values: np.ndarray,
    series: dict[str, np.ndarray],
    path: Path,
    title: str,
    x_label: str,
    y_label: str,
    baseline_x: float | None = None,
) -> None:
    width, height = 1500, 850
    left, right, top, bottom = 120, 80, 90, 110
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = [(32, 115, 210), (226, 96, 62), (50, 155, 94)]
    all_y = np.concatenate([np.asarray(v, dtype=float) for v in series.values()])
    y_min, y_max = float(all_y.min()), float(all_y.max())
    pad = max((y_max - y_min) * 0.10, 0.05)
    y_min -= pad
    y_max += pad
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    plot_w = width - left - right
    plot_h = height - top - bottom

    def xp(value: float) -> float:
        return left + (value - x_min) / max(x_max - x_min, 1e-12) * plot_w

    def yp(value: float) -> float:
        return top + (y_max - value) / max(y_max - y_min, 1e-12) * plot_h

    draw.text((left, 30), title, fill=(25, 32, 42), font=font)
    for step in range(6):
        value = y_min + (y_max - y_min) * step / 5
        y = yp(value)
        draw.line((left, y, width - right, y), fill=(225, 229, 234), width=1)
        draw.text((25, y - 7), f"{value:.2f}", fill=(55, 62, 72), font=font)
    for value in x_values:
        x = xp(float(value))
        draw.text((x - 15, height - bottom + 20), f"{value:.2f}", fill=(55, 62, 72), font=font)
    draw.rectangle((left, top, width - right, height - bottom), outline=(145, 152, 163), width=1)
    if baseline_x is not None:
        x = xp(baseline_x)
        draw.line((x, top, x, height - bottom), fill=(20, 20, 20), width=2)
        draw.text((x + 6, top + 8), "55/45 baseline", fill=(20, 20, 20), font=font)
    for position, (label, values) in enumerate(series.items()):
        color = colors[position % len(colors)]
        points = [(xp(float(x)), yp(float(y))) for x, y in zip(x_values, values)]
        draw.line(points, fill=color, width=4, joint="curve")
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=color)
        legend_y = top + 20 + position * 28
        draw.line((width - right - 260, legend_y, width - right - 215, legend_y), fill=color, width=4)
        draw.text((width - right - 205, legend_y - 7), label, fill=(30, 35, 42), font=font)
    draw.text((width // 2 - 40, height - 38), x_label, fill=(30, 35, 42), font=font)
    draw.text((24, 65), y_label, fill=(30, 35, 42), font=font)
    image.save(path)


def draw_phase_heatmap(validation: pd.DataFrame, path: Path) -> None:
    pivot = validation.pivot(
        index="short_offset", columns="long_offset", values="sharpe"
    ).reindex(index=range(3), columns=range(20))
    values = pivot.to_numpy(dtype=float)
    v_min, v_max = float(values.min()), float(values.max())
    width, height = 1800, 620
    left, top = 120, 110
    cell_w, cell_h = 78, 115
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((left, 35), "Validation Sharpe by Rebalance Phase (20 x 3)", fill=(25, 32, 42), font=font)
    for col in range(20):
        draw.text((left + col * cell_w + 28, top - 24), str(col), fill=(35, 42, 52), font=font)
    for row in range(3):
        draw.text((28, top + row * cell_h + 45), f"short {row}", fill=(35, 42, 52), font=font)
        for col in range(20):
            value = float(values[row, col])
            ratio = (value - v_min) / max(v_max - v_min, 1e-12)
            color = (
                int(40 + 200 * ratio),
                int(105 + 100 * (1 - abs(2 * ratio - 1))),
                int(220 - 170 * ratio),
            )
            x0 = left + col * cell_w
            y0 = top + row * cell_h
            draw.rectangle((x0, y0, x0 + cell_w - 3, y0 + cell_h - 3), fill=color)
            draw.text((x0 + 18, y0 + 45), f"{value:.2f}", fill="white", font=font)
            if row == 0 and col == 0:
                draw.rectangle((x0, y0, x0 + cell_w - 3, y0 + cell_h - 3), outline=(255, 230, 50), width=5)
                draw.text((x0 + 4, y0 + 8), "BASE", fill=(255, 245, 90), font=font)
    draw.text((left + 650, height - 48), "Long-sleeve 20-day offset", fill=(30, 35, 42), font=font)
    image.save(path)


def draw_symbol_contribution(table: pd.DataFrame, path: Path) -> None:
    positive = table.nlargest(10, "combo_gross_contribution")
    negative = table.nsmallest(10, "combo_gross_contribution")
    display = pd.concat([negative, positive]).drop_duplicates("symbol")
    display = display.sort_values("combo_gross_contribution")
    values = display["combo_gross_contribution"].to_numpy(dtype=float)
    labels = display["symbol"].tolist()
    width, height = 1450, 950
    left, right, top, bottom = 180, 90, 80, 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    x_bound = max(abs(float(values.min())), abs(float(values.max()))) * 1.12
    plot_w = width - left - right
    row_h = (height - top - bottom) / len(values)

    def xp(value: float) -> float:
        return left + (value + x_bound) / (2 * x_bound) * plot_w

    zero_x = xp(0.0)
    draw.text((left, 28), "Largest Positive and Negative Gross Contributions", fill=(25, 32, 42), font=font)
    draw.line((zero_x, top, zero_x, height - bottom), fill=(35, 40, 48), width=2)
    for i, (label, value) in enumerate(zip(labels, values)):
        y0 = top + i * row_h + 5
        y1 = top + (i + 1) * row_h - 5
        color = (46, 157, 95) if value >= 0 else (215, 75, 70)
        draw.rectangle((min(zero_x, xp(value)), y0, max(zero_x, xp(value)), y1), fill=color)
        draw.text((25, y0 + 5), label, fill=(35, 42, 52), font=font)
        draw.text((max(zero_x, xp(value)) + 7, y0 + 5), f"{value:.4f}", fill=(35, 42, 52), font=font)
    image.save(path)


def draw_rolling_correlation(table: pd.DataFrame, path: Path) -> None:
    pivot = table.pivot(
        index="date", columns="cost_label", values="pearson"
    ).dropna()
    x_values = np.arange(len(pivot), dtype=float)
    series = {
        "Gross": pivot["gross"].to_numpy(dtype=float),
        "5bp + Roll": pivot["5bp_with_roll_proxy"].to_numpy(dtype=float),
    }
    draw_line_chart(
        x_values,
        series,
        path,
        "Rolling 63-day Pearson Correlation",
        "Trading-day observations",
        "Correlation",
    )


def contribution_matrix(
    signal_weights: pd.DataFrame, return_arrays: dict[str, Any]
) -> pd.DataFrame:
    """Per-symbol gross return contribution using formal execution timing."""
    aligned = signal_weights.reindex(
        index=return_arrays["index"], columns=return_arrays["columns"]
    ).fillna(0.0)
    weights = aligned.to_numpy(dtype=float)
    executed = np.zeros_like(weights)
    executed[formal.EXECUTION_LAG :] = weights[: -formal.EXECUTION_LAG]
    contributions = executed * return_arrays["fwd_ret"]
    frame = pd.DataFrame(
        contributions,
        index=return_arrays["index"],
        columns=return_arrays["columns"],
    )
    return frame.loc[return_arrays["valid_day"]]


def assert_finite_table(name: str, table: pd.DataFrame) -> None:
    numeric = table.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{name} contains NaN or Inf in numeric fields")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    if not FORMAL_DAILY.exists():
        raise FileNotFoundError(FORMAL_DAILY)

    close, warrant = formal.load_data()
    fwd_ret = close.shift(-1) / close - 1
    return_arrays = formal.prepare_return_arrays(close)
    (
        formal_v4_weights,
        formal_diagnostic_weights,
        v4_daily,
        diagnostic_daily,
        selected_symbols,
        consistency,
    ) = make_baseline(close, warrant)
    write_json(OUT / "baseline_consistency.json", consistency)
    if not consistency["passed"]:
        raise RuntimeError(
            "Baseline consistency failed; robustness analysis stopped"
        )

    baseline_combo = formal.combine_sleeves(
        v4_daily,
        diagnostic_daily,
        BASELINE_LONG_WEIGHT,
        BASELINE_SHORT_WEIGHT,
    )

    # Weight plateau: only the long/short sleeve mix changes.
    weight_rows: list[dict[str, Any]] = []
    for long_weight in WEIGHT_GRID:
        short_weight = float(1.0 - long_weight)
        combo = formal.combine_sleeves(
            v4_daily,
            diagnostic_daily,
            float(long_weight),
            short_weight,
        )
        weight_rows.extend(
            metrics_rows(
                combo,
                {
                    "long_weight": float(long_weight),
                    "short_weight": short_weight,
                },
            )
        )
    weight_table = pd.DataFrame(weight_rows)
    assert_finite_table("weight_plateau", weight_table)
    weight_table.to_csv(
        OUT / "weight_plateau.csv", index=False, encoding="utf-8-sig"
    )
    weight_pivot = weight_table.pivot(
        index="long_weight", columns="period", values="sharpe"
    ).reindex(WEIGHT_GRID)
    draw_line_chart(
        WEIGHT_GRID,
        {
            "Training": weight_pivot[
                "train_2018_2021_strict"
            ].to_numpy(),
            "Validation": weight_pivot[
                "validation_2022_2025"
            ].to_numpy(),
            "Full sample": weight_pivot["full_sample"].to_numpy(),
        },
        OUT / "weight_plateau_sharpe.png",
        "55/45 Weight Plateau - Official 5bp + Roll Proxy",
        "Long-sleeve weight",
        "Sharpe",
        baseline_x=BASELINE_LONG_WEIGHT,
    )

    # Rebalance phases: compute the expensive signals once, then vary offsets.
    v4_raw, v4_gate = prepare_v4_phase_components(np.log1p(warrant))
    diagnostic_raw, phase_selected = prepare_diagnostic_phase_components(
        warrant, fwd_ret
    )
    phase_v4_baseline = build_v4_weights_phase(v4_raw, v4_gate, 0)
    phase_diagnostic_baseline = build_diagnostic_weights_phase(
        diagnostic_raw, 0
    )
    v4_signal_diff = max_abs_frame_difference(
        formal_v4_weights, phase_v4_baseline
    )
    diagnostic_signal_diff = max_abs_frame_difference(
        formal_diagnostic_weights, phase_diagnostic_baseline
    )
    if v4_signal_diff >= 1e-12 or diagnostic_signal_diff >= 1e-12:
        raise RuntimeError("Phase baseline does not reproduce formal weights")
    if phase_selected != selected_symbols:
        raise RuntimeError("Phase diagnostic symbol selection changed")

    v4_phase_daily = {
        offset: formal.add_complete_costs(
            formal.strategy_daily(
                build_v4_weights_phase(v4_raw, v4_gate, offset),
                return_arrays,
            )
        )
        for offset in range(20)
    }
    diagnostic_phase_daily = {
        offset: formal.add_complete_costs(
            formal.strategy_daily(
                build_diagnostic_weights_phase(diagnostic_raw, offset),
                return_arrays,
            )
        )
        for offset in range(3)
    }
    phase_rows: list[dict[str, Any]] = []
    for long_offset in range(20):
        for short_offset in range(3):
            combo = formal.combine_sleeves(
                v4_phase_daily[long_offset],
                diagnostic_phase_daily[short_offset],
                BASELINE_LONG_WEIGHT,
                BASELINE_SHORT_WEIGHT,
            )
            phase_rows.extend(
                metrics_rows(
                    combo,
                    {
                        "long_offset": long_offset,
                        "short_offset": short_offset,
                    },
                )
            )
    phase_table = pd.DataFrame(phase_rows)
    assert_finite_table("rebalance_phase", phase_table)
    phase_table.to_csv(
        OUT / "rebalance_phase.csv", index=False, encoding="utf-8-sig"
    )
    phase_validation = phase_table[
        phase_table["period"] == "validation_2022_2025"
    ].copy()
    draw_phase_heatmap(
        phase_validation,
        OUT / "rebalance_phase_validation_heatmap.png",
    )

    # Gross attribution by symbol.  Costs remain aggregate and are not assigned.
    v4_contribution = contribution_matrix(
        formal_v4_weights, return_arrays
    ).sum(axis=0)
    diagnostic_contribution = contribution_matrix(
        formal_diagnostic_weights, return_arrays
    ).sum(axis=0)
    symbol_table = pd.DataFrame(
        {
            "long_sleeve_contribution": v4_contribution,
            "short_sleeve_contribution": diagnostic_contribution,
        }
    )
    symbol_table["combo_gross_contribution"] = (
        BASELINE_LONG_WEIGHT
        * symbol_table["long_sleeve_contribution"]
        + BASELINE_SHORT_WEIGHT
        * symbol_table["short_sleeve_contribution"]
    )
    symbol_table["abs_contribution"] = symbol_table[
        "combo_gross_contribution"
    ].abs()
    symbol_table["rank"] = symbol_table["abs_contribution"].rank(
        ascending=False, method="first"
    ).astype(int)
    symbol_map = formal.load_symbol_map()
    symbol_table.insert(
        0,
        "symbol_name",
        [symbol_map.get(str(symbol), str(symbol)) for symbol in symbol_table.index],
    )
    symbol_table = symbol_table.reset_index(names="symbol").sort_values(
        "rank"
    )
    assert_finite_table("symbol_contribution", symbol_table)
    symbol_table.to_csv(
        OUT / "symbol_contribution.csv", index=False, encoding="utf-8-sig"
    )
    draw_symbol_contribution(
        symbol_table, OUT / "symbol_contribution_top.png"
    )

    # Sleeve correlations for gross and the official cost convention.
    v4_aligned, diagnostic_aligned = v4_daily.align(
        diagnostic_daily, join="inner", axis=0
    )
    if not v4_aligned.index.equals(diagnostic_aligned.index):
        raise RuntimeError("Sleeve dates are not aligned")
    correlation_rows: list[dict[str, Any]] = []
    correlation_columns = {
        "gross": "ret_gross",
        "5bp_with_roll_proxy": OFFICIAL_RETURN,
    }
    masks = period_masks(v4_aligned.index)
    for cost_label, column in correlation_columns.items():
        for period, mask in masks.items():
            left = v4_aligned.loc[mask, column]
            right = diagnostic_aligned.loc[mask, column]
            correlation_rows.append(
                {
                    "cost_label": cost_label,
                    "scope": "period",
                    "period": period,
                    "start": left.index.min().date().isoformat(),
                    "end": left.index.max().date().isoformat(),
                    "n_days": int(len(left)),
                    "pearson": float(left.corr(right, method="pearson")),
                    "spearman": float(left.corr(right, method="spearman")),
                }
            )
        years = sorted(v4_aligned.index.year.unique())
        for year in years:
            year_mask = v4_aligned.index.year == year
            left = v4_aligned.loc[year_mask, column]
            right = diagnostic_aligned.loc[year_mask, column]
            correlation_rows.append(
                {
                    "cost_label": cost_label,
                    "scope": "year",
                    "period": str(year),
                    "start": left.index.min().date().isoformat(),
                    "end": left.index.max().date().isoformat(),
                    "n_days": int(len(left)),
                    "pearson": float(left.corr(right, method="pearson")),
                    "spearman": float(left.corr(right, method="spearman")),
                }
            )
    correlation_table = pd.DataFrame(correlation_rows)
    assert_finite_table("sleeve_correlation", correlation_table)
    correlation_table.to_csv(
        OUT / "sleeve_correlation.csv", index=False, encoding="utf-8-sig"
    )

    rolling_rows: list[pd.DataFrame] = []
    for cost_label, column in correlation_columns.items():
        rolling = v4_aligned[column].rolling(
            63, min_periods=63
        ).corr(diagnostic_aligned[column]).dropna()
        rolling_rows.append(
            pd.DataFrame(
                {
                    "date": rolling.index,
                    "cost_label": cost_label,
                    "rolling_window": 63,
                    "pearson": rolling.to_numpy(),
                }
            )
        )
    rolling_table = pd.concat(rolling_rows, ignore_index=True)
    assert_finite_table("sleeve_correlation_rolling", rolling_table)
    rolling_table.to_csv(
        OUT / "sleeve_correlation_rolling.csv",
        index=False,
        encoding="utf-8-sig",
    )
    draw_rolling_correlation(
        rolling_table, OUT / "sleeve_correlation_rolling.png"
    )

    # Summary values and self-checks.
    train_weight = weight_table[
        weight_table["period"] == "train_2018_2021_strict"
    ].copy()
    train_weight["rank"] = train_weight["sharpe"].rank(
        ascending=False, method="min"
    ).astype(int)
    train_best = train_weight.sort_values(
        ["sharpe", "long_weight"], ascending=[False, True]
    ).iloc[0]
    baseline_train = train_weight[
        np.isclose(train_weight["long_weight"], BASELINE_LONG_WEIGHT)
    ].iloc[0]
    neighbor_results: dict[str, dict[str, float]] = {}
    for weight in (0.50, 0.55, 0.60):
        subset = weight_table[np.isclose(weight_table["long_weight"], weight)]
        neighbor_results[f"{weight:.2f}"] = {
            row["period"]: float(row["sharpe"])
            for _, row in subset.iterrows()
        }

    validation_values = phase_validation["sharpe"]
    baseline_phase = phase_validation[
        (phase_validation["long_offset"] == 0)
        & (phase_validation["short_offset"] == 0)
    ].iloc[0]
    baseline_phase_rank = int(
        validation_values.rank(ascending=False, method="min").loc[
            baseline_phase.name
        ]
    )
    baseline_percentile = float(
        (validation_values <= baseline_phase["sharpe"]).mean()
    )

    positive = symbol_table[
        symbol_table["combo_gross_contribution"] > 0
    ].sort_values("combo_gross_contribution", ascending=False)
    negative = symbol_table[
        symbol_table["combo_gross_contribution"] < 0
    ].sort_values("combo_gross_contribution")
    positive_pool = float(positive["combo_gross_contribution"].sum())

    def positive_share(count: int) -> float:
        return float(
            positive.head(count)["combo_gross_contribution"].sum()
            / positive_pool
        )

    attribution_sum = float(symbol_table["combo_gross_contribution"].sum())
    combo_gross_sum = float(baseline_combo["ret_gross"].sum())
    aggregate_cost = float(
        (
            baseline_combo["ret_gross"]
            - baseline_combo[OFFICIAL_RETURN]
        ).sum()
    )
    official_corr = correlation_table[
        (correlation_table["cost_label"] == "5bp_with_roll_proxy")
        & (correlation_table["scope"] == "period")
    ].set_index("period")
    gross_corr = correlation_table[
        (correlation_table["cost_label"] == "gross")
        & (correlation_table["scope"] == "period")
    ].set_index("period")
    official_rolling = rolling_table[
        rolling_table["cost_label"] == "5bp_with_roll_proxy"
    ]["pearson"]
    gross_rolling = rolling_table[
        rolling_table["cost_label"] == "gross"
    ]["pearson"]
    baseline_full = formal.calc_metrics(
        baseline_combo[OFFICIAL_RETURN]
    )["sharpe"]
    validation_mask = period_masks(baseline_combo.index)[
        "validation_2022_2025"
    ]
    baseline_validation = formal.calc_metrics(
        baseline_combo.loc[validation_mask, OFFICIAL_RETURN]
    )["sharpe"]
    baseline_full_reference_diff = abs(
        float(baseline_full) - 1.9402002475
    )
    baseline_validation_reference_diff = abs(
        float(baseline_validation) - 1.4301090287
    )
    strict_train_mask = period_masks(baseline_combo.index)[
        "train_2018_2021_strict"
    ]
    forward_end = pd.Series(
        baseline_combo.index, index=baseline_combo.index
    ).shift(-1)
    strict_train_forward_end = forward_end.loc[strict_train_mask]
    strict_train_forward_end_passed = bool(
        (strict_train_forward_end <= formal.TRAIN_END).all()
    )
    sleeve_date_alignment_passed = bool(
        v4_daily.index.equals(diagnostic_daily.index)
        and v4_daily.index.equals(baseline_combo.index)
    )

    summary = {
        "baseline_full_sharpe": float(baseline_full),
        "baseline_validation_sharpe": float(baseline_validation),
        "baseline_consistency": consistency,
        "weight_plateau": {
            "train_best_weight": float(train_best["long_weight"]),
            "baseline_weight": BASELINE_LONG_WEIGHT,
            "baseline_train_rank": int(baseline_train["rank"]),
            "neighboring_weights_results": neighbor_results,
            "selection_note": (
                "Diagnostics use the frozen official 5bp+roll convention; "
                "no full-sample or validation result is used to change v1.0."
            ),
        },
        "rebalance_phase": {
            "validation_mean_sharpe": float(validation_values.mean()),
            "validation_median_sharpe": float(validation_values.median()),
            "validation_min_sharpe": float(validation_values.min()),
            "validation_max_sharpe": float(validation_values.max()),
            "validation_q25_sharpe": float(validation_values.quantile(0.25)),
            "validation_q75_sharpe": float(validation_values.quantile(0.75)),
            "positive_ratio": float((validation_values > 0).mean()),
            "sharpe_gt_1_ratio": float((validation_values > 1).mean()),
            "baseline_sharpe": float(baseline_phase["sharpe"]),
            "baseline_rank": baseline_phase_rank,
            "baseline_percentile": baseline_percentile,
            "v4_offset0_signal_max_abs_diff": v4_signal_diff,
            "diagnostic_offset0_signal_max_abs_diff": diagnostic_signal_diff,
        },
        "symbol_contribution": {
            "positive_symbols": int(len(positive)),
            "negative_symbols": int(len(negative)),
            "zero_symbols": int(len(symbol_table) - len(positive) - len(negative)),
            "top1_share": positive_share(1),
            "top3_share": positive_share(3),
            "top5_share": positive_share(5),
            "largest_positive_symbol": str(positive.iloc[0]["symbol"]),
            "largest_negative_symbol": str(negative.iloc[0]["symbol"]),
            "positive_share_denominator": (
                "sum of positive 55/45 combined gross contributions"
            ),
            "contribution_definition": (
                "arithmetic sum over daily executed-weight times forward return"
            ),
            "combo_gross_return_sum": combo_gross_sum,
            "symbol_attribution_sum": attribution_sum,
            "attribution_sum_difference": float(
                attribution_sum - combo_gross_sum
            ),
            "aggregate_5bp_roll_cost": aggregate_cost,
            "cost_allocation_note": (
                "Costs are reported only in aggregate and are not allocated "
                "to individual symbols."
            ),
        },
        "sleeve_correlation": {
            "full_pearson": float(
                official_corr.loc["full_sample", "pearson"]
            ),
            "full_spearman": float(
                official_corr.loc["full_sample", "spearman"]
            ),
            "validation_pearson": float(
                official_corr.loc["validation_2022_2025", "pearson"]
            ),
            "validation_spearman": float(
                official_corr.loc["validation_2022_2025", "spearman"]
            ),
            "rolling63_median": float(official_rolling.median()),
            "rolling63_q10": float(official_rolling.quantile(0.10)),
            "rolling63_q90": float(official_rolling.quantile(0.90)),
            "gross_full_pearson": float(
                gross_corr.loc["full_sample", "pearson"]
            ),
            "gross_full_spearman": float(
                gross_corr.loc["full_sample", "spearman"]
            ),
            "gross_validation_pearson": float(
                gross_corr.loc["validation_2022_2025", "pearson"]
            ),
            "gross_validation_spearman": float(
                gross_corr.loc["validation_2022_2025", "spearman"]
            ),
            "gross_rolling63_median": float(gross_rolling.median()),
            "dates_aligned": True,
        },
        "self_checks": {
            "baseline_consistency_passed": bool(consistency["passed"]),
            "weight_grid_count": int(weight_table["long_weight"].nunique()),
            "phase_combination_count": int(
                phase_validation[["long_offset", "short_offset"]]
                .drop_duplicates()
                .shape[0]
            ),
            "strict_train_last_start": (
                baseline_combo.index[strict_train_mask][-1]
                .date()
                .isoformat()
            ),
            "strict_train_last_forward_end": (
                strict_train_forward_end.iloc[-1].date().isoformat()
            ),
            "strict_train_forward_end_passed": (
                strict_train_forward_end_passed
            ),
            "execution_lag": int(formal.EXECUTION_LAG),
            "sleeve_date_alignment_passed": sleeve_date_alignment_passed,
            "attribution_reconciliation_abs_diff": abs(
                attribution_sum - combo_gross_sum
            ),
            "nan_inf_numeric_cells": 0,
            "formal_full_sharpe_reference_abs_diff": (
                baseline_full_reference_diff
            ),
            "formal_validation_sharpe_reference_abs_diff": (
                baseline_validation_reference_diff
            ),
        },
    }

    if summary["self_checks"]["weight_grid_count"] != 9:
        raise RuntimeError("Weight plateau does not contain exactly 9 weights")
    if summary["self_checks"]["phase_combination_count"] != 60:
        raise RuntimeError("Phase analysis does not contain exactly 60 combinations")
    if abs(attribution_sum - combo_gross_sum) >= 1e-12:
        raise RuntimeError("Symbol attribution does not reconcile to gross return sum")
    if not strict_train_forward_end_passed:
        raise RuntimeError("Strict training boundary admits a future endpoint")
    if not sleeve_date_alignment_passed:
        raise RuntimeError("Sleeve and combo dates are not aligned")
    if (
        baseline_full_reference_diff >= 1e-9
        or baseline_validation_reference_diff >= 1e-9
    ):
        raise RuntimeError("Baseline Sharpe differs from the frozen references")
    write_json(OUT / "robustness_summary.json", summary)

    print("Warehouse Strategy v1.0 Robustness Freeze")
    print(f"Baseline max abs diff: {consistency['max_abs_diff']:.3e}")
    print(f"Baseline full Sharpe: {baseline_full:.10f}")
    print(f"Baseline validation Sharpe: {baseline_validation:.10f}")
    print(f"Weight grid: {summary['self_checks']['weight_grid_count']}")
    print(f"Phase combinations: {summary['self_checks']['phase_combination_count']}")


if __name__ == "__main__":
    main()
