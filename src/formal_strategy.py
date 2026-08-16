"""
Final pure warrant dual-strategy backtest.

Sleeves:
1. V4: long-horizon warrant scarcity signal.
2. Diagnostic: short-cycle warrant shock signal.
3. Main portfolio: 55% V4 + 45% Diagnostic, selected on the training period.

Important constraints:
- Signals use only warehouse warrant data.
- Price data is used only for forward return calculation.
- Execution lag is 1 day to avoid same-day look-ahead bias.
- Cost stress includes strategy turnover and a monthly futures roll-turnover proxy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "warehouse_receipts_standardized.csv"
SYMBOL_MAP_FILE = PROJECT_ROOT / "data" / "symbol_mapping.csv"
OUT = PROJECT_ROOT / "results" / "formal_v1"

ANN = 252
EXECUTION_LAG = 1
MAIN_WEIGHTS = (0.55, 0.45)
REFERENCE_WEIGHTS = (0.60, 0.40)

Q = 0.40
WEIGHT_REBALANCE_DAYS = 20
SCALE_REBALANCE_DAYS = 63
CAP_ABS_WEIGHT = 0.08

TRAIN_START = pd.Timestamp("2018-01-01")
TRAIN_END = pd.Timestamp("2021-12-31")
VALIDATION_START = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2025-12-31")

COST_LABELS = (
    "gross",
    "2bp_strategy_only",
    "5bp_strategy_only",
    "10bp_strategy_only",
    "2bp_with_roll_proxy",
    "5bp_with_roll_proxy",
)


def strict_training_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Select returns whose start and next-trading-day endpoint are in train."""
    forward_end = pd.Series(index, index=index).shift(-1)
    return pd.Series(
        (index >= TRAIN_START)
        & (index <= TRAIN_END)
        & (forward_end <= TRAIN_END),
        index=index,
        dtype=bool,
    )


def validation_mask(index: pd.DatetimeIndex) -> pd.Series:
    """Select returns whose start belongs to the frozen validation interval."""
    return pd.Series(
        (index >= VALIDATION_START) & (index <= VALIDATION_END),
        index=index,
        dtype=bool,
    )


def load_symbol_map() -> dict[str, str]:
    """Load the clean public symbol-name map when it is available."""
    if not SYMBOL_MAP_FILE.exists():
        return {}
    try:
        table = pd.read_csv(SYMBOL_MAP_FILE)
        required = {"symbol", "symbol_name"}
        if not required.issubset(table.columns):
            return {}
        return dict(zip(table["symbol"].astype(str), table["symbol_name"].astype(str)))
    except (OSError, ValueError, pd.errors.ParserError):
        return {}


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "Full empirical reproduction requires the licensed research "
            "input dataset at data/warehouse_receipts_standardized.csv. "
            "See data/README.md for the required schema and data-availability "
            "notes."
        )
    data = pd.read_csv(DATA_FILE, parse_dates=["datetime"])
    required = {"datetime", "id", "close", "on_warrant"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.sort_values(["datetime", "id"]).drop_duplicates(
        ["datetime", "id"]
    )
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["on_warrant"] = pd.to_numeric(data["on_warrant"], errors="coerce")
    close = data.pivot(
        index="datetime", columns="id", values="close"
    ).sort_index()
    warrant = (
        data.pivot(index="datetime", columns="id", values="on_warrant")
        .sort_index()
        .clip(lower=0)
    )
    return close, warrant


def rolling_z(
    frame: pd.DataFrame, window: int, min_periods: int | None = None
) -> pd.DataFrame:
    if min_periods is None:
        min_periods = max(10, window // 2)
    mean = frame.rolling(window, min_periods=min_periods).mean()
    std = frame.rolling(window, min_periods=min_periods).std(ddof=0)
    return (frame - mean) / std.replace(0, np.nan)


def percentile_last(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    if len(valid) < 2 or not np.isfinite(values[-1]):
        return np.nan
    return float(pd.Series(valid).rank(pct=True).iloc[-1])


def apply_rebalance(
    frame: pd.DataFrame | pd.Series, days: int
) -> pd.DataFrame | pd.Series:
    if days <= 1:
        return frame.fillna(0.0)
    result = frame.copy()
    rebalance_day = np.arange(len(result)) % days == 0
    result.loc[~rebalance_day] = np.nan
    return result.ffill().fillna(0.0)


def allocate_side(
    mask: pd.DataFrame, target: float, cap_abs_weight: float
) -> pd.DataFrame:
    count = mask.sum(axis=1).replace(0, np.nan)
    raw = mask.astype(float).div(count, axis=0) * abs(target)
    capped = raw.clip(upper=cap_abs_weight)
    rescale = (
        abs(target) / capped.sum(axis=1).replace(0, np.nan)
    ).clip(upper=1.0)
    return capped.mul(rescale, axis=0).fillna(0.0) * np.sign(target)


def build_v4_weights(log_warrant: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the final V4 scarcity score, portfolio and state gate."""
    level_z_252 = rolling_z(log_warrant, 252, min_periods=84)
    level_pct_252 = log_warrant.rolling(
        252, min_periods=84
    ).apply(percentile_last, raw=True)

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
    long_mask = rank.ge(1 - Q).mul(enough_names, axis=0)
    short_mask = rank.le(Q).mul(enough_names, axis=0)
    base_weights = allocate_side(
        long_mask, 0.5, CAP_ABS_WEIGHT
    ) + allocate_side(short_mask, -0.5, CAP_ABS_WEIGHT)
    base_weights = apply_rebalance(
        base_weights, WEIGHT_REBALANCE_DAYS
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
    weight_gate = apply_rebalance(gate, WEIGHT_REBALANCE_DAYS)
    state_gate = apply_rebalance(gate, SCALE_REBALANCE_DAYS)
    signal_scale = weight_gate * state_gate
    return base_weights.mul(signal_scale, axis=0)


def causal_stale_mask(frame: pd.DataFrame, days: int = 60) -> pd.DataFrame:
    """Mask a value once it has remained unchanged for `days` observations."""
    same = frame.eq(frame.shift(1)) & frame.notna() & frame.shift(1).notna()
    stale = (
        same.rolling(days - 1, min_periods=days - 1).sum()
        >= days - 1
    )
    return frame.mask(stale)


def direction_train_symbols(
    score: pd.DataFrame, fwd_ret: pd.DataFrame
) -> tuple[list[str], pd.Series]:
    """Select Diagnostic symbols using training-period information only."""
    lagged_score = score.shift(EXECUTION_LAG)
    train_mask = strict_training_mask(fwd_ret.index)
    train_ic = lagged_score.loc[train_mask].rank().corrwith(
        fwd_ret.loc[train_mask].rank()
    )
    selected = train_ic[train_ic > 0.01].index.tolist()
    if len(selected) < 10:
        selected = train_ic.nlargest(
            min(10, train_ic.notna().sum())
        ).index.tolist()
    return selected, train_ic


def smooth_with_band(
    target: pd.DataFrame, alpha: float, band: float
) -> pd.DataFrame:
    values = target.to_numpy(dtype=float)
    result = np.zeros_like(values)
    previous = np.zeros(values.shape[1], dtype=float)
    for row in range(len(values)):
        proposed = alpha * values[row] + (1.0 - alpha) * previous
        current = np.where(
            np.abs(proposed - previous) < band, previous, proposed
        )
        result[row] = current
        previous = current
    return pd.DataFrame(
        result, index=target.index, columns=target.columns
    )


def build_diagnostic_weights(
    warrant: pd.DataFrame, fwd_ret: pd.DataFrame
) -> tuple[pd.DataFrame, list[str]]:
    processed = causal_stale_mask(warrant.clip(lower=0), days=60)
    log_warrant = np.log1p(processed)
    change_10 = log_warrant - log_warrant.shift(10)
    score = rolling_z(change_10, 60, min_periods=30)
    selected, _ = direction_train_symbols(score, fwd_ret)

    subset = score[selected]
    long_mask = subset > 1.0
    short_mask = subset < -1.0
    weights = allocate_side(
        long_mask, 0.5, 0.15
    ) + allocate_side(short_mask, -0.5, 0.15)
    weights = weights.reindex(columns=score.columns, fill_value=0.0)
    weights = apply_rebalance(weights, days=3)
    return smooth_with_band(weights, alpha=0.3, band=0.0), selected


def prepare_return_arrays(close: pd.DataFrame) -> dict[str, Any]:
    """Prepare forward returns and a simple calendar-month roll-day proxy."""
    fwd_ret = close.shift(-1) / close - 1
    month = fwd_ret.index.to_period("M")
    return {
        "fwd_ret": fwd_ret.to_numpy(dtype=float),
        "valid_day": fwd_ret.notna().any(axis=1).to_numpy(),
        "roll_day": np.r_[True, month[1:] != month[:-1]],
        "index": fwd_ret.index,
        "columns": fwd_ret.columns,
    }


def strategy_daily(
    signal_weights: pd.DataFrame, return_arrays: dict[str, Any]
) -> pd.DataFrame:
    """Apply the one-day execution lag before multiplying by forward returns."""
    aligned = signal_weights.reindex(
        index=return_arrays["index"],
        columns=return_arrays["columns"],
    ).fillna(0.0)
    weights = aligned.to_numpy(dtype=float)
    exec_weights = np.zeros_like(weights)
    exec_weights[EXECUTION_LAG:] = weights[:-EXECUTION_LAG]

    gross = np.nansum(
        exec_weights * return_arrays["fwd_ret"], axis=1
    )
    strategy_turnover = np.abs(
        np.diff(
            exec_weights,
            axis=0,
            prepend=np.zeros((1, weights.shape[1])),
        )
    ).sum(axis=1)
    gross_exposure = np.abs(exec_weights).sum(axis=1)
    roll_turnover = np.where(
        return_arrays["roll_day"], gross_exposure, 0.0
    )
    daily = pd.DataFrame(
        {
            "ret_gross": gross,
            "strategy_turnover": strategy_turnover,
            "roll_turnover": roll_turnover,
            "gross_exposure": gross_exposure,
        },
        index=return_arrays["index"],
    )
    return daily.loc[return_arrays["valid_day"]]


def add_complete_costs(daily: pd.DataFrame) -> pd.DataFrame:
    """Add strategy-only and strategy-plus-roll-proxy cost conventions."""
    result = daily.copy()
    for bp in (2, 5, 10):
        result[f"ret_{bp}bp_strategy_only"] = (
            result["ret_gross"]
            - result["strategy_turnover"] * bp / 10000
        )
    for bp in (2, 5):
        result[f"ret_{bp}bp_with_roll_proxy"] = (
            result["ret_gross"]
            - (
                result["strategy_turnover"]
                + result["roll_turnover"]
            )
            * bp
            / 10000
        )
    return result


def combine_sleeves(
    v4_daily: pd.DataFrame,
    diagnostic_daily: pd.DataFrame,
    w_v4: float,
    w_diag: float,
) -> pd.DataFrame:
    v4_aligned, diagnostic_aligned = v4_daily.align(
        diagnostic_daily, join="inner", axis=0
    )
    combo = pd.DataFrame(index=v4_aligned.index)
    for column in (
        "ret_gross",
        "strategy_turnover",
        "roll_turnover",
        "gross_exposure",
    ):
        combo[column] = (
            w_v4 * v4_aligned[column]
            + w_diag * diagnostic_aligned[column]
        )
    return add_complete_costs(combo)


def calc_metrics(ret: pd.Series) -> dict[str, Any]:
    ret = ret.dropna()
    nav = (1 + ret).cumprod()
    std = ret.std(ddof=1)
    annual_return = float(nav.iloc[-1] ** (ANN / len(ret)) - 1)
    max_drawdown = float((nav / nav.cummax() - 1).min())
    return {
        "start": ret.index.min().date().isoformat(),
        "end": ret.index.max().date().isoformat(),
        "n_days": int(len(ret)),
        "annual_return": annual_return,
        "annual_vol": float(std * np.sqrt(ANN)),
        "sharpe": (
            float(ret.mean() / std * np.sqrt(ANN))
            if std > 0
            else np.nan
        ),
        "max_drawdown": max_drawdown,
        "calmar": (
            float(annual_return / abs(max_drawdown))
            if max_drawdown < 0
            else np.nan
        ),
        "final_nav": float(nav.iloc[-1]),
    }


def return_column(cost_label: str) -> str:
    return (
        "ret_gross"
        if cost_label == "gross"
        else f"ret_{cost_label}"
    )


def metrics_table(
    strategies: dict[str, pd.DataFrame],
    correlation_by_cost: dict[str, float],
) -> pd.DataFrame:
    rows = []
    for strategy_name, daily in strategies.items():
        for cost_label in COST_LABELS:
            rows.append(
                {
                    "strategy": strategy_name,
                    "cost_label": cost_label,
                    **calc_metrics(
                        daily[return_column(cost_label)]
                    ),
                    "avg_strategy_turnover": float(
                        daily["strategy_turnover"].mean()
                    ),
                    "avg_roll_turnover_proxy": float(
                        daily["roll_turnover"].mean()
                    ),
                    "avg_gross_exposure": float(
                        daily["gross_exposure"].mean()
                    ),
                    "correlation_v4_diag": correlation_by_cost[
                        cost_label
                    ],
                }
            )
    columns = [
        "strategy",
        "cost_label",
        "annual_return",
        "annual_vol",
        "sharpe",
        "max_drawdown",
        "calmar",
        "final_nav",
        "avg_strategy_turnover",
        "avg_roll_turnover_proxy",
        "avg_gross_exposure",
        "correlation_v4_diag",
        "start",
        "end",
        "n_days",
    ]
    return pd.DataFrame(rows)[columns]


def train_validation_table(main_combo: pd.DataFrame) -> pd.DataFrame:
    periods = {
        "train_2018_2021_strict": strict_training_mask(main_combo.index),
        "validation_2022_2025": validation_mask(main_combo.index),
        "full_sample": pd.Series(True, index=main_combo.index),
    }
    rows = []
    for period_name, mask in periods.items():
        part = main_combo.loc[mask]
        for cost_label in (
            "gross",
            "2bp_strategy_only",
            "5bp_with_roll_proxy",
        ):
            rows.append(
                {
                    "strategy": "Combo train-selected 55/45",
                    "period": period_name,
                    "cost_label": cost_label,
                    "w_v4": MAIN_WEIGHTS[0],
                    "w_diag": MAIN_WEIGHTS[1],
                    **calc_metrics(
                        part[return_column(cost_label)]
                    ),
                    "avg_strategy_turnover": float(
                        part["strategy_turnover"].mean()
                    ),
                    "avg_roll_turnover_proxy": float(
                        part["roll_turnover"].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_daily_output(
    v4_daily: pd.DataFrame,
    diagnostic_daily: pd.DataFrame,
    main_combo: pd.DataFrame,
    reference_combo: pd.DataFrame,
) -> pd.DataFrame:
    index = main_combo.index
    output = pd.DataFrame(index=index)
    sources = {
        "v4": v4_daily.reindex(index),
        "diagnostic": diagnostic_daily.reindex(index),
        "combo_55_45": main_combo,
        "combo_60_40_reference": reference_combo,
    }
    for prefix, source in sources.items():
        for cost_label in COST_LABELS:
            return_name = f"{prefix}_ret_{cost_label}"
            output[return_name] = source[return_column(cost_label)]
            output[return_name.replace("_ret_", "_nav_")] = (
                1 + output[return_name]
            ).cumprod()
        output[f"{prefix}_strategy_turnover"] = source[
            "strategy_turnover"
        ]
        output[f"{prefix}_roll_turnover_proxy"] = source[
            "roll_turnover"
        ]
        output[f"{prefix}_gross_exposure"] = source[
            "gross_exposure"
        ]
    return output


def plot_nav(daily: pd.DataFrame, path: Path) -> None:
    series = {
        "V4 Gross": daily["v4_nav_gross"],
        "Diagnostic Gross": daily["diagnostic_nav_gross"],
        "Combo 55/45 Gross": daily["combo_55_45_nav_gross"],
        "Combo 55/45 Net 5bp + Roll": daily[
            "combo_55_45_nav_5bp_with_roll_proxy"
        ],
    }
    width, height = 1600, 850
    left, right, top, bottom = 120, 70, 95, 105
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = [
        (40, 150, 90),
        (220, 76, 70),
        (26, 115, 232),
        (130, 86, 180),
    ]
    ink, grid = (34, 40, 49), (222, 226, 230)
    values = (
        pd.concat(list(series.values()))
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    y_min, y_max = float(values.min()), float(values.max())
    padding = max((y_max - y_min) * 0.08, 0.02)
    y_min, y_max = y_min - padding, y_max + padding
    plot_width = width - left - right
    plot_height = height - top - bottom
    dates = daily.index

    def x_pos(position: int) -> float:
        return left + position * plot_width / max(len(dates) - 1, 1)

    def y_pos(value: float) -> float:
        return (
            top
            + (y_max - value)
            * plot_height
            / max(y_max - y_min, 1e-12)
        )

    draw.text(
        (left, 34),
        "Pure Warrant Dual Strategy - Train Selected 55/45",
        fill=ink,
        font=font,
    )
    for step in range(6):
        value = y_min + step * (y_max - y_min) / 5
        y = y_pos(value)
        draw.line((left, y, width - right, y), fill=grid, width=1)
        draw.text((24, y - 7), f"{value:.2f}", fill=ink, font=font)
        position = int(round(step * (len(dates) - 1) / 5))
        x = x_pos(position)
        draw.line(
            (x, top, x, height - bottom),
            fill=(238, 241, 244),
            width=1,
        )
        draw.text(
            (x - 34, height - bottom + 20),
            dates[position].strftime("%Y-%m"),
            fill=ink,
            font=font,
        )
    draw.rectangle(
        (left, top, width - right, height - bottom),
        outline=(170, 176, 184),
        width=1,
    )
    for position, (label, nav) in enumerate(series.items()):
        points = [
            (x_pos(i), y_pos(value))
            for i, value in enumerate(nav.to_numpy())
        ]
        draw.line(
            points, fill=colors[position], width=3, joint="curve"
        )
        legend_x = width - right - 310
        legend_y = top + 20 + position * 30
        draw.line(
            (legend_x, legend_y, legend_x + 42, legend_y),
            fill=colors[position],
            width=4,
        )
        draw.text(
            (legend_x + 52, legend_y - 7),
            label,
            fill=ink,
            font=font,
        )
    footer = (
        f"Start: {dates[0].date()}    End: {dates[-1].date()}    "
        f"Gross: {series['Combo 55/45 Gross'].iloc[-1]:.3f}    "
        "Net 5bp+Roll: "
        f"{series['Combo 55/45 Net 5bp + Roll'].iloc[-1]:.3f}"
    )
    draw.text((left, height - 48), footer, fill=ink, font=font)
    image.save(path)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    close, warrant = load_data()
    fwd_ret = close.shift(-1) / close - 1
    return_arrays = prepare_return_arrays(close)

    v4_weights = build_v4_weights(np.log1p(warrant))
    diagnostic_weights, diagnostic_symbols = build_diagnostic_weights(
        warrant, fwd_ret
    )
    v4_daily = add_complete_costs(
        strategy_daily(v4_weights, return_arrays)
    )
    diagnostic_daily = add_complete_costs(
        strategy_daily(diagnostic_weights, return_arrays)
    )
    common = v4_daily.index.intersection(diagnostic_daily.index)
    v4_daily = v4_daily.loc[common]
    diagnostic_daily = diagnostic_daily.loc[common]

    main_combo = combine_sleeves(
        v4_daily, diagnostic_daily, *MAIN_WEIGHTS
    )
    reference_combo = combine_sleeves(
        v4_daily, diagnostic_daily, *REFERENCE_WEIGHTS
    )
    correlation_by_cost = {
        label: float(
            v4_daily[return_column(label)].corr(
                diagnostic_daily[return_column(label)]
            )
        )
        for label in COST_LABELS
    }
    strategies = {
        "V4 standalone": v4_daily,
        "Diagnostic standalone": diagnostic_daily,
        "Combo train-selected 55/45": main_combo,
        "Combo full-sample reference 60/40": reference_combo,
    }
    metrics = metrics_table(strategies, correlation_by_cost)
    train_validation = train_validation_table(main_combo)
    daily = build_daily_output(
        v4_daily, diagnostic_daily, main_combo, reference_combo
    )

    plot_nav(daily, OUT / "final_dual_combo_nav.png")
    metrics.to_csv(
        OUT / "final_dual_combo_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    train_validation.to_csv(
        OUT / "final_dual_combo_train_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    daily.reset_index(names="datetime").to_csv(
        OUT / "final_dual_combo_daily.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_index = metrics.set_index(["strategy", "cost_label"])
    validation_index = train_validation.set_index(
        ["period", "cost_label"]
    )
    main_name = "Combo train-selected 55/45"
    main_gross = metric_index.loc[(main_name, "gross")]
    main_net2 = metric_index.loc[
        (main_name, "2bp_strategy_only")
    ]
    main_roll5 = metric_index.loc[
        (main_name, "5bp_with_roll_proxy")
    ]
    validation_gross = validation_index.loc[
        ("validation_2022_2025", "gross")
    ]
    validation_net2 = validation_index.loc[
        ("validation_2022_2025", "2bp_strategy_only")
    ]
    validation_roll5 = validation_index.loc[
        ("validation_2022_2025", "5bp_with_roll_proxy")
    ]
    summary = {
        "main_weight_v4": MAIN_WEIGHTS[0],
        "main_weight_diag": MAIN_WEIGHTS[1],
        "reference_weight_v4": REFERENCE_WEIGHTS[0],
        "reference_weight_diag": REFERENCE_WEIGHTS[1],
        "main_gross_sharpe": float(main_gross["sharpe"]),
        "main_net_2bp_sharpe": float(main_net2["sharpe"]),
        "main_net_5bp_roll_proxy_sharpe": float(
            main_roll5["sharpe"]
        ),
        "validation_gross_sharpe": float(
            validation_gross["sharpe"]
        ),
        "validation_net_2bp_sharpe": float(
            validation_net2["sharpe"]
        ),
        "validation_net_5bp_roll_proxy_sharpe": float(
            validation_roll5["sharpe"]
        ),
        "v4_gross_sharpe": float(
            metric_index.loc[("V4 standalone", "gross"), "sharpe"]
        ),
        "diagnostic_gross_sharpe": float(
            metric_index.loc[
                ("Diagnostic standalone", "gross"), "sharpe"
            ]
        ),
        "correlation_gross": correlation_by_cost["gross"],
        "correlation_net_2bp": correlation_by_cost[
            "2bp_strategy_only"
        ],
        "avg_strategy_turnover": float(
            main_combo["strategy_turnover"].mean()
        ),
        "avg_roll_turnover_proxy": float(
            main_combo["roll_turnover"].mean()
        ),
        "roll_proxy_is_complete": True,
        "execution_lag": EXECUTION_LAG,
        "diagnostic_training_start": TRAIN_START.date().isoformat(),
        "diagnostic_training_end": TRAIN_END.date().isoformat(),
        "diagnostic_selected_symbol_count": len(diagnostic_symbols),
        "optional_symbol_map_count": len(load_symbol_map()),
        "start": common.min().date().isoformat(),
        "end": common.max().date().isoformat(),
        "n_days": int(len(common)),
    }
    (OUT / "final_dual_combo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Final Pure Warrant Dual Combo")
    print("Main Weight: V4 55%, Diagnostic 45%")
    print(f"Gross Sharpe: {summary['main_gross_sharpe']:.2f}")
    print(f"Net 2bp Sharpe: {summary['main_net_2bp_sharpe']:.2f}")
    print(
        "Net 5bp + Roll Proxy Sharpe: "
        f"{summary['main_net_5bp_roll_proxy_sharpe']:.2f}"
    )
    print(
        "Validation Gross Sharpe: "
        f"{summary['validation_gross_sharpe']:.2f}"
    )
    print(
        "Correlation V4 vs Diagnostic: "
        f"{summary['correlation_gross']:.2f}"
    )
    print(
        "Avg Strategy Turnover: "
        f"{summary['avg_strategy_turnover']:.2%}"
    )
    print(
        "Avg Roll Proxy Turnover: "
        f"{summary['avg_roll_turnover_proxy']:.2%}"
    )


if __name__ == "__main__":
    main()
