"""Reproduce the frozen 55/45 sleeve-weight selection from tracked code.

The selection criterion is gross Sharpe on the strict 2018-2021 training
returns.  A return is training-eligible only when its next-trading-day
endpoint is no later than 2021-12-31.  The later 5bp + monthly roll-proxy
weight plateau is a post-freeze robustness check, not a re-selection step.
"""


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import formal_strategy as formal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "results" / "weight_selection_v1"
ANN = 252
TRAIN_START = formal.TRAIN_START
TRAIN_END = formal.TRAIN_END
VALIDATION_START = formal.VALIDATION_START
VALIDATION_END = formal.VALIDATION_END
SELECTION_RETURN_LABEL = "gross"
SELECTION_METRIC = "gross_sharpe"
FROZEN_V4_WEIGHT = 0.55
FROZEN_DIAGNOSTIC_WEIGHT = 0.45
WEIGHT_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
COST_LABELS = (
    "gross",
    "net_2bp_strategy_only",
    "net_5bp_strategy_only",
    "net_10bp_strategy_only",
    "net_2bp_with_roll_proxy",
    "net_5bp_with_roll_proxy",
)


def pick_return_column(
    frame: pd.DataFrame, cost_label: str
) -> tuple[str, bool]:
    candidates = {
        "gross": ["ret_gross"],
        "net_2bp_strategy_only": [
            "ret_2bp_strategy_only",
            "ret_net_2bp_strategy_only",
            "ret_net_2bp",
        ],
        "net_5bp_strategy_only": [
            "ret_5bp_strategy_only",
            "ret_net_5bp_strategy_only",
            "ret_net_5bp",
        ],
        "net_10bp_strategy_only": [
            "ret_10bp_strategy_only",
            "ret_net_10bp_strategy_only",
            "ret_net_10bp",
        ],
        "net_2bp_with_roll_proxy": [
            "ret_2bp_with_roll_proxy",
            "ret_net_2bp_with_roll_proxy",
            "ret_net_2bp_strategy_only",
            "ret_net_2bp",
        ],
        "net_5bp_with_roll_proxy": [
            "ret_5bp_with_roll_proxy",
            "ret_net_5bp_with_roll_proxy",
            "ret_net_5bp_strategy_only",
            "ret_net_5bp",
        ],
    }
    if cost_label not in candidates:
        raise KeyError(f"Unsupported cost label: {cost_label}")
    for column in candidates[cost_label]:
        if column in frame.columns:
            is_roll_fallback = (
                "with_roll_proxy" in cost_label and "with_roll_proxy" not in column
            )
            return column, is_roll_fallback
    raise KeyError(
        f"No return column for {cost_label}. Available columns: {list(frame.columns)}"
    )


def align_returns(
    v4: pd.DataFrame, diagnostic: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    aligned: dict[str, pd.DataFrame] = {}
    fallback: dict[str, dict[str, Any]] = {}
    for cost_label in COST_LABELS:
        v4_column, v4_fallback = pick_return_column(v4, cost_label)
        diag_column, diag_fallback = pick_return_column(diagnostic, cost_label)
        pair = pd.concat(
            [
                v4[v4_column].rename("v4"),
                diagnostic[diag_column].rename("diagnostic"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        aligned[cost_label] = pair
        fallback[cost_label] = {
            "v4_column": v4_column,
            "diagnostic_column": diag_column,
            "v4_fallback": v4_fallback,
            "diagnostic_fallback": diag_fallback,
        }
    return aligned, fallback


def build_formal_sleeve_daily() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild both frozen sleeves directly from the formal tracked code."""
    close, warrant = formal.load_data()
    fwd_ret = close.shift(-1) / close - 1
    return_arrays = formal.prepare_return_arrays(close)

    v4_weights = formal.build_v4_weights(np.log1p(warrant))
    diagnostic_weights, _ = formal.build_diagnostic_weights(
        warrant, fwd_ret
    )
    v4_daily = formal.add_complete_costs(
        formal.strategy_daily(v4_weights, return_arrays)
    )
    diagnostic_daily = formal.add_complete_costs(
        formal.strategy_daily(diagnostic_weights, return_arrays)
    )
    common = v4_daily.index.intersection(diagnostic_daily.index)
    return v4_daily.loc[common], diagnostic_daily.loc[common]


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
        "sharpe": float(ret.mean() / std * np.sqrt(ANN)) if std > 0 else np.nan,
        "max_drawdown": max_drawdown,
        "calmar": float(annual_return / abs(max_drawdown))
        if max_drawdown < 0 else np.nan,
        "final_nav": float(nav.iloc[-1]),
        "win_rate": float((ret > 0).mean()),
    }


def fixed_weight_leaderboard(
    aligned: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    weights = WEIGHT_GRID
    rows = []
    for cost_label, pair in aligned.items():
        returns = (
            pair["v4"].to_numpy()[:, None] * weights[None, :]
            + pair["diagnostic"].to_numpy()[:, None] * (1 - weights)[None, :]
        )
        for position, weight in enumerate(weights):
            combo = pd.Series(returns[:, position], index=pair.index)
            rows.append(
                {
                    "method": "fixed_weight_grid",
                    "cost_label": cost_label,
                    "w_v4": float(weight),
                    "w_diag": float(1 - weight),
                    "correlation_between_substrategies": float(
                        pair["v4"].corr(pair["diagnostic"])
                    ),
                    **calc_metrics(combo),
                }
            )
    return pd.DataFrame(rows)


def rolling_inverse_vol_weights(gross_pair: pd.DataFrame) -> pd.DataFrame:
    # shift(1) guarantees the weight used on day t only sees returns through t-1.
    vol = gross_pair.rolling(252, min_periods=126).std().shift(1)
    inverse = 1.0 / vol.replace(0, np.nan)
    raw_v4 = (inverse["v4"] / inverse.sum(axis=1)).clip(0.2, 0.8)
    rebalance = np.arange(len(raw_v4)) % 20 == 0
    w_v4 = raw_v4.where(rebalance).ffill().fillna(0.5)
    return pd.DataFrame({"w_v4": w_v4, "w_diag": 1 - w_v4})


def rolling_inverse_vol_leaderboard(
    aligned: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for cost_label, pair in aligned.items():
        current_weights = weights.reindex(pair.index).ffill().fillna(0.5)
        combo = (
            current_weights["w_v4"] * pair["v4"]
            + current_weights["w_diag"] * pair["diagnostic"]
        )
        rows.append(
            {
                "method": "rolling_inverse_vol",
                "cost_label": cost_label,
                "w_v4": float(current_weights["w_v4"].mean()),
                "w_diag": float(current_weights["w_diag"].mean()),
                "correlation_between_substrategies": float(
                    pair["v4"].corr(pair["diagnostic"])
                ),
                **calc_metrics(combo),
            }
        )
    return pd.DataFrame(rows)


def select_training_weight(
    gross_pair: pd.DataFrame,
) -> tuple[float, pd.Series]:
    """Maximize gross Sharpe on the strict 2018-2021 training return set."""
    train_pair = gross_pair.loc[formal.strict_training_mask(gross_pair.index)]
    train_returns = (
        train_pair["v4"].to_numpy()[:, None] * WEIGHT_GRID[None, :]
        + train_pair["diagnostic"].to_numpy()[:, None]
        * (1 - WEIGHT_GRID)[None, :]
    )
    train_sharpes = (
        np.nanmean(train_returns, axis=0)
        / np.nanstd(train_returns, axis=0, ddof=1)
        * np.sqrt(ANN)
    )
    sharpe_by_weight = pd.Series(train_sharpes, index=WEIGHT_GRID)
    selected_weight = float(sharpe_by_weight.idxmax())
    return selected_weight, sharpe_by_weight


def train_selected_fixed_weight(
    aligned: dict[str, pd.DataFrame],
) -> tuple[float, pd.DataFrame]:
    """Apply the frozen training-only weight selection and report each period."""
    gross = aligned[SELECTION_RETURN_LABEL]
    selected_weight, _ = select_training_weight(gross)
    if not np.isclose(selected_weight, FROZEN_V4_WEIGHT):
        raise RuntimeError(
            "Training selection no longer reproduces frozen 55/45 weights: "
            f"selected {selected_weight:.2f}/{1 - selected_weight:.2f}"
        )

    rows = []
    for cost_label, pair in aligned.items():
        combo = (
            selected_weight * pair["v4"]
            + (1 - selected_weight) * pair["diagnostic"]
        )
        periods = {
            "train_2018_2021_strict": formal.strict_training_mask(combo.index),
            "validation_2022_2025": formal.validation_mask(combo.index),
            "full_sample": pd.Series(True, index=combo.index),
        }
        for period_name, mask in periods.items():
            part = combo.loc[mask]
            rows.append(
                {
                    "method": "train_selected_fixed_weight",
                    "selection_metric": SELECTION_METRIC,
                    "selection_return_label": SELECTION_RETURN_LABEL,
                    "period": period_name,
                    "cost_label": cost_label,
                    "w_v4": selected_weight,
                    "w_diag": 1 - selected_weight,
                    "correlation_between_substrategies": float(
                        pair.loc[part.index, "v4"].corr(
                            pair.loc[part.index, "diagnostic"]
                        )
                    ),
                    **calc_metrics(part),
                }
            )
    return selected_weight, pd.DataFrame(rows)


def combo_return(
    pair: pd.DataFrame,
    method: str,
    w_v4: float,
    rolling_weights: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    if method == "rolling_inverse_vol":
        weights = rolling_weights.reindex(pair.index).ffill().fillna(0.5)
    else:
        weights = pd.DataFrame(
            {"w_v4": w_v4, "w_diag": 1 - w_v4}, index=pair.index
        )
    ret = weights["w_v4"] * pair["v4"] + weights["w_diag"] * pair["diagnostic"]
    return ret, weights


def select_view(
    leaderboard: pd.DataFrame, cost_label: str
) -> pd.Series:
    return (
        leaderboard.loc[leaderboard["cost_label"] == cost_label]
        .sort_values("sharpe", ascending=False)
        .iloc[0]
    )


def build_best_daily(
    aligned: dict[str, pd.DataFrame],
    best_gross: pd.Series,
    rolling_weights: pd.DataFrame,
) -> pd.DataFrame:
    gross_pair = aligned["gross"]
    gross_ret, weights = combo_return(
        gross_pair,
        str(best_gross["method"]),
        float(best_gross["w_v4"]),
        rolling_weights,
    )
    net2_pair = aligned["net_2bp_strategy_only"].reindex(gross_pair.index)
    net2_ret, _ = combo_return(
        net2_pair,
        str(best_gross["method"]),
        float(best_gross["w_v4"]),
        rolling_weights,
    )
    roll5_pair = aligned["net_5bp_with_roll_proxy"].reindex(gross_pair.index)
    roll5_ret, _ = combo_return(
        roll5_pair,
        str(best_gross["method"]),
        float(best_gross["w_v4"]),
        rolling_weights,
    )
    daily = pd.DataFrame(
        {
            "v4_ret_gross": gross_pair["v4"],
            "diag_ret_gross": gross_pair["diagnostic"],
            "combo_ret_gross": gross_ret,
            "combo_ret_net_2bp": net2_ret,
            "combo_ret_net_5bp_roll_proxy": roll5_ret,
            "w_v4": weights["w_v4"],
            "w_diag": weights["w_diag"],
            "rolling_w_v4": rolling_weights.reindex(gross_pair.index)["w_v4"],
            "rolling_w_diag": rolling_weights.reindex(gross_pair.index)["w_diag"],
        }
    ).dropna()
    daily["combo_nav_gross"] = (1 + daily["combo_ret_gross"]).cumprod()
    daily["combo_nav_net_2bp"] = (1 + daily["combo_ret_net_2bp"]).cumprod()
    daily["combo_nav_net_5bp_roll_proxy"] = (
        1 + daily["combo_ret_net_5bp_roll_proxy"]
    ).cumprod()
    return daily[
        [
            "v4_ret_gross",
            "diag_ret_gross",
            "combo_ret_gross",
            "combo_nav_gross",
            "combo_ret_net_2bp",
            "combo_nav_net_2bp",
            "combo_ret_net_5bp_roll_proxy",
            "combo_nav_net_5bp_roll_proxy",
            "w_v4",
            "w_diag",
            "rolling_w_v4",
            "rolling_w_diag",
        ]
    ]


def plot_nav_compare(daily: pd.DataFrame, path: Path) -> None:
    series = {
        "V4 Gross": (1 + daily["v4_ret_gross"]).cumprod(),
        "Diagnostic Gross": (1 + daily["diag_ret_gross"]).cumprod(),
        "Best Combo Gross": daily["combo_nav_gross"],
    }
    width, height = 1600, 850
    left, right, top, bottom = 120, 70, 95, 105
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = [(40, 150, 90), (220, 76, 70), (26, 115, 232)]
    ink, grid = (34, 40, 49), (222, 226, 230)
    values = pd.concat(list(series.values())).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    y_min, y_max = float(values.min()), float(values.max())
    pad = max((y_max - y_min) * 0.08, 0.02)
    y_min, y_max = y_min - pad, y_max + pad
    plot_width = width - left - right
    plot_height = height - top - bottom
    dates = daily.index

    def x_pos(index: int) -> float:
        return left + index * plot_width / max(len(dates) - 1, 1)

    def y_pos(value: float) -> float:
        return top + (y_max - value) * plot_height / max(y_max - y_min, 1e-12)

    draw.text((left, 34), "Pure Warrant Dual Strategy Combination", fill=ink, font=font)
    for step in range(6):
        value = y_min + step * (y_max - y_min) / 5
        y = y_pos(value)
        draw.line((left, y, width - right, y), fill=grid, width=1)
        draw.text((24, y - 7), f"{value:.2f}", fill=ink, font=font)
        position = int(round(step * (len(dates) - 1) / 5))
        x = x_pos(position)
        draw.line((x, top, x, height - bottom), fill=(238, 241, 244), width=1)
        draw.text((x - 34, height - bottom + 20), dates[position].strftime("%Y-%m"), fill=ink, font=font)
    draw.rectangle((left, top, width - right, height - bottom), outline=(170, 176, 184), width=1)

    for position, (label, nav) in enumerate(series.items()):
        points = [(x_pos(i), y_pos(value)) for i, value in enumerate(nav.to_numpy())]
        draw.line(points, fill=colors[position], width=3, joint="curve")
        legend_x = width - right - 250
        legend_y = top + 20 + position * 30
        draw.line((legend_x, legend_y, legend_x + 42, legend_y), fill=colors[position], width=4)
        draw.text((legend_x + 52, legend_y - 7), label, fill=ink, font=font)
    image.save(path)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    v4, diagnostic = build_formal_sleeve_daily()
    aligned, fallbacks = align_returns(v4, diagnostic)
    gross_pair = aligned["gross"]

    # The leaderboard is a retained full-sample exploratory diagnostic. It is
    # never consulted by select_training_weight or the frozen config contract.
    fixed = fixed_weight_leaderboard(aligned)
    rolling_weights = rolling_inverse_vol_weights(gross_pair)
    rolling = rolling_inverse_vol_leaderboard(aligned, rolling_weights)
    leaderboard = pd.concat([fixed, rolling], ignore_index=True)
    selected_weight, train_validation = train_selected_fixed_weight(aligned)

    best_gross = select_view(leaderboard, "gross")
    best_net2 = select_view(leaderboard, "net_2bp_strategy_only")
    best_practical = select_view(leaderboard, "net_5bp_with_roll_proxy")
    validation_row = train_validation.loc[
        (train_validation["period"] == "validation_2022_2025")
        & (train_validation["cost_label"] == "gross")
    ].iloc[0]
    best_daily = build_best_daily(aligned, best_gross, rolling_weights)
    plot_nav_compare(best_daily, OUT / "dual_combo_nav_compare.png")

    leaderboard.to_csv(
        OUT / "dual_combo_leaderboard.csv", index=False, encoding="utf-8-sig"
    )
    train_validation.to_csv(
        OUT / "dual_combo_train_validation.csv", index=False, encoding="utf-8-sig"
    )
    best_daily.reset_index(names="datetime").to_csv(
        OUT / "dual_combo_best_daily.csv", index=False, encoding="utf-8-sig"
    )

    v4_metrics = calc_metrics(gross_pair["v4"])
    diag_metrics = calc_metrics(gross_pair["diagnostic"])
    correlation_gross = float(gross_pair["v4"].corr(gross_pair["diagnostic"]))
    correlation_net2 = float(
        aligned["net_2bp_strategy_only"]["v4"].corr(
            aligned["net_2bp_strategy_only"]["diagnostic"]
        )
    )
    reached_two = bool(best_gross["sharpe"] >= 2.0)
    if reached_two:
        judgment = (
            "Combining long-horizon warrant scarcity and short-cycle warrant shock "
            "improves the pure warrant strategy to gross Sharpe above 2, but validation "
            "and cost-stress results should be checked before practical use."
        )
    elif best_gross["sharpe"] > v4_metrics["sharpe"]:
        judgment = (
            "Strategy-level combination improves over the single V4 baseline, but "
            "does not reach gross Sharpe 2 under the current data and lag1 rules."
        )
    else:
        judgment = (
            "The short-cycle diagnostic branch does not add enough independent alpha; "
            "the stable V4 long-horizon warrant scarcity strategy remains the better "
            "standalone version."
        )

    summary = {
        "sample_start": gross_pair.index.min().date().isoformat(),
        "sample_end": gross_pair.index.max().date().isoformat(),
        "selection_metric": SELECTION_METRIC,
        "selection_return_label": SELECTION_RETURN_LABEL,
        "selection_period": "2018-01-01 through 2021-12-31 strict endpoint",
        "selection_grid_step": 0.05,
        "selection_inputs": "tracked formal sleeve builders",
        "formal_parameter_source": "strict-train gross Sharpe grid only",
        "exploratory_leaderboard_scope": (
            "full sample descriptive only; does not select or overwrite v1.0"
        ),
        "frozen_weight_v4": FROZEN_V4_WEIGHT,
        "frozen_weight_diagnostic": FROZEN_DIAGNOSTIC_WEIGHT,
        "official_cost_plateau_role": "post-freeze robustness only",
        "common_trading_days": int(len(gross_pair)),
        "best_gross_method": str(best_gross["method"]),
        "best_gross_sharpe": float(best_gross["sharpe"]),
        "best_gross_weight_v4": float(best_gross["w_v4"]),
        "best_gross_weight_diag": float(best_gross["w_diag"]),
        "best_net2bp_method": str(best_net2["method"]),
        "best_net2bp_sharpe": float(best_net2["sharpe"]),
        "best_practical_method": str(best_practical["method"]),
        "best_roll_proxy_sharpe": float(best_practical["sharpe"]),
        "best_practical_weight_v4": float(best_practical["w_v4"]),
        "best_practical_weight_diag": float(best_practical["w_diag"]),
        "train_selected_weight": selected_weight,
        "train_selected_weight_diag": 1 - selected_weight,
        "validation_sharpe": float(validation_row["sharpe"]),
        "best_validation_method": "train_selected_fixed_weight",
        "best_validation_weight_v4": selected_weight,
        "best_validation_weight_diag": 1 - selected_weight,
        "v4_gross_sharpe": float(v4_metrics["sharpe"]),
        "diagnostic_gross_sharpe": float(diag_metrics["sharpe"]),
        "correlation_gross": correlation_gross,
        "correlation_net_2bp": correlation_net2,
        "reached_gross_sharpe_2": reached_two,
        "return_column_fallbacks": fallbacks,
        "research_judgment": judgment,
    }
    (OUT / "dual_combo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"V4 gross Sharpe: {summary['v4_gross_sharpe']:.2f}")
    print(f"Diagnostic gross Sharpe: {summary['diagnostic_gross_sharpe']:.2f}")
    print(f"Gross correlation: {correlation_gross:.3f}")
    print(
        "Exploratory full-sample best gross Sharpe: "
        f"{summary['best_gross_sharpe']:.2f}"
    )
    print(
        "Exploratory full-sample best net 2bp Sharpe: "
        f"{summary['best_net2bp_sharpe']:.2f}"
    )
    print(
        "Exploratory full-sample best roll-proxy Sharpe: "
        f"{summary['best_roll_proxy_sharpe']:.2f}"
    )
    print(f"Gross Sharpe >= 2: {reached_two}")


if __name__ == "__main__":
    main()
