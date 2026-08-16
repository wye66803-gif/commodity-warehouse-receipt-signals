# Methodology

## Research question

Can changes in standardized commodity warehouse receipts—an observable proxy for deliverable inventory tightness or abundance—support a diversified, causal cross-sectional futures signal after conservative turnover and roll-cost stress?

## Frozen v1.0 design

The portfolio combines two warehouse-receipt-only sleeves:

- **V4 (55%)**: a slower scarcity sleeve based on long-horizon receipt levels, cross-sectional ranks, and a market-state gate.
- **Diagnostic (45%)**: a faster sleeve based on standardized 10-day receipt changes. Its eligible symbol set is learned only from lagged score/return relationships inside the strict 2018–2021 training interval and then frozen for validation.

Prices are used only to calculate forward returns and volatility diagnostics, never as the economic signal. Each sleeve forms cross-sectional long/short allocations. The V4 sleeve rebalances weights every 20 trading days and updates its state scale every 63 trading days; the Diagnostic sleeve rebalances every 3 trading days. Formal absolute symbol caps and the 55/45 sleeve mix are recorded in `configs/v1.0.json`.

## Causality and timing

Signals and target weights formed on date *t* are shifted by one trading row before exposure is applied. The resulting executed weights earn the supplied close-to-close forward return for that execution row. Rebalance values are sampled only on scheduled rows and forward-filled; there is no backward fill.

Strict training performance requires both the return-start date and its next-trading-day endpoint to be no later than 2021-12-31. The last eligible return start in the formal data is 2021-12-30. The 2022–2025 interval is frozen validation/out-of-sample relative to parameter selection. Full-sample statistics combine the in-sample training interval and this validation interval and must not be read as wholly out-of-sample.

## Parameter provenance

The sleeve mix is selected on a 5 percentage-point grid from 0/100 to 100/0 using **gross Sharpe on strict 2018–2021 training returns**. This procedure selects 55% V4 / 45% Diagnostic and then freezes it. Validation and full-sample performance do not participate in this selection.

The later weight-plateau exercise evaluates neighboring frozen combinations under the official **5bp strategy cost plus monthly roll-turnover proxy**. It is post-freeze robustness evidence, not a second optimization. The `dual_combo_leaderboard`, `dual_combo_best_daily`, and comparison chart in `results/weight_selection_v1/` are explicitly full-sample exploratory diagnostics; they neither define nor overwrite the frozen 55/45 contract.

## Return and risk definitions

All definitions below describe the implementation in `src/formal_strategy.py`:

- **Instrument forward return:** `close[t+1] / close[t] - 1` on the supplied trading-row index.
- **Daily portfolio gross return:** sum of executed lag-1 symbol weights times the same row's instrument forward returns. Rows are kept when at least one instrument has a valid forward return; a missing symbol return contributes zero through `numpy.nansum`.
- **Transaction cost:** on each daily row, strategy turnover multiplied by the stated basis-point rate is subtracted from gross return.
- **Strategy turnover:** the sum of absolute changes in executed symbol weights versus the preceding row; it is not divided by two.
- **Monthly roll-turnover proxy:** gross exposure is charged once on the first available row of each calendar month. The official return subtracts 5bp times `(strategy turnover + roll-turnover proxy)`.
- **NAV:** cumulative product of `1 + daily return`, with an implicit starting value of 1.
- **Annualized return:** geometric 252-trading-day annualization, `final_NAV ** (252 / N) - 1`.
- **Annualized volatility:** sample standard deviation of daily returns (`ddof=1`) times `sqrt(252)`.
- **Sharpe:** arithmetic mean daily return divided by sample daily volatility, times `sqrt(252)`; the risk-free rate is assumed to be zero.
- **Maximum drawdown:** minimum of `NAV / cumulative_max(NAV) - 1` over the evaluated period.

## Reporting convention

The headline convention is `5bp_with_roll_proxy`. Gross and strategy-only alternatives remain in the formal artifact tables for transparency. Costs are applied to the same daily return series before NAV and period metrics are computed.
