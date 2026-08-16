# Limitations

- Warehouse receipts measure registered or deliverable inventory, not total economic inventory. Exchange rules, delivery incentives and registration behavior can change their relationship to scarcity.
- Standardized data availability does not by itself prove point-in-time publication timestamps. The one-day execution lag is a conservative causal convention, not a full vendor-timestamp audit.
- The full-sample result includes the 2018–2021 in-sample parameter-selection interval. Only 2022–2025 is frozen validation relative to the documented selection process.
- Returns use the supplied standardized close series. The monthly roll-turnover term is a proxy rather than a contract-by-contract execution simulation.
- The full historical input is not redistributed. Public CI and tests therefore validate code contracts with synthetic fixtures rather than rerunning the private empirical dataset.
- The continuous futures series was constructed by the research instructor from CTP market data using a custom dominant-contract stitching process. Dominant-contract switches may differ by roughly one or two trading days across providers.
- Exact regeneration of the frozen empirical artifacts requires data construction matching the private research input. This publication constraint does not alter the original frozen results.
- Missing per-symbol forward returns contribute zero when another symbol keeps the portfolio row valid. This is deterministic and non-forward-looking, but may differ from production tradability handling.
- Reported costs are stylized and exclude market impact, capacity constraints, margin financing, exchange fees, taxes and broker-specific slippage.
- The finite 2018–2021 selection window means a discrete grid choice can still reflect sampling variation. The sleeve-weight plateau reduces, but does not eliminate, parameter-mining risk and does not establish robustness of every strategy parameter.
- Results are a historical backtest, not investable performance or a forecast. Symbol contribution is arithmetic attribution and does not allocate portfolio-level costs to individual symbols.
