# tnotes-10y
Machine learning-based trading model for 10-Year US Treasury Note futures (ZN), featuring CatBoost classification, historical backtesting, risk management and out-of-sample performance analysis (2023–2026).


# TNotes-10Y

### Machine Learning Trading Model for 10-Year US Treasury Note Futures (ZN)

TNotes-10Y is a quantitative trading model designed to identify directional opportunities in 10-Year US Treasury Note futures using machine learning.

The strategy uses a CatBoost classifier to estimate market direction and generate systematic long/short trading signals.

## Strategy Overview

- **Market:** 10-Year US Treasury Note Futures (ZN)
- **Model:** CatBoost Classifier
- **Trading approach:** Long / Short
- **Prediction horizon:** 5 trading sessions
- **Execution:** Next session Open
- **Risk management:** Fixed 0.75-point stop loss
- **Position management:** Maximum 5 bars, no overlapping trades

## Model Configuration

The model uses 13 technical features:

- Historical returns: 1, 2, 5, 10 and 20 sessions
- Historical volatility: 5, 10 and 20 sessions
- ATR 14 percentage
- Distance from 10-session and 20-session highs and lows

Model parameters:

- Trees: 1
- Maximum depth: 5
- Learning rate: 0.03
- L2 regularization: 5
- Random seed: 42

## Research Methodology

The historical dataset is divided chronologically to avoid look-ahead bias:

| Dataset | Period | Purpose |
|---|---|---|
| Training | 2010–2019 | Model training |
| Validation | 2020–2022 | Out-of-sample validation |
| Test | 2023–2026 | Final out-of-sample evaluation |

## Out-of-Sample Results

**Test period: 2023–2026**

| Metric | Result |
|---|---:|
| Total trades | 181 |
| Gross return | +11.5563 points |
| Maximum drawdown | -7.4375 points |
| Winning trades | 51.38% |

### Estimated Transaction Costs

| Cost scenario | Net P&L |
|---|---:|
| Optimistic | +$8,127.20 |
| Intermediate | +$5,299.10 |
| Prudent | +$2,471.00 |

Calculations use a $1,000 multiplier per point.

## Validation Results

The 2020–2022 validation period was also evaluated independently.

- Trades: 154
- Gross result: +0.8125 points
- Maximum drawdown: -10.09 points

The validation results are reported separately from the final test period.

## Risk Management

The strategy uses a fixed stop-loss mechanism of 0.75 points, with a maximum holding period of five bars.

Transaction costs are considered separately to provide a more realistic assessment of potential performance.

## Disclaimer

This repository is intended for quantitative research and educational purposes.

Historical backtest results do not guarantee future performance. Actual trading results may differ due to execution costs, slippage, liquidity, market conditions and other factors.

---

**Project:** TNotes-10Y  
**Instrument:** US 10-Year Treasury Note Futures (ZN)  
**Approach:** Systematic Machine Learning Trading
