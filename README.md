# TNotes-10Y

Machine learning-based trading model for 10-Year US Treasury Note futures (ZN).

## Model

- Algorithm: CatBoostClassifier
- Trees: 1
- Depth: 5
- Learning rate: 0.03
- L2 leaf regularization: 5
- Random seed: 42
- Prediction horizon: 5 trading sessions
- Features: 13 technical and volatility features

## Data

Historical daily OHLC data for Yahoo Finance ticker `ZN=F`.

## Application

The Streamlit application provides:
- Latest ZN market price
- LONG / SHORT directional signal
- Model probability
- Historical price chart
- Model feature values

## Project structure

```text
tnotes-10y/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   ├── TNotes10Y_CatBoost.cbm
│   └── TNotes10Y_config.json
└── src/
    ├── __init__.py
    ├── features.py
    └── predictions.py
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Backtest

Out-of-sample test period: 2023–2026.

- Trades: 181
- Gross P&L: +11.5563 points
- Maximum closed-trade drawdown: -7.4375 points
- Win rate: 51.38%

Net results depend on transaction costs and execution assumptions.

## Disclaimer

This project is for research and educational purposes only. It is not investment advice or a solicitation to trade futures.
