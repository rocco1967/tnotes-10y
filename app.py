import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from catboost import CatBoostClassifier
from src.features import calculate_features, FEATURES

st.set_page_config(page_title="TNotes-10Y | Backtest CatBoost",
                   page_icon="📊", layout="wide")

st.title("TNotes-10Y — CatBoost Historical Backtest")
st.caption("Modello originale invariato | Segnale LONG/SHORT | Stop fisso 0.75 punti")

STOP = 0.75
HOLDING = 5

@st.cache_data(ttl=3600)
def download_data():
    raw = yf.download(
        "ZN=F",
        start="2000-01-01",
        interval="1d",
        auto_adjust=False,
        progress=False
    )
    if raw.empty:
        raise ValueError("Yahoo Finance non ha restituito dati.")
    return raw

def get_ohlc(raw):
    x = raw.copy()
    if isinstance(x.columns, pd.MultiIndex):
        for level in range(x.columns.nlevels):
            vals = x.columns.get_level_values(level)
            if {"Open", "High", "Low", "Close"}.issubset(set(vals)):
                x.columns = vals
                break
    x = x[["Open", "High", "Low", "Close"]].copy()
    x.columns = ["open", "high", "low", "close"]
    x.index = pd.to_datetime(x.index)
    return x.astype(float)

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("model/TNotes10Y_CatBoost.cbm")
    return model

def make_backtest(prices, probabilities, cost):
    data = prices.copy()
    data["probability_long"] = probabilities
    data["signal"] = np.where(data["probability_long"] >= 0.50, 1, -1)

    trades = []
    i = 0
    n = len(data)

    while i + HOLDING < n:
        direction = int(data["signal"].iloc[i])
        entry_i = i + 1
        last_i = i + HOLDING

        entry = float(data["open"].iloc[entry_i])
        stop_price = entry - STOP if direction == 1 else entry + STOP

        exit_i = last_i
        exit_price = float(data["close"].iloc[last_i])
        reason = "Scadenza 5 sessioni"

        for j in range(entry_i, last_i + 1):
            low = float(data["low"].iloc[j])
            high = float(data["high"].iloc[j])

            if direction == 1 and low <= stop_price:
                exit_i = j
                exit_price = stop_price
                reason = "Stop"
                break

            if direction == -1 and high >= stop_price:
                exit_i = j
                exit_price = stop_price
                reason = "Stop"
                break

        gross = direction * (exit_price - entry)
        net = gross - cost

        trades.append({
            "Signal date": data.index[i],
            "Entry date": data.index[entry_i],
            "Exit date": data.index[exit_i],
            "Direction": "LONG" if direction == 1 else "SHORT",
            "Probability LONG": data["probability_long"].iloc[i],
            "Entry": entry,
            "Exit": exit_price,
            "Reason": reason,
            "Gross PnL": gross,
            "Net PnL": net
        })

        # Nessuna posizione sovrapposta
        i = exit_i

    trades = pd.DataFrame(trades)

    daily = pd.Series(0.0, index=data.index)
    if not trades.empty:
        pnl_by_date = trades.groupby("Exit date")["Net PnL"].sum()
        daily.loc[pnl_by_date.index] = pnl_by_date

    equity = daily.cumsum()
    drawdown = equity - equity.cummax()

    return data, trades, equity, drawdown

try:
    with st.spinner("Scaricamento dati e calcolo delle predizioni..."):
        raw = download_data()
        prices = get_ohlc(raw)

        feature_data = calculate_features(raw)
        X = feature_data[FEATURES].dropna()

        model = load_model()
        probabilities = model.predict_proba(X)[:, 1]

        prices = prices.reindex(X.index).dropna()
        probabilities = pd.Series(probabilities, index=X.index).reindex(prices.index)

        valid = probabilities.notna()
        prices = prices.loc[valid]
        probabilities = probabilities.loc[valid]

        cost = st.sidebar.number_input(
            "Costo round-trip (punti)",
            min_value=0.0,
            value=0.0,
            step=0.01
        )

        predictions, trades, equity, drawdown = make_backtest(
            prices, probabilities, cost
        )

    st.success(
        f"Dati: {prices.index.min().date()} → {prices.index.max().date()} "
        f"| {len(prices)} sessioni"
    )

    if trades.empty:
        st.warning("Nessun trade generato.")
    else:
        total = trades["Net PnL"].sum()
        max_dd = drawdown.min()
        win_rate = (trades["Net PnL"] > 0).mean() * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Trade", f"{len(trades)}")
        c2.metric("P&L netto (punti)", f"{total:.2f}")
        c3.metric("Max drawdown (punti)", f"{max_dd:.2f}")
        c4.metric("Trade positivi", f"{win_rate:.1f}%")

        st.subheader("Equity curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=equity.index, y=equity.values,
            mode="lines", name="Equity netta"
        ))
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Punti",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Drawdown")
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            mode="lines", fill="tozeroy", name="Drawdown"
        ))
        fig_dd.update_layout(xaxis_title="Data", yaxis_title="Punti")
        st.plotly_chart(fig_dd, use_container_width=True)
        st.write("Distribuzione probabilità LONG")
        st.write(predictions["probability_long"].describe())
        st.write("Valori distinti:", predictions["probability_long"].nunique())

        st.subheader("Probabilità LONG")
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=predictions.index,
            y=predictions["probability_long"],
            mode="lines", name="Probabilità LONG"
        ))
        fig_p.add_hline(y=0.50, line_dash="dash")
        fig_p.update_layout(
            xaxis_title="Data", yaxis_title="Probabilità",
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig_p, use_container_width=True)

        st.subheader("Ultimi trade")
        st.dataframe(trades.tail(30), use_container_width=True)

        st.download_button(
            "Scarica tutti i trade CSV",
            trades.to_csv(index=False).encode("utf-8"),
            "TNotes10Y_trades.csv",
            "text/csv"
        )

        equity_csv = pd.DataFrame({
            "date": equity.index,
            "equity": equity.values,
            "drawdown": drawdown.values
        })
        st.download_button(
            "Scarica equity CSV",
            equity_csv.to_csv(index=False).encode("utf-8"),
            "TNotes10Y_equity.csv",
            "text/csv"
        )

    st.caption(
        "Nota: le predizioni anteriori al periodo di addestramento sono "
        "retrospettive, non costituiscono risultati out-of-sample."
    )

except Exception as e:
    st.error(f"Errore: {e}")
    st.exception(e)

   
   
