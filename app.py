
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import json
import base64
from datetime import datetime
from zoneinfo import ZoneInfo
from catboost import CatBoostClassifier

from src.features import calculate_features, FEATURES

st.set_page_config(page_title="TNotes-10Y", page_icon="📊", layout="wide")
st.title("TNotes-10Y")
st.caption("CatBoost | Paper trading | Yahoo Finance")

STOP = 0.75
HOLDING = 5
COST = 0.03

# ---------------------------------------------------------
# GITHUB CONFIGURATION
# ---------------------------------------------------------

TOKEN = st.secrets["GITHUB_TOKEN"]
REPO = st.secrets["GITHUB_REPO"]
BRANCH = st.secrets.get("GITHUB_BRANCH", "main")
STATE_PATH = "data/live_state.json"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json"
}

def github_url():
    return f"https://api.github.com/repos/{REPO}/contents/{STATE_PATH}"

def empty_state():
    return {
        "last_processed_date": None,
        "pending_signal": None,
        "position": None,
        "trades": []
    }

def load_state():
    r = requests.get(
        github_url(),
        headers=HEADERS,
        params={"ref": BRANCH},
        timeout=20
    )

    if r.status_code == 404:
        return empty_state(), None

    r.raise_for_status()
    obj = r.json()
    decoded = base64.b64decode(obj["content"]).decode("utf-8")
    return json.loads(decoded), obj["sha"]

def save_state(state, sha):
    encoded = base64.b64encode(
        json.dumps(state, indent=2, ensure_ascii=False).encode("utf-8")
    ).decode("utf-8")

    payload = {
        "message": "Aggiornamento paper trading TNotes-10Y",
        "content": encoded,
        "branch": BRANCH
    }

    if sha:
        payload["sha"] = sha

    r = requests.put(
        github_url(),
        headers=HEADERS,
        json=payload,
        timeout=20
    )
    r.raise_for_status()

# ---------------------------------------------------------
# MARKET DATA AND MODEL
# ---------------------------------------------------------

@st.cache_data(ttl=900)
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

    idx = pd.to_datetime(x.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)

    x.index = idx.normalize()
    return x.astype(float).sort_index()

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("model/TNotes10Y_CatBoost.cbm")
    return model

# ---------------------------------------------------------
# ORIGINAL OOS BACKTEST LOGIC
# ---------------------------------------------------------

def make_backtest(prices, probabilities):
    data = prices.copy()
    data["probability_long"] = probabilities

    trades = []
    i = 0
    n = len(data)

    while i < n - 1:
        entry_i = i + 1
        side = 1 if data["probability_long"].iloc[i] >= 0.50 else -1

        entry = float(data["open"].iloc[entry_i])
        stop_price = entry - side * STOP

        exit_price = None
        reason = None

        for j in range(entry_i, min(entry_i + HOLDING, n)):
            o = float(data["open"].iloc[j])
            h = float(data["high"].iloc[j])
            l = float(data["low"].iloc[j])

            if side == 1:
                if o <= stop_price:
                    exit_price, reason = o, "Stop gap"
                elif l <= stop_price:
                    exit_price, reason = stop_price, "Stop"
            else:
                if o >= stop_price:
                    exit_price, reason = o, "Stop gap"
                elif h >= stop_price:
                    exit_price, reason = stop_price, "Stop"

            if exit_price is not None:
                exit_i = j
                break
        else:
            exit_i = min(entry_i + HOLDING - 1, n - 1)
            exit_price = float(data["close"].iloc[exit_i])
            reason = "Scadenza 5 sessioni"

        gross = side * (exit_price - entry)
        net = gross - COST

        trades.append({
            "Signal date": data.index[i],
            "Entry date": data.index[entry_i],
            "Exit date": data.index[exit_i],
            "Direction": "LONG" if side == 1 else "SHORT",
            "Entry": entry,
            "Exit": exit_price,
            "Reason": reason,
            "Gross PnL": gross,
            "Net PnL": net
        })

        i = exit_i + 1

    trades = pd.DataFrame(trades)
    daily = pd.Series(0.0, index=data.index)

    if not trades.empty:
        pnl = trades.groupby("Exit date")["Net PnL"].sum()
        daily.loc[pnl.index] = pnl

    equity = daily.cumsum()
    drawdown = equity - equity.cummax()

    return trades, equity, drawdown

# ---------------------------------------------------------
# PAPER TRADING STATE MACHINE
# ---------------------------------------------------------

def make_signal(date, probability):
    return {
        "date": date.strftime("%Y-%m-%d"),
        "direction": "LONG" if probability >= 0.50 else "SHORT",
        "probability": float(probability)
    }

def initialize_state(state, prices, probabilities):
    # Primo avvio: nessun trade retrospettivo.
    # Prepara il segnale sull'ultima candela completa.
    if state["last_processed_date"] is None:
        date = prices.index[-1]
        state["pending_signal"] = make_signal(
            date, probabilities.loc[date]
        )
        state["last_processed_date"] = date.strftime("%Y-%m-%d")
        return state, True

    return state, False

def process_sessions(state, prices, probabilities):
    changed = False
    last_date = pd.Timestamp(state["last_processed_date"])
    new_dates = prices.index[prices.index > last_date]

    for date in new_dates:
        row = prices.loc[date]
        date_string = date.strftime("%Y-%m-%d")

        # Ingresso al prossimo Open dopo il segnale
        if state["position"] is None and state["pending_signal"] is not None:
            signal = state["pending_signal"]
            side = 1 if signal["direction"] == "LONG" else -1
            entry = float(row["open"])

            state["position"] = {
                "direction": signal["direction"],
                "side": side,
                "signal_date": signal["date"],
                "probability": signal["probability"],
                "entry_date": date_string,
                "entry": entry,
                "stop_price": entry - side * STOP,
                "holding_sessions": 0
            }
            state["pending_signal"] = None

        # Gestione della posizione: stop e durata
        position = state["position"]

        if position is not None:
            side = position["side"]
            stop_price = position["stop_price"]
            entry = position["entry"]

            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])

            exit_price = None
            reason = None

            if side == 1:
                if o <= stop_price:
                    exit_price, reason = o, "Stop gap"
                elif l <= stop_price:
                    exit_price, reason = stop_price, "Stop"
            else:
                if o >= stop_price:
                    exit_price, reason = o, "Stop gap"
                elif h >= stop_price:
                    exit_price, reason = stop_price, "Stop"

            position["holding_sessions"] += 1

            if exit_price is None and position["holding_sessions"] >= HOLDING:
                exit_price = float(row["close"])
                reason = "Scadenza 5 sessioni"

            if exit_price is not None:
                gross = side * (exit_price - entry)

                state["trades"].append({
                    "Signal date": position["signal_date"],
                    "Entry date": position["entry_date"],
                    "Exit date": date_string,
                    "Direction": position["direction"],
                    "Entry": entry,
                    "Exit": exit_price,
                    "Reason": reason,
                    "Gross PnL": gross,
                    "Net PnL": gross - COST
                })

                state["position"] = None

        # Segnale di fine giornata, soltanto se flat
        if state["position"] is None:
            state["pending_signal"] = make_signal(
                date, probabilities.loc[date]
            )

        state["last_processed_date"] = date_string
        changed = True

    return state, changed

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

try:
    with st.spinner("Aggiornamento dati e modello..."):
        raw = download_data()
        prices = get_ohlc(raw)

        # Non usare la candela odierna prima della chiusura
        now_us = datetime.now(ZoneInfo("America/New_York"))
        if len(prices) and prices.index[-1].date() == now_us.date():
            if now_us.hour < 17:
                prices = prices.iloc[:-1]

        feature_data = calculate_features(raw)
        X = feature_data[FEATURES].dropna()

        model = load_model()
        probabilities = pd.Series(
            model.predict_proba(X)[:, 1],
            index=X.index
        )

        # Allineamento prezzi / feature e rimozione barra incompleta
        prices = prices.loc[prices.index.isin(probabilities.index)]
        probabilities = probabilities.reindex(prices.index).dropna()
        prices = prices.reindex(probabilities.index)

        if len(prices) < 2:
            raise ValueError("Dati insufficienti per elaborare il segnale.")

        state, sha = load_state()
        state, initialized = initialize_state(state, prices, probabilities)

        state, processed = process_sessions(
            state, prices, probabilities
        )

        if initialized or processed:
            save_state(state, sha)

    # -----------------------------------------------------
    # CURRENT OPERATION
    # -----------------------------------------------------

    st.subheader("Operatività corrente")

    st.caption(
        "Segnale calcolato sulla candela giornaliera completata. "
        "Ingresso paper al successivo Open Yahoo."
    )

    position = state["position"]
    pending = state["pending_signal"]

    if position is not None:
        st.success(f"POSIZIONE APERTA — {position['direction']}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Prezzo ingresso", f"{position['entry']:.5f}")
        c2.metric("Stop", f"{position['stop_price']:.5f}")
        c3.metric("Sessioni", f"{position['holding_sessions']}/5")
        c4.metric("Data ingresso", position["entry_date"])

        st.write(
            f"Segnale: {position['signal_date']} | "
            f"Probabilità LONG: {position['probability']:.4f}"
        )

    elif pending is not None:
        st.info(f"SEGNALE {pending['direction']}")

        c1, c2 = st.columns(2)
        c1.metric("Probabilità LONG", f"{pending['probability']:.4f}")
        c2.metric("Data segnale", pending["date"])

        st.write("Ingresso previsto: al prossimo Open Yahoo.")

    else:
        st.warning("IN ATTESA DI SEGNALE")

    st.caption(
        "Paper trading su prezzi Yahoo Finance: non vengono inviati "
        "ordini a un broker."
    )

    # -----------------------------------------------------
    # OOS RESULTS ONLY
    # -----------------------------------------------------

    st.divider()
    st.subheader("Risultati out-of-sample")
    st.caption("Periodo storico: 03/01/2023 – 21/10/2026")

    oos_mask = (
        (prices.index >= pd.Timestamp("2023-01-03"))
        & (prices.index <= pd.Timestamp("2026-10-21"))
    )

    oos_prices = prices.loc[oos_mask]
    oos_probabilities = probabilities.reindex(oos_prices.index)

    oos_trades, oos_equity, oos_drawdown = make_backtest(
        oos_prices, oos_probabilities
    )

    if not oos_trades.empty:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Trade OOS", len(oos_trades))
        c2.metric("P&L netto OOS", f"{oos_trades['Net PnL'].sum():.2f} punti")
        c3.metric("Max drawdown OOS", f"{oos_drawdown.min():.2f} punti")
        c4.metric(
            "Trade positivi",
            f"{(oos_trades['Net PnL'] > 0).mean() * 100:.1f}%"
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=oos_equity.index,
            y=oos_equity.values,
            mode="lines",
            name="Equity netta OOS"
        ))
        fig.update_layout(
            xaxis_title="Data",
            yaxis_title="Punti",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("Nessun trade OOS disponibile.")

    # -----------------------------------------------------
    # LIVE TRADE LOG
    # -----------------------------------------------------

    st.divider()
    st.subheader("Registro paper trading")

    if state["trades"]:
        live_trades = pd.DataFrame(state["trades"])

        st.dataframe(
            live_trades.tail(50),
            use_container_width=True
        )

        st.metric(
            "P&L netto paper trading",
            f"{live_trades['Net PnL'].sum():.2f} punti"
        )

        st.download_button(
            "Scarica registro CSV",
            live_trades.to_csv(index=False).encode("utf-8"),
            "TNotes10Y_paper_trades.csv",
            "text/csv"
        )
    else:
        st.write("Nessun trade paper ancora completato.")

    st.caption(
        "Ultima sessione elaborata: "
        f"{state['last_processed_date']}"
    )

    if st.button("Aggiorna dati e segnale"):
        st.cache_data.clear()
        st.rerun()

except Exception as e:
    st.error(f"Errore applicazione: {e}")
    st.exception(e)
