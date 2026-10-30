import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import date

from src.predictions import generate_prediction
from src.features import calculate_features, FEATURES

st.set_page_config(
    page_title="TNotes-10Y | CatBoost",
    page_icon="📊",
    layout="wide"
)

st.title("TNotes-10Y")
st.caption("CatBoost classification model | 10-Year US Treasury Note Futures")

@st.cache_data(ttl=900)
def download_data():
    raw = yf.download(
        "ZN=F",
        period="2y",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if raw.empty:
        raise ValueError("Dati di mercato non disponibili.")

    return raw

try:
    raw = download_data()
    df = calculate_features(raw)

    result = generate_prediction(raw)

    price = float(df["close"].iloc[-1])
    probability = result["probability_up"]
    signal = result["signal"]

    c1, c2, c3 = st.columns(3)

    c1.metric("Ultimo close", f"{price:.4f}")
    c2.metric("Segnale modello", signal)
    c3.metric("Probabilità LONG", f"{probability:.2%}")

    st.divider()

    st.subheader("Probabilità direzionale")

    st.progress(min(max(probability, 0.0), 1.0))

    left, right = st.columns(2)
    left.metric("LONG", f"{probability:.2%}")
    right.metric("SHORT", f"{1-probability:.2%}")

    st.caption(
        f"Segnale calcolato sui dati disponibili alla chiusura del "
        f"{pd.Timestamp(result['date']).strftime('%d/%m/%Y')}."
    )

    st.divider()

    st.subheader("Andamento storico ZN")

    chart = go.Figure()
    chart.add_trace(go.Scatter(
        x=df.index,
        y=df["close"],
        mode="lines",
        name="ZN Close"
    ))

    chart.update_layout(
        height=450,
        xaxis_title="Data",
        yaxis_title="Prezzo",
        hovermode="x unified"
    )

    st.plotly_chart(chart, use_container_width=True)

    st.divider()

    st.subheader("Feature utilizzate dal modello")

    feature_table = pd.DataFrame(
        result["features"].items(),
        columns=["Feature", "Valore"]
    )

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Modello CatBoost originale: 1 albero, depth 5, "
        "learning rate 0.03, L2 5, seed 42."
    )

except Exception as e:
    st.error(f"Errore durante l'elaborazione: {e}")
    st.info("Verificare la disponibilità dei dati Yahoo Finance.")
