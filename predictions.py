from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier

from src.features import FEATURES, calculate_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "TNotes10Y_CatBoost.cbm"

def load_model():
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    if list(model.feature_names_) != FEATURES:
        raise ValueError("Le feature del modello non corrispondono alla pipeline.")

    return model


def generate_prediction(ohlc):
    df = calculate_features(ohlc)
    usable = df[FEATURES].dropna()

    if usable.empty:
        raise ValueError("Dati insufficienti per calcolare le feature.")

    model = load_model()
    latest_features = usable.iloc[[-1]]
    probability = float(model.predict_proba(latest_features)[0, 1])

    return {
        "date": latest_features.index[-1],
        "probability_up": probability,
        "signal": "LONG" if probability >= 0.5 else "SHORT",
        "features": latest_features.iloc[0].to_dict()
    }
