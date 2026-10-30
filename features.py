import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_2", "ret_5", "ret_10", "ret_20",
    "vol_5", "vol_10", "vol_20", "atr14_pct",
    "dist_high_10", "dist_low_10",
    "dist_high_20", "dist_low_20"
]

def prepare_ohlc(df):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]

    df = df[["open", "high", "low", "close"]].copy()

    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[~df.index.duplicated(keep="first")].sort_index()

    return df


def calculate_features(df):
    df = prepare_ohlc(df)

    c = df["close"]
    h = df["high"]
    l = df["low"]

    for n in [1, 2, 5, 10, 20]:
        df[f"ret_{n}"] = c.pct_change(n)

    ret1 = c.pct_change()

    for n in [5, 10, 20]:
        df[f"vol_{n}"] = ret1.rolling(n).std()

    prev_close = c.shift(1)

    true_range = pd.concat([
        h - l,
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr14_pct"] = true_range.rolling(14).mean() / c

    for n in [10, 20]:
        rolling_high = h.rolling(n).max()
        rolling_low = l.rolling(n).min()

        df[f"dist_high_{n}"] = c / rolling_high - 1
        df[f"dist_low_{n}"] = c / rolling_low - 1

    return df
