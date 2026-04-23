import os
import zipfile
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import yfinance as yf
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Nishe FX Portfolio App", layout="wide")

# =========================================================
# CONFIG
# =========================================================
PPO_MODEL_PATH = "ppo_fx_final_model_sharpe.zip"
DATASET_PATH = "dataset_step2_features.csv"
LSTM_BUNDLE_ZIP = "lstm_fx_models_bundle.zip"
LSTM_DIR = "lstm_fx_models"
POSITION_LIMIT = 0.9
LOOKBACK = 180
HORIZON_TRADING_DAYS = 21
VOL_LOOKBACK_MONTHS = 12

CURRENCY_CONFIG = {
    "EUR": {"ticker": "EURUSD=X", "pair": "EUR/USD", "orientation": "direct"},
    "GBP": {"ticker": "GBPUSD=X", "pair": "GBP/USD", "orientation": "direct"},
    "AUD": {"ticker": "AUDUSD=X", "pair": "AUD/USD", "orientation": "direct"},
    "JPY": {"ticker": "JPY=X",    "pair": "USD/JPY", "orientation": "inverse"},
    "INR": {"ticker": "INR=X",    "pair": "USD/INR", "orientation": "inverse"},
    "CNY": {"ticker": "CNY=X",    "pair": "USD/CNY", "orientation": "inverse"},
}

FX_FEATURE_MAP = {
    "EUR": ["EUR_spot", "EUR_fx", "EUR_close"],
    "GBP": ["GBP_spot", "GBP_fx", "GBP_close"],
    "AUD": ["AUD_spot", "AUD_fx", "AUD_close"],
    "JPY": ["JPY_spot", "JPY_fx", "JPY_close"],
    "INR": ["INR_spot", "INR_fx", "INR_close"],
    "CNY": ["CNY_spot", "CNY_fx", "CNY_close"],
}

RATE_DIFF_COLS = {
    "EUR": "EUR_rate_diff",
    "GBP": "GBP_rate_diff",
    "AUD": "AUD_rate_diff",
    "JPY": "JPY_rate_diff",
    "INR": "INR_rate_diff",
    "CNY": "CNY_rate_diff",
}

# =========================================================
# PYTORCH LSTM MODEL
# =========================================================
class FXLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# =========================================================
# UTILS
# =========================================================
def ensure_lstm_bundle_extracted(zip_path: str = LSTM_BUNDLE_ZIP, extract_dir: str = LSTM_DIR):
    if os.path.isdir(extract_dir):
        return
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Could not find {zip_path}. Put your downloaded LSTM bundle in the project root."
        )
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(".")


def _get_close_from_download(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            s = df[("Close", ticker)].copy()
        else:
            s = df["Close"].iloc[:, 0].copy()
    else:
        s = df["Close"].copy()

    s = pd.to_numeric(s, errors="coerce").dropna()
    s.name = "close"
    return s


def action_to_weights(action: np.ndarray, position_limit: float = POSITION_LIMIT) -> np.ndarray:
    weights = np.clip(action, -1.0, 1.0).astype(np.float64)
    weights = weights * position_limit

    for _ in range(50):
        weights = weights - weights.mean()
        weights = np.clip(weights, -position_limit, position_limit)

    weights = weights - weights.mean()
    return weights.astype(np.float32)


def annual_diff_to_monthly_carry(rate_diff_pct: float) -> float:
    return (float(rate_diff_pct) / 100.0) / 12.0


# =========================================================
# LOADERS
# =========================================================
@st.cache_resource
def load_ppo_model():
    return PPO.load(PPO_MODEL_PATH)


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATASET_PATH, index_col=0, parse_dates=True)

    rename_map = {}
    for col in df.columns:
        if col.endswith("_ret_1m_target"):
            rename_map[col] = col.replace("_ret_1m_target", "_target")

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


@st.cache_resource
def load_lstm_assets():
    ensure_lstm_bundle_extracted()

    models = {}
    scalers = {}
    metadata = {}

    for ccy in CURRENCY_CONFIG.keys():
        meta_path = os.path.join(LSTM_DIR, f"{ccy}_metadata.json")
        scaler_path = os.path.join(LSTM_DIR, f"{ccy}_scaler.pkl")
        model_path = os.path.join(LSTM_DIR, f"{ccy}_lstm_fx.pth")

        if os.path.exists(meta_path):
            metadata[ccy] = pd.read_json(meta_path, typ="series").to_dict()

        if os.path.exists(scaler_path):
            scalers[ccy] = joblib.load(scaler_path)

        if os.path.exists(model_path):
            if ccy not in metadata:
                raise ValueError(f"Missing metadata for {ccy}. Expected {meta_path}")

            input_size = int(metadata[ccy]["input_size"])
            hidden_size = int(metadata[ccy].get("hidden_size", 64))
            num_layers = int(metadata[ccy].get("num_layers", 2))
            dropout = float(metadata[ccy].get("dropout", 0.2))

            model = FXLSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
            )

            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
            model.eval()

            models[ccy] = model

    return models, scalers, metadata
