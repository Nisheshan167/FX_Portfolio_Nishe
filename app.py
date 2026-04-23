import os
import json
import zipfile
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import yfinance as yf
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

# =========================================================
# PAGE
# =========================================================
st.set_page_config(page_title="Nishe FX Portfolio App", layout="wide")

# =========================================================
# CONFIG
# =========================================================
APP_DIR = Path(".")
DATASET_PATH = APP_DIR / "dataset_step2_features.csv"
PPO_MODEL_PATH = APP_DIR / "ppo_fx_final_model_sharpe.zip"
LSTM_BUNDLE_ZIP = APP_DIR / "lstm_fx_models_bundle.zip"
LSTM_EXTRACT_DIR = APP_DIR / "lstm_fx_models"
LOOKBACK_DEFAULT = 180
HORIZON_TRADING_DAYS = 21
POSITION_LIMIT = 0.90
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

KERAS_EXTS = {".keras", ".h5"}
SCALER_EXTS = {".pkl", ".joblib"}
METADATA_EXTS = {".json"}

# =========================================================
# FILE HELPERS
# =========================================================
def extract_zip_if_needed(zip_path: Path, extract_dir: Path):
    if extract_dir.exists() and any(extract_dir.rglob("*")):
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing file: {zip_path.name}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def find_first_matching_file(base_dir: Path, currency: str, exts: set[str]):
    currency_lower = currency.lower()
    candidates = []
    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and currency_lower in p.stem.lower():
            candidates.append(p)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda x: (len(str(x)), str(x)))
    return candidates[0]


def list_bundle_files(base_dir: Path):
    return sorted([str(p.relative_to(base_dir)) for p in base_dir.rglob("*") if p.is_file()])


def load_json_file(path: Path):
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# =========================================================
# BASIC HELPERS
# =========================================================
def action_to_weights(action: np.ndarray, position_limit: float = POSITION_LIMIT) -> np.ndarray:
    weights = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    weights = weights * position_limit
    for _ in range(50):
        weights = weights - weights.mean()
        weights = np.clip(weights, -position_limit, position_limit)
    weights = weights - weights.mean()
    return weights.astype(np.float32)


def annual_diff_to_monthly_carry(rate_diff_pct: float) -> float:
    return (float(rate_diff_pct) / 100.0) / 12.0


def invert_if_needed(raw_ret: float, orientation: str) -> float:
    return -float(raw_ret) if orientation == "inverse" else float(raw_ret)

# =========================================================
# LOADERS
# =========================================================
@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError("dataset_step2_features.csv was not found.")
    df = pd.read_csv(DATASET_PATH, index_col=0, parse_dates=True)
    rename_map = {}
    for col in df.columns:
        if col.endswith("_ret_1m_target"):
            rename_map[col] = col.replace("_ret_1m_target", "_target")
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


@st.cache_resource
def load_ppo_model():
    if not PPO_MODEL_PATH.exists():
        raise FileNotFoundError(f"{PPO_MODEL_PATH.name} was not found.")
    return PPO.load(str(PPO_MODEL_PATH))


@st.cache_resource
def load_lstm_assets():
    extract_zip_if_needed(LSTM_BUNDLE_ZIP, LSTM_EXTRACT_DIR)
    assets = {}
    debug = {
        "bundle_files": list_bundle_files(LSTM_EXTRACT_DIR),
        "loaded_currencies": [],
        "missing_model": [],
        "errors": {},
    }

    for ccy in CURRENCY_CONFIG.keys():
        model_path = find_first_matching_file(LSTM_EXTRACT_DIR, ccy, KERAS_EXTS)
        scaler_path = find_first_matching_file(LSTM_EXTRACT_DIR, ccy, SCALER_EXTS)
        meta_path = find_first_matching_file(LSTM_EXTRACT_DIR, ccy, METADATA_EXTS)

        if model_path is None:
            debug["missing_model"].append(ccy)
            continue

        try:
            model = tf.keras.models.load_model(model_path)
            scaler = joblib.load(scaler_path) if scaler_path else None
            metadata = load_json_file(meta_path)
            assets[ccy] = {
                "model": model,
                "scaler": scaler,
                "metadata": metadata,
                "model_path": str(model_path),
                "scaler_path": str(scaler_path) if scaler_path else None,
                "meta_path": str(meta_path) if meta_path else None,
            }
            debug["loaded_currencies"].append(ccy)
        except Exception as e:
            debug["errors"][ccy] = str(e)

    return assets, debug

# =========================================================
# MARKET DATA
# =========================================================
def _extract_close_series(df: pd.DataFrame, ticker: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if ("Close", ticker) in df.columns:
            out = df[("Close", ticker)].copy()
        elif "Close" in df.columns.get_level_values(0):
            out = df["Close"].iloc[:, 0].copy()
        else:
            out = df.iloc[:, 0].copy()
    else:
        out = df["Close"].copy() if "Close" in df.columns else df.iloc[:, 0].copy()
    out = pd.to_numeric(out, errors="coerce").dropna()
    out.name = "close"
    return out


def download_close_history(ticker: str, years: int = 5) -> pd.Series:
    raw = yf.download(ticker, period=f"{years}y", auto_adjust=False, progress=False)
    close = _extract_close_series(raw, ticker)
    if close.empty:
        raise ValueError(f"No price data returned for ticker {ticker}")
    close.index = pd.to_datetime(close.index)
    return close


def get_latest_live_rates() -> dict:
    live_fx = {}
    for ccy, cfg in CURRENCY_CONFIG.items():
        try:
            close = download_close_history(cfg["ticker"], years=1)
            live_fx[ccy] = float(close.iloc[-1])
        except Exception:
            live_fx[ccy] = np.nan
    return live_fx

# =========================================================
# KERAS FEATURE ENGINEERING
# =========================================================
def build_features_from_close(close: pd.Series, horizon: int = HORIZON_TRADING_DAYS) -> pd.DataFrame:
    df = pd.DataFrame(index=close.index)
    df["close"] = close
    df["log_close"] = np.log(df["close"])
    df["ret_1d"] = df["close"].pct_change()
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_21d"] = df["close"].pct_change(21)
    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_21"] = df["close"].rolling(21).mean()
    df["ma_63"] = df["close"].rolling(63).mean()
    df["vol_21"] = df["ret_1d"].rolling(21).std()
    df["vol_63"] = df["ret_1d"].rolling(63).std()
    df["mom_21"] = df["close"] / df["close"].shift(21) - 1.0
    df["mom_63"] = df["close"] / df["close"].shift(63) - 1.0
    df["mom_126"] = df["close"] / df["close"].shift(126) - 1.0
    df["price_vs_ma21"] = df["close"] / df["ma_21"] - 1.0
    df["price_vs_ma63"] = df["close"] / df["ma_63"] - 1.0
    df["ma21_vs_ma63"] = df["ma_21"] / df["ma_63"] - 1.0
    df["target_1m"] = df["close"].shift(-horizon) / df["close"] - 1.0
    df = df.dropna().copy()
    return df


def choose_feature_columns(frame: pd.DataFrame, metadata: dict) -> list[str]:
    cols = metadata.get("feature_columns")
    if isinstance(cols, list) and len(cols) > 0:
        matched = [c for c in cols if c in frame.columns]
        if matched:
            return matched
    return [c for c in frame.columns if c != "target_1m"]


def prepare_keras_sequence(ccy: str, close: pd.Series, assets: dict):
    asset = assets[ccy]
    scaler = asset["scaler"]
    metadata = asset["metadata"]
    lookback = int(metadata.get("lookback_days", metadata.get("lookback", LOOKBACK_DEFAULT)))

    feature_frame = build_features_from_close(close)
    feature_cols = choose_feature_columns(feature_frame, metadata)
    X = feature_frame[feature_cols].copy()

    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X.values

    if len(X_scaled) < lookback:
        raise ValueError(f"Not enough rows for {ccy}. Need at least {lookback}, got {len(X_scaled)}")

    seq = X_scaled[-lookback:]
    return np.expand_dims(seq.astype(np.float32), axis=0)


def forecast_next_month_fx_return(ccy: str, close: pd.Series, assets: dict) -> float:
    asset = assets[ccy]
    model = asset["model"]
    seq = prepare_keras_sequence(ccy, close, assets)
    pred = model.predict(seq, verbose=0).reshape(-1)
    raw_pred = float(pred[0])
    return invert_if_needed(raw_pred, CURRENCY_CONFIG[ccy]["orientation"])


def get_live_predicted_fx_vector(active_currencies: list[str], assets: dict, scenario_spots: dict | None = None):
    preds = []
    close_map = {}
    for ccy in active_currencies:
        cfg = CURRENCY_CONFIG[ccy]
        close = download_close_history(cfg["ticker"], years=5).copy()
        if scenario_spots is not None and ccy in scenario_spots and np.isfinite(scenario_spots[ccy]):
            close.iloc[-1] = float(scenario_spots[ccy])
        pred = forecast_next_month_fx_return(ccy, close, assets)
        preds.append(pred)
        close_map[ccy] = close
    return np.array(preds, dtype=np.float32), close_map

# =========================================================
# PPO STATE HELPERS
# =========================================================
def get_active_currencies(dataset: pd.DataFrame, lstm_assets: dict) -> tuple[list[str], list[str]]:
    active = []
    dataset_currencies = []
    for ccy in CURRENCY_CONFIG.keys():
        dataset_has_currency = (f"{ccy}_target" in dataset.columns or any(col in dataset.columns for col in FX_FEATURE_MAP[ccy]))
        if dataset_has_currency:
            dataset_currencies.append(ccy)
        if dataset_has_currency and ccy in lstm_assets:
            active.append(ccy)
    return active, dataset_currencies


def build_state_features(dataset: pd.DataFrame) -> list[str]:
    cols = [c for c in dataset.columns if not c.endswith("_target")]
    return [c for c in cols if pd.api.types.is_numeric_dtype(dataset[c])]


def build_current_row(base_dataset: pd.DataFrame, live_fx: dict) -> pd.Series:
    row = base_dataset.iloc[-1].copy()
    for ccy, live_val in live_fx.items():
        if not np.isfinite(live_val):
            continue
        for col in FX_FEATURE_MAP.get(ccy, []):
            if col in row.index:
                row[col] = float(live_val)
    return row


def overwrite_row_for_scenario(row: pd.Series, scenario_spots: dict, scenario_rate_diffs: dict) -> pd.Series:
    out = row.copy()
    for ccy, fx_val in scenario_spots.items():
        if np.isfinite(fx_val):
            for col in FX_FEATURE_MAP.get(ccy, []):
                if col in out.index:
                    out[col] = float(fx_val)
    for ccy, rate_val in scenario_rate_diffs.items():
        rate_col = RATE_DIFF_COLS.get(ccy)
        if rate_col in out.index:
            out[rate_col] = float(rate_val)
    return out


def get_state_from_row(row: pd.Series, state_features: list[str], active_currencies: list[str], prev_weights=None) -> np.ndarray:
    if prev_weights is None:
        prev_weights = np.zeros(len(active_currencies), dtype=np.float32)
    feature_values = row[state_features].values.astype(np.float32)
    obs = np.concatenate([feature_values, prev_weights]).astype(np.float32)
    return obs


def get_carry_vector_from_row(row: pd.Series, active_currencies: list[str]) -> np.ndarray:
    carry = []
    for ccy in active_currencies:
        rate_col = RATE_DIFF_COLS[ccy]
        val = row.get(rate_col, 0.0)
        val = 0.0 if pd.isna(val) else float(val)
        carry.append(annual_diff_to_monthly_carry(val))
    return np.array(carry, dtype=np.float32)


def carry_trade_weights_from_row(row: pd.Series, active_currencies: list[str], position_limit: float = POSITION_LIMIT):
    rate_diffs = []
    for ccy in active_currencies:
        rate_col = RATE_DIFF_COLS[ccy]
        val = row.get(rate_col, 0.0)
        val = 0.0 if pd.isna(val) else float(val)
        rate_diffs.append(val)
    rate_diffs = np.array(rate_diffs, dtype=float)
    longs = np.where(rate_diffs > 0, rate_diffs, 0.0)
    shorts = np.where(rate_diffs < 0, -rate_diffs, 0.0)
    if longs.sum() == 0 and shorts.sum() == 0:
        return np.zeros(len(active_currencies), dtype=np.float32)
    if longs.sum() == 0:
        longs = np.ones_like(longs)
    if shorts.sum() == 0:
        shorts = np.ones_like(shorts)
    long_w = longs / longs.sum()
    short_w = shorts / shorts.sum()
    w = 0.5 * long_w - 0.5 * short_w
    return action_to_weights(w / max(position_limit, 1e-8), position_limit=position_limit)

# =========================================================
# METRICS
# =========================================================
def compute_portfolio_metrics(weights: np.ndarray, row: pd.Series, dataset: pd.DataFrame, active_currencies: list[str], lstm_assets: dict, scenario_spots: dict | None = None):
    pred_fx, _ = get_live_predicted_fx_vector(active_currencies, lstm_assets, scenario_spots=scenario_spots)
    carry = get_carry_vector_from_row(row, active_currencies)
    expected_asset_returns = pred_fx + carry
    fx_contribution = float(np.dot(weights, pred_fx))
    carry_contribution = float(np.dot(weights, carry))
    total_return = float(np.dot(weights, expected_asset_returns))

    hist_returns = []
    tail = dataset.tail(VOL_LOOKBACK_MONTHS)
    for _, hist_row in tail.iterrows():
        hist_carry = get_carry_vector_from_row(hist_row, active_currencies)
        hist_total = pred_fx + hist_carry
        hist_returns.append(float(np.dot(weights, hist_total)))
    hist_returns = np.asarray(hist_returns, dtype=float)
    volatility = float(np.std(hist_returns, ddof=0)) if len(hist_returns) > 0 else 0.0
    sharpe = float((np.mean(hist_returns) / volatility) * np.sqrt(12)) if volatility > 1e-12 else 0.0

    return {
        "predicted_fx_returns": pred_fx,
        "carry_vector": carry,
        "expected_asset_returns": expected_asset_returns,
        "fx_contribution": fx_contribution,
        "carry_contribution": carry_contribution,
        "total_return": total_return,
        "volatility": volatility,
        "sharpe": sharpe,
    }


def make_weights_table(weights: np.ndarray, metrics: dict, active_currencies: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({
        "Currency": active_currencies,
        "Weight": weights,
        "Predicted FX 1M Return": metrics["predicted_fx_returns"],
        "Carry 1M": metrics["carry_vector"],
        "Expected Asset 1M Return": metrics["expected_asset_returns"],
    })
    df["Position"] = np.where(df["Weight"] >= 0, "Long", "Short")
    return df.sort_values("Weight", ascending=False).reset_index(drop=True)


def make_market_inputs_table(live_fx: dict, row: pd.Series, active_currencies: list[str]) -> pd.DataFrame:
    records = []
    for ccy in active_currencies:
        records.append({
            "Currency": ccy,
            "FX Pair": CURRENCY_CONFIG[ccy]["pair"],
            "Current FX Rate": live_fx.get(ccy, np.nan),
            "Rate Differential vs USD (%)": row.get(RATE_DIFF_COLS[ccy], np.nan),
        })
    return pd.DataFrame(records)


def plot_weights_bar(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["Currency"], df["Weight"])
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    st.pyplot(fig)


def plot_return_decomposition(metrics: dict, title: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["FX", "Carry"], [metrics["fx_contribution"], metrics["carry_contribution"]])
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Contribution")
    st.pyplot(fig)

# =========================================================
# INIT
# =========================================================
try:
    DATASET = load_dataset()
    PPO_MODEL = load_ppo_model()
    LSTM_ASSETS, LSTM_DEBUG = load_lstm_assets()
except Exception as e:
    st.error(f"Startup failed: {e}")
    st.stop()

ACTIVE_CURRENCIES, DATASET_CURRENCIES = get_active_currencies(DATASET, LSTM_ASSETS)
STATE_FEATURES = build_state_features(DATASET)

if not ACTIVE_CURRENCIES:
    st.error("No currencies could be activated.")
    st.write("Currencies found in dataset:", DATASET_CURRENCIES)
    st.write("Currencies with loadable Keras LSTM models:", list(LSTM_ASSETS.keys()))
    if LSTM_DEBUG.get("bundle_files"):
        st.write("Files found inside LSTM bundle:", LSTM_DEBUG["bundle_files"])
    if LSTM_DEBUG.get("errors"):
        st.write("Model load errors:", LSTM_DEBUG["errors"])
    st.stop()

# =========================================================
# UI
# =========================================================
st.title("FX Portfolio Optimization using PPO + Keras LSTM")
st.caption("Keras LSTM forecasts next 1-month FX returns. PPO then allocates the long/short FX portfolio.")

with st.sidebar:
    st.header("Model Settings")
    st.write(f"Currencies found in dataset: {', '.join(DATASET_CURRENCIES) if DATASET_CURRENCIES else 'None'}")
    st.write(f"Currencies with loadable Keras LSTMs: {', '.join(ACTIVE_CURRENCIES) if ACTIVE_CURRENCIES else 'None'}")
    st.write(f"Forecast horizon: {HORIZON_TRADING_DAYS} trading days (~1 month)")
    st.write(f"Position limit per currency: {POSITION_LIMIT:.0%}")
    st.write(f"Risk proxy window: {VOL_LOOKBACK_MONTHS} months")
    with st.expander("Loaded model files"):
        st.write(f"PPO: {PPO_MODEL_PATH.name}")
        for ccy in ACTIVE_CURRENCIES:
            asset = LSTM_ASSETS[ccy]
            st.write(f"{ccy} model: {Path(asset['model_path']).name}")
            if asset["scaler_path"]:
                st.write(f"{ccy} scaler: {Path(asset['scaler_path']).name}")
            if asset["meta_path"]:
                st.write(f"{ccy} metadata: {Path(asset['meta_path']).name}")

live_fx = get_latest_live_rates()
current_row = build_current_row(DATASET, live_fx)
timestamp_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

current_tab, scenario_tab = st.tabs(["Current", "Simulation"])

with current_tab:
    st.subheader("Current Optimal Portfolio")
    st.write(f"Refresh timestamp: **{timestamp_label}**")
    st.markdown("### Current Market Inputs")
    st.dataframe(make_market_inputs_table(live_fx, current_row, ACTIVE_CURRENCIES), use_container_width=True)

    try:
        obs = get_state_from_row(current_row, STATE_FEATURES, ACTIVE_CURRENCIES)
        action, _ = PPO_MODEL.predict(obs, deterministic=True)
        weights = action_to_weights(action, position_limit=POSITION_LIMIT)
        metrics = compute_portfolio_metrics(weights, current_row, DATASET, ACTIVE_CURRENCIES, LSTM_ASSETS)
        weights_df = make_weights_table(weights, metrics, ACTIVE_CURRENCIES)

        carry_weights = carry_trade_weights_from_row(current_row, ACTIVE_CURRENCIES)
        carry_metrics = compute_portfolio_metrics(carry_weights, current_row, DATASET, ACTIVE_CURRENCIES, LSTM_ASSETS)
        carry_df = make_weights_table(carry_weights, carry_metrics, ACTIVE_CURRENCIES)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("PPO Expected 1M Return", f"{metrics['total_return']:.2%}")
        c2.metric("PPO Estimated Volatility", f"{metrics['volatility']:.2%}")
        c3.metric("PPO Estimated Sharpe", f"{metrics['sharpe']:.2f}")
        c4.metric("PPO Sum of Weights", f"{weights.sum():.10f}")

        c5, c6, c7 = st.columns(3)
        c5.metric("Carry Expected 1M Return", f"{carry_metrics['total_return']:.2%}")
        c6.metric("Carry Estimated Volatility", f"{carry_metrics['volatility']:.2%}")
        c7.metric("Carry Estimated Sharpe", f"{carry_metrics['sharpe']:.2f}")

        d1, d2 = st.columns(2)
        d1.metric("PPO FX Contribution", f"{metrics['fx_contribution']:.2%}")
        d2.metric("PPO Carry Contribution", f"{metrics['carry_contribution']:.2%}")

        st.markdown("### Predicted 1-Month FX Returns")
        pred_df = pd.DataFrame({"Currency": ACTIVE_CURRENCIES, "Predicted 1M FX Return": metrics["predicted_fx_returns"]})
        st.dataframe(pred_df, use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("### PPO Portfolio Weights")
            st.dataframe(weights_df, use_container_width=True)
            plot_weights_bar(weights_df, "Current PPO Portfolio Weights")
            plot_return_decomposition(metrics, "PPO Expected 1-Month Return Decomposition")
        with right:
            st.markdown("### Carry Trade Benchmark")
            st.dataframe(carry_df, use_container_width=True)
            plot_weights_bar(carry_df, "Current Carry Trade Weights")
            plot_return_decomposition(carry_metrics, "Carry Trade Expected 1-Month Return Decomposition")
    except Exception as e:
        st.error(f"Current portfolio calculation failed: {e}")

with scenario_tab:
    st.subheader("Scenario Simulation")
    st.write("Change FX rates and rate differentials, then run the scenario.")

    scenario_spots = {}
    scenario_rate_diffs = {}
    sc1, sc2 = st.columns(2)

    with sc1:
        st.markdown("### Scenario FX Rates")
        for ccy in ACTIVE_CURRENCIES:
            default_spot = float(live_fx.get(ccy, np.nan)) if np.isfinite(live_fx.get(ccy, np.nan)) else 0.0
            scenario_spots[ccy] = st.number_input(
                f"{ccy} FX Rate ({CURRENCY_CONFIG[ccy]['pair']})",
                value=float(default_spot),
                format="%.6f",
                key=f"spot_{ccy}",
            )

    with sc2:
        st.markdown("### Scenario Rate Differentials")
        for ccy in ACTIVE_CURRENCIES:
            rate_col = RATE_DIFF_COLS[ccy]
            default_rate = float(current_row.get(rate_col, 0.0)) if pd.notna(current_row.get(rate_col, 0.0)) else 0.0
            scenario_rate_diffs[ccy] = st.number_input(
                f"{ccy} Rate Differential vs USD (%)",
                value=float(default_rate),
                step=0.10,
                format="%.2f",
                key=f"rate_{ccy}",
            )

    run_sim = st.button("Run Simulation", type="primary")

    if run_sim:
        try:
            scenario_row = overwrite_row_for_scenario(current_row, scenario_spots, scenario_rate_diffs)
            scenario_obs = get_state_from_row(scenario_row, STATE_FEATURES, ACTIVE_CURRENCIES)
            scenario_action, _ = PPO_MODEL.predict(scenario_obs, deterministic=True)
            scenario_weights = action_to_weights(scenario_action, position_limit=POSITION_LIMIT)
            scenario_metrics = compute_portfolio_metrics(
                scenario_weights, scenario_row, DATASET, ACTIVE_CURRENCIES, LSTM_ASSETS, scenario_spots=scenario_spots
            )
            scenario_weights_df = make_weights_table(scenario_weights, scenario_metrics, ACTIVE_CURRENCIES)

            sim_carry_weights = carry_trade_weights_from_row(scenario_row, ACTIVE_CURRENCIES)
            sim_carry_metrics = compute_portfolio_metrics(
                sim_carry_weights, scenario_row, DATASET, ACTIVE_CURRENCIES, LSTM_ASSETS, scenario_spots=scenario_spots
            )
            sim_carry_df = make_weights_table(sim_carry_weights, sim_carry_metrics, ACTIVE_CURRENCIES)

            s1, s2, s3, s4 = st.columns(4)
            s1.metric("PPO Expected 1M Return", f"{scenario_metrics['total_return']:.2%}")
            s2.metric("PPO Estimated Volatility", f"{scenario_metrics['volatility']:.2%}")
            s3.metric("PPO Estimated Sharpe", f"{scenario_metrics['sharpe']:.2f}")
            s4.metric("PPO Sum of Weights", f"{scenario_weights.sum():.10f}")

            s5, s6, s7 = st.columns(3)
            s5.metric("Carry Expected 1M Return", f"{sim_carry_metrics['total_return']:.2%}")
            s6.metric("Carry Estimated Volatility", f"{sim_carry_metrics['volatility']:.2%}")
            s7.metric("Carry Estimated Sharpe", f"{sim_carry_metrics['sharpe']:.2f}")

            s8, s9 = st.columns(2)
            s8.metric("PPO FX Contribution", f"{scenario_metrics['fx_contribution']:.2%}")
            s9.metric("PPO Carry Contribution", f"{scenario_metrics['carry_contribution']:.2%}")

            st.markdown("### Predicted 1-Month FX Returns (Scenario)")
            sim_pred_df = pd.DataFrame({"Currency": ACTIVE_CURRENCIES, "Predicted 1M FX Return": scenario_metrics["predicted_fx_returns"]})
            st.dataframe(sim_pred_df, use_container_width=True)

            left2, right2 = st.columns(2)
            with left2:
                st.markdown("### Scenario PPO Portfolio")
                st.dataframe(scenario_weights_df, use_container_width=True)
                plot_weights_bar(scenario_weights_df, "Scenario PPO Portfolio Weights")
                plot_return_decomposition(scenario_metrics, "Scenario PPO Return Decomposition")
            with right2:
                st.markdown("### Scenario Carry Trade Benchmark")
                st.dataframe(sim_carry_df, use_container_width=True)
                plot_weights_bar(sim_carry_df, "Scenario Carry Trade Weights")
                plot_return_decomposition(sim_carry_metrics, "Scenario Carry Trade Return Decomposition")
        except Exception as e:
            st.error(f"Scenario simulation failed: {e}")
    else:
        st.info("Adjust the inputs and click Run Simulation.")

st.markdown("---")
st.write("Forecast engine: Keras LSTM | Allocation engine: PPO | Return definition: predicted FX change + carry | Constraint: self-financing (weights sum to zero)")
