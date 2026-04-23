import os
import zipfile
import warnings
from datetime import datetime
import tensorflow as tf
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from stable_baselines3 import PPO
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Nishe FX Portfolio App", layout="wide")

# =========================================================
# CONFIG
# =========================================================
PPO_MODEL_PATH = "ppo_fx_final_model_sharpe.zip"  # or folder name if you saved differently
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

# These names must match the PPO dataset columns if present.
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
        model_path = os.path.join(LSTM_DIR, f"{ccy}_lstm_fx.keras")
        scaler_path = os.path.join(LSTM_DIR, f"{ccy}_scaler.pkl")
        meta_path = os.path.join(LSTM_DIR, f"{ccy}_metadata.json")

        if os.path.exists(model_path):
            models[ccy] = load_model(model_path)
        if os.path.exists(scaler_path):
            scalers[ccy] = joblib.load(scaler_path)
        if os.path.exists(meta_path):
            metadata[ccy] = pd.read_json(meta_path, typ="series").to_dict()

    return models, scalers, metadata


# =========================================================
# DATA PREP FOR LSTM
# =========================================================
def build_lstm_feature_frame_from_close(close: pd.Series) -> pd.DataFrame:
    """
    This matches the LSTM training feature engineering.
    The training code used these exact fields before sequence creation.
    """
    df = pd.DataFrame(index=close.index)
    df["close"] = pd.to_numeric(close, errors="coerce")
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

    df = df.dropna().copy()
    return df


@st.cache_data(ttl=3600)
def download_fx_history(ticker: str, period: str = "25y") -> pd.Series:
    hist = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    return _get_close_from_download(hist, ticker)


def get_latest_live_rates() -> dict:
    latest = {}
    for ccy, cfg in CURRENCY_CONFIG.items():
        close = download_fx_history(cfg["ticker"], period="3mo")
        latest[ccy] = float(close.iloc[-1]) if len(close) else np.nan
    return latest


def prepare_lstm_input_for_currency(ccy: str, scenario_spot: float | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    close = download_fx_history(CURRENCY_CONFIG[ccy]["ticker"], period="25y").copy()
    if len(close) < LOOKBACK + 130:
        raise ValueError(f"Not enough history for {ccy} to create a {LOOKBACK}-day LSTM input window.")

    # For scenario testing, overwrite only the latest spot.
    # This is an approximation, but it is the cleanest way to let the user stress the current level.
    if scenario_spot is not None and np.isfinite(scenario_spot):
        close.iloc[-1] = float(scenario_spot)

    feat_df = build_lstm_feature_frame_from_close(close)
    if len(feat_df) < LOOKBACK:
        raise ValueError(f"Not enough engineered feature rows for {ccy} after rolling calculations.")

    return feat_df.iloc[-LOOKBACK:].copy(), feat_df


def predict_fx_return_1m(ccy: str, scenario_spot: float | None = None) -> float:
    if ccy not in LSTM_MODELS or ccy not in LSTM_SCALERS:
        return 0.0

    seq_df, _ = prepare_lstm_input_for_currency(ccy, scenario_spot=scenario_spot)
    scaler = LSTM_SCALERS[ccy]
    model = LSTM_MODELS[ccy]

    x = seq_df.values.astype(np.float32)
    x_scaled = scaler.transform(x)
    x_scaled = x_scaled.reshape(1, x_scaled.shape[0], x_scaled.shape[1])

    pred = model.predict(x_scaled, verbose=0).flatten()[0]
    return float(pred)


def get_live_predicted_fx_vector(scenario_spots: dict | None = None) -> np.ndarray:
    preds = []
    for ccy in ACTIVE_CURRENCIES:
        spot = None
        if scenario_spots is not None:
            spot = scenario_spots.get(ccy)
        preds.append(predict_fx_return_1m(ccy, scenario_spot=spot))
    return np.array(preds, dtype=np.float32)


# =========================================================
# PPO STATE PREP
# =========================================================
def get_state_from_row(row: pd.Series, prev_weights: np.ndarray | None = None) -> np.ndarray:
    if prev_weights is None:
        prev_weights = np.zeros(len(ACTIVE_CURRENCIES), dtype=np.float32)
    feature_values = row[STATE_FEATURES].values.astype(np.float32)
    obs = np.concatenate([feature_values, prev_weights]).astype(np.float32)
    return obs


def build_current_row(base_dataset: pd.DataFrame, live_fx: dict) -> pd.Series:
    """
    Start from the latest dataset row, then overwrite any FX spot fields that also exist in PPO features.
    Rate differentials stay as the latest dataset values unless user changes them in scenario mode.
    """
    row = base_dataset.iloc[-1].copy()
    for ccy, live_val in live_fx.items():
        if not np.isfinite(live_val):
            continue
        for col in FX_FEATURE_MAP.get(ccy, []):
            if col in row.index:
                row[col] = live_val
    return row


def get_carry_vector_from_row(row: pd.Series) -> np.ndarray:
    carry_vec = []
    for ccy in ACTIVE_CURRENCIES:
        rate_col = RATE_DIFF_COLS[ccy]
        rate_diff = float(row.get(rate_col, 0.0)) if pd.notna(row.get(rate_col, 0.0)) else 0.0
        carry_vec.append(annual_diff_to_monthly_carry(rate_diff))
    return np.array(carry_vec, dtype=np.float32)


def overwrite_row_for_scenario(row: pd.Series, scenario_spots: dict, scenario_rate_diffs: dict) -> pd.Series:
    row = row.copy()

    for ccy, fx_val in scenario_spots.items():
        for col in FX_FEATURE_MAP.get(ccy, []):
            if col in row.index:
                row[col] = fx_val

    for ccy, rate_val in scenario_rate_diffs.items():
        rate_col = RATE_DIFF_COLS[ccy]
        if rate_col in row.index:
            row[rate_col] = rate_val

    return row


# =========================================================
# PORTFOLIO METRICS
# =========================================================
def compute_portfolio_metrics(weights: np.ndarray, row: pd.Series, scenario_spots: dict | None = None):
    pred_fx = get_live_predicted_fx_vector(scenario_spots=scenario_spots)
    carry = get_carry_vector_from_row(row)

    fx_contribution = float(np.dot(weights, pred_fx))
    carry_contribution = float(np.dot(weights, carry))
    total_return = fx_contribution + carry_contribution

    # Approximate risk from the latest 12 monthly-equivalent states using the latest LSTM prediction framework.
    # Since LSTM is live and forward-looking, we estimate expected-return dispersion using recent PPO rows.
    hist_returns = []
    tail_df = DATASET.tail(VOL_LOOKBACK_MONTHS)

    for _, hist_row in tail_df.iterrows():
        hist_carry = get_carry_vector_from_row(hist_row)
        hist_pred_fx = pred_fx  # keep the same current prediction set for expected-return-based risk proxy
        hist_returns.append(float(np.dot(weights, hist_pred_fx + hist_carry)))

    hist_returns = np.array(hist_returns, dtype=float)
    volatility = float(np.std(hist_returns, ddof=0)) if len(hist_returns) else 0.0
    sharpe = float((np.mean(hist_returns) / volatility) * np.sqrt(12)) if volatility > 1e-12 else 0.0

    return {
        "predicted_fx_returns": pred_fx,
        "carry_vector": carry,
        "fx_contribution": fx_contribution,
        "carry_contribution": carry_contribution,
        "total_return": total_return,
        "volatility": volatility,
        "sharpe": sharpe,
    }


def make_weights_table(weights: np.ndarray, metrics: dict) -> pd.DataFrame:
    df = pd.DataFrame({
        "Currency": ACTIVE_CURRENCIES,
        "Weight": weights,
        "Predicted FX 1M Return": metrics["predicted_fx_returns"],
        "Carry 1M": metrics["carry_vector"],
    })
    df["Expected Asset 1M Return"] = df["Predicted FX 1M Return"] + df["Carry 1M"]
    df["Position"] = np.where(df["Weight"] >= 0, "Long", "Short")
    return df.sort_values("Weight", ascending=False).reset_index(drop=True)


def make_market_inputs_table(live_fx: dict, row: pd.Series) -> pd.DataFrame:
    records = []
    for ccy in ACTIVE_CURRENCIES:
        records.append({
            "Currency": ccy,
            "FX Pair": CURRENCY_CONFIG[ccy]["pair"],
            "Current FX Rate": live_fx.get(ccy, np.nan),
            "Rate Differential vs USD (%)": row.get(RATE_DIFF_COLS[ccy], np.nan),
        })
    return pd.DataFrame(records)


# =========================================================
# INIT
# =========================================================
ensure_lstm_bundle_extracted()
DATASET = load_dataset()
ACTIVE_CURRENCIES = [ccy for ccy in CURRENCY_CONFIG.keys() if f"{ccy}_target" in DATASET.columns or any(c in DATASET.columns for c in FX_FEATURE_MAP[ccy])]
STATE_FEATURES = [col for col in DATASET.columns if not col.endswith("_target")]
STATE_FEATURES = [col for col in STATE_FEATURES if pd.api.types.is_numeric_dtype(DATASET[col])]

PPO_MODEL = load_ppo_model()
LSTM_MODELS, LSTM_SCALERS, LSTM_METADATA = load_lstm_assets()

# =========================================================
# UI
# =========================================================
st.title("FX Portfolio Optimization using PPO + LSTM")
st.caption(
    "LSTM predicts next 1-month FX returns from the latest 180 trading days. "
    "PPO then chooses the long/short currency allocation."
)

with st.sidebar:
    st.header("Model Settings")
    st.write(f"Active currencies: {', '.join(ACTIVE_CURRENCIES)}")
    st.write(f"LSTM lookback window: {LOOKBACK} trading days")
    st.write(f"Forecast horizon: {HORIZON_TRADING_DAYS} trading days (~1 month)")
    st.write(f"Position limit per currency: {POSITION_LIMIT:.0%}")
    st.write(f"Risk proxy window: {VOL_LOOKBACK_MONTHS} months")

live_fx = get_latest_live_rates()
current_row = build_current_row(DATASET, live_fx)
timestamp_label = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Tabs
current_tab, scenario_tab = st.tabs(["Current Optimal Portfolio", "Scenario Simulation"])

# =========================================================
# CURRENT TAB
# =========================================================
with current_tab:
    st.subheader("Current PPO Portfolio")
    st.write(f"Refresh timestamp: **{timestamp_label}**")

    st.markdown("### Current Market Inputs")
    st.dataframe(make_market_inputs_table(live_fx, current_row), use_container_width=True)

    current_obs = get_state_from_row(current_row)
    current_action, _ = PPO_MODEL.predict(current_obs, deterministic=True)
    current_weights = action_to_weights(current_action)
    current_metrics = compute_portfolio_metrics(current_weights, current_row)
    current_weights_df = make_weights_table(current_weights, current_metrics)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected 1M Return", f"{current_metrics['total_return']:.2%}")
    c2.metric("Estimated Volatility", f"{current_metrics['volatility']:.2%}")
    c3.metric("Estimated Sharpe Ratio", f"{current_metrics['sharpe']:.2f}")
    c4.metric("Sum of Weights", f"{current_weights.sum():.10f}")

    c5, c6 = st.columns(2)
    c5.metric("FX Contribution", f"{current_metrics['fx_contribution']:.2%}")
    c6.metric("Carry Contribution", f"{current_metrics['carry_contribution']:.2%}")

    st.markdown("### Predicted 1-Month FX Returns")
    pred_df = pd.DataFrame({
        "Currency": ACTIVE_CURRENCIES,
        "Predicted 1M FX Return": current_metrics["predicted_fx_returns"],
    })
    st.dataframe(pred_df, use_container_width=True)

    st.markdown("### Portfolio Weights")
    st.dataframe(current_weights_df, use_container_width=True)

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(current_weights_df["Currency"], current_weights_df["Weight"])
    ax1.axhline(0, linewidth=1)
    ax1.set_title("Current Portfolio Weights")
    ax1.set_ylabel("Weight")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(["FX", "Carry"], [current_metrics["fx_contribution"], current_metrics["carry_contribution"]])
    ax2.axhline(0, linewidth=1)
    ax2.set_title("Expected 1-Month Return Decomposition")
    ax2.set_ylabel("Contribution")
    st.pyplot(fig2)

# =========================================================
# SCENARIO TAB
# =========================================================
with scenario_tab:
    st.subheader("Scenario Simulation")
    st.write("Adjust the latest FX level and rate differential, then run the portfolio simulation.")

    scenario_spots = {}
    scenario_rate_diffs = {}

    left, right = st.columns(2)

    with left:
        st.markdown("### Scenario FX Rates")
        for ccy in ACTIVE_CURRENCIES:
            default_fx = float(live_fx.get(ccy, 0.0)) if pd.notna(live_fx.get(ccy, np.nan)) else 0.0
            scenario_spots[ccy] = st.number_input(
                f"{ccy} FX Rate ({CURRENCY_CONFIG[ccy]['pair']})",
                value=default_fx,
                step=0.01,
                format="%.4f",
                key=f"fx_{ccy}",
            )

    with right:
        st.markdown("### Scenario Rate Differentials")
        for ccy in ACTIVE_CURRENCIES:
            rate_col = RATE_DIFF_COLS[ccy]
            default_rate = float(current_row.get(rate_col, 0.0)) if pd.notna(current_row.get(rate_col, np.nan)) else 0.0
            scenario_rate_diffs[ccy] = st.number_input(
                f"{ccy} Rate Differential vs USD (%)",
                value=default_rate,
                step=0.10,
                format="%.2f",
                key=f"rate_{ccy}",
            )

    run_sim = st.button("Run Simulation", type="primary")

    if run_sim:
        scenario_row = overwrite_row_for_scenario(current_row, scenario_spots, scenario_rate_diffs)
        scenario_obs = get_state_from_row(scenario_row)
        scenario_action, _ = PPO_MODEL.predict(scenario_obs, deterministic=True)
        scenario_weights = action_to_weights(scenario_action)
        scenario_metrics = compute_portfolio_metrics(
            scenario_weights,
            scenario_row,
            scenario_spots=scenario_spots,
        )
        scenario_weights_df = make_weights_table(scenario_weights, scenario_metrics)

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Expected 1M Return", f"{scenario_metrics['total_return']:.2%}")
        s2.metric("Estimated Volatility", f"{scenario_metrics['volatility']:.2%}")
        s3.metric("Estimated Sharpe Ratio", f"{scenario_metrics['sharpe']:.2f}")
        s4.metric("Sum of Weights", f"{scenario_weights.sum():.10f}")

        s5, s6 = st.columns(2)
        s5.metric("FX Contribution", f"{scenario_metrics['fx_contribution']:.2%}")
        s6.metric("Carry Contribution", f"{scenario_metrics['carry_contribution']:.2%}")

        st.markdown("### Predicted 1-Month FX Returns (Scenario)")
        sim_pred_df = pd.DataFrame({
            "Currency": ACTIVE_CURRENCIES,
            "Predicted 1M FX Return": scenario_metrics["predicted_fx_returns"],
        })
        st.dataframe(sim_pred_df, use_container_width=True)

        st.markdown("### Simulated Portfolio Weights")
        st.dataframe(scenario_weights_df, use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.bar(scenario_weights_df["Currency"], scenario_weights_df["Weight"])
        ax3.axhline(0, linewidth=1)
        ax3.set_title("Scenario Portfolio Weights")
        ax3.set_ylabel("Weight")
        st.pyplot(fig3)

        fig4, ax4 = plt.subplots(figsize=(6, 4))
        ax4.bar(["FX", "Carry"], [scenario_metrics["fx_contribution"], scenario_metrics["carry_contribution"]])
        ax4.axhline(0, linewidth=1)
        ax4.set_title("Scenario Expected 1-Month Return Decomposition")
        ax4.set_ylabel("Contribution")
        st.pyplot(fig4)
    else:
        st.info("Adjust the inputs and click Run Simulation.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.write(
    "Forecast engine: LSTM | Allocation engine: PPO | "
    "Return definition: predicted FX change + carry | "
    "Constraint: self-financing (weights sum to zero)"
)

st.warning(
    "Scenario mode changes only the latest FX level and rate differential. "
    "That is a useful stress test, but it is still an approximation of how the full market path would evolve."
)
