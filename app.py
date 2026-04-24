import os
import zipfile
import tempfile
import warnings
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from tensorflow.keras.models import load_model
from stable_baselines3 import PPO

warnings.filterwarnings("ignore")

# =====================================================
# APP CONFIG
# =====================================================

st.set_page_config(
    page_title="FX Portfolio Optimisation using LSTM + PPO",
    layout="wide"
)

CURRENCIES = ["EUR", "GBP", "AUD", "JPY", "INR", "CNY"]

TICKERS = {
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "AUD": "AUDUSD=X",
    "JPY": "JPY=X",
    "INR": "INR=X",
    "CNY": "CNY=X",
}

FEATURE_COLUMNS = [
    "close",
    "log_close",
    "ret_1d",
    "ret_5d",
    "ret_21d",
    "ma_5",
    "ma_21",
    "ma_63",
    "vol_21",
    "vol_63",
    "mom_21",
    "mom_63",
    "mom_126",
    "price_vs_ma21",
    "price_vs_ma63",
    "ma21_vs_ma63",
]

LOOKBACK_DAYS = 180
PPO_STATE_SIZE = 85

PPO_MODEL_PATH = "ppo_fx_final_model_sharpe.zip"
LSTM_BUNDLE_PATH = "lstm_fx_models_bundle.zip"


# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def extract_lstm_bundle():
    extract_dir = os.path.join(tempfile.gettempdir(), "lstm_fx_models")

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(LSTM_BUNDLE_PATH, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    return extract_dir


@st.cache_resource
def load_ppo_model():
    return PPO.load(PPO_MODEL_PATH)


@st.cache_resource
def load_lstm_models():
    model_dir = extract_lstm_bundle()

    models = {}
    scalers = {}

    for currency in CURRENCIES:
        model_path = os.path.join(model_dir, f"{currency}_lstm_fx.keras")
        scaler_path = os.path.join(model_dir, f"{currency}_scaler.pkl")

        models[currency] = load_model(model_path)
        scalers[currency] = joblib.load(scaler_path)

    return models, scalers


# =====================================================
# DATA + FEATURE ENGINEERING
# =====================================================

def download_fx_data(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=900)

    data = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]

    df = pd.DataFrame({"close": close})
    df = df.dropna()

    return df


def build_lstm_features(df):
    df = df.copy()

    df["log_close"] = np.log(df["close"])
    df["ret_1d"] = df["close"].pct_change(1)
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_21d"] = df["close"].pct_change(21)

    df["ma_5"] = df["close"].rolling(5).mean()
    df["ma_21"] = df["close"].rolling(21).mean()
    df["ma_63"] = df["close"].rolling(63).mean()

    df["vol_21"] = df["ret_1d"].rolling(21).std()
    df["vol_63"] = df["ret_1d"].rolling(63).std()

    df["mom_21"] = df["close"] / df["close"].shift(21) - 1
    df["mom_63"] = df["close"] / df["close"].shift(63) - 1
    df["mom_126"] = df["close"] / df["close"].shift(126) - 1

    df["price_vs_ma21"] = df["close"] / df["ma_21"] - 1
    df["price_vs_ma63"] = df["close"] / df["ma_63"] - 1
    df["ma21_vs_ma63"] = df["ma_21"] / df["ma_63"] - 1

    df = df.dropna()

    return df[FEATURE_COLUMNS]


def predict_lstm_return(currency, features, models, scalers):
    if len(features) < LOOKBACK_DAYS:
        raise ValueError(f"Not enough feature rows for {currency}")

    latest_window = features.tail(LOOKBACK_DAYS)

    scaler = scalers[currency]
    model = models[currency]

    scaled = scaler.transform(latest_window)
    X = np.expand_dims(scaled, axis=0)

    prediction = model.predict(X, verbose=0)

    return float(prediction.flatten()[0])


# =====================================================
# CARRY + PPO STATE
# =====================================================

def calculate_carry_return(annual_rate_diff):
    return annual_rate_diff / 12


def build_ppo_state(forecast_df):
    """
    PPO model expects 85 features.

    Since the exact PPO training environment state-builder is not in this app,
    this creates a structured 85-feature state from:
    - predicted FX return
    - carry return
    - expected total return
    - volatility
    - momentum
    - current FX level

    Best version: replace this with the exact state formula used during PPO training.
    """

    predicted_fx = forecast_df["Predicted FX Return"].values
    carry = forecast_df["Carry Return"].values
    total = forecast_df["Expected Total Return"].values
    vol = forecast_df["Volatility 21D"].values
    momentum = forecast_df["Momentum 21D"].values
    spot = forecast_df["Current FX Rate"].values

    risk_adjusted = np.divide(total, vol + 1e-8)

    state_parts = [
        predicted_fx,
        carry,
        total,
        vol,
        momentum,
        spot,
        risk_adjusted,
    ]

    state = np.concatenate(state_parts).astype(np.float32)

    if len(state) < PPO_STATE_SIZE:
        state = np.pad(state, (0, PPO_STATE_SIZE - len(state)), constant_values=0)

    if len(state) > PPO_STATE_SIZE:
        state = state[:PPO_STATE_SIZE]

    return state


def convert_action_to_weights(action):
    """
    PPO action space is [-1, 1].
    This converts actions into long-only portfolio weights summing to 100%.
    """

    action = np.array(action).flatten()

    positive_action = np.maximum(action, 0)

    if positive_action.sum() == 0:
        weights = np.repeat(1 / len(action), len(action))
    else:
        weights = positive_action / positive_action.sum()

    return weights


def calculate_portfolio_metrics(weights, expected_returns, volatilities):
    portfolio_return = float(np.dot(weights, expected_returns))

    portfolio_risk = float(
        np.sqrt(np.dot(weights ** 2, volatilities ** 2))
    )

    sharpe = portfolio_return / portfolio_risk if portfolio_risk != 0 else 0

    return portfolio_return, portfolio_risk, sharpe


# =====================================================
# STREAMLIT UI
# =====================================================

st.title("FX Portfolio Optimisation using LSTM Forecasting + PPO")

st.write(
    "This app predicts next 1-month FX returns using saved LSTM models, "
    "then passes the forecasted market state into a trained PPO model to generate optimal FX portfolio weights."
)

with st.sidebar:
    st.header("Interest Rate Differentials")

    st.caption(
        "Enter annual interest rate differentials manually. "
        "Example: if AUD rate is 4.35% and USD rate is 5.25%, enter -0.009."
    )

    rate_diffs = {}

    for currency in CURRENCIES:
        rate_diffs[currency] = st.number_input(
            f"{currency} annual rate differential vs USD",
            value=0.00,
            step=0.001,
            format="%.4f"
        )

    st.divider()

    fx_shock = st.slider(
        "Simulation FX forecast shock",
        min_value=-0.10,
        max_value=0.10,
        value=0.00,
        step=0.01
    )

    carry_shock = st.slider(
        "Simulation carry shock",
        min_value=-0.05,
        max_value=0.05,
        value=0.00,
        step=0.005
    )


# =====================================================
# RUN MODELS
# =====================================================

try:
    ppo_model = load_ppo_model()
    lstm_models, lstm_scalers = load_lstm_models()

    rows = []

    for currency in CURRENCIES:
        raw_df = download_fx_data(TICKERS[currency])
        features = build_lstm_features(raw_df)

        predicted_fx_return = predict_lstm_return(
            currency,
            features,
            lstm_models,
            lstm_scalers
        )

        latest_close = float(raw_df["close"].iloc[-1])
        latest_vol_21 = float(features["vol_21"].iloc[-1])
        latest_momentum_21 = float(features["mom_21"].iloc[-1])

        carry_return = calculate_carry_return(rate_diffs[currency])
        expected_total_return = predicted_fx_return + carry_return

        rows.append({
            "Currency": currency,
            "Ticker": TICKERS[currency],
            "Current FX Rate": latest_close,
            "Predicted FX Return": predicted_fx_return,
            "Carry Return": carry_return,
            "Expected Total Return": expected_total_return,
            "Volatility 21D": latest_vol_21,
            "Momentum 21D": latest_momentum_21,
        })

    forecast_df = pd.DataFrame(rows)

    tab1, tab2, tab3 = st.tabs([
        "Live FX Forecast",
        "PPO Optimal Portfolio",
        "Simulation"
    ])

    # =================================================
    # TAB 1
    # =================================================

    with tab1:
        st.subheader("LSTM-Based 1-Month FX Forecast")

        display_df = forecast_df.copy()

        for col in [
            "Predicted FX Return",
            "Carry Return",
            "Expected Total Return",
            "Volatility 21D",
            "Momentum 21D"
        ]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2%}")

        st.dataframe(display_df, use_container_width=True)

    # =================================================
    # TAB 2
    # =================================================

    with tab2:
        st.subheader("PPO Optimal Portfolio Allocation")

        state = build_ppo_state(forecast_df)

        action, _ = ppo_model.predict(state, deterministic=True)
        weights = convert_action_to_weights(action)

        expected_returns = forecast_df["Expected Total Return"].values
        volatilities = forecast_df["Volatility 21D"].values

        portfolio_return, portfolio_risk, sharpe = calculate_portfolio_metrics(
            weights,
            expected_returns,
            volatilities
        )

        result_df = forecast_df[[
            "Currency",
            "Predicted FX Return",
            "Carry Return",
            "Expected Total Return",
            "Volatility 21D"
        ]].copy()

        result_df["PPO Weight"] = weights

        display_result = result_df.copy()

        for col in [
            "Predicted FX Return",
            "Carry Return",
            "Expected Total Return",
            "Volatility 21D",
            "PPO Weight"
        ]:
            display_result[col] = display_result[col].map(lambda x: f"{x:.2%}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Expected Portfolio Return", f"{portfolio_return:.2%}")
        col2.metric("Portfolio Risk", f"{portfolio_risk:.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

        st.dataframe(display_result, use_container_width=True)

        st.bar_chart(
            pd.DataFrame({
                "Currency": CURRENCIES,
                "Weight": weights
            }).set_index("Currency")
        )

    # =================================================
    # TAB 3
    # =================================================

    with tab3:
        st.subheader("Scenario Simulation")

        sim_df = forecast_df.copy()

        sim_df["Predicted FX Return"] = sim_df["Predicted FX Return"] + fx_shock
        sim_df["Carry Return"] = sim_df["Carry Return"] + carry_shock
        sim_df["Expected Total Return"] = (
            sim_df["Predicted FX Return"] + sim_df["Carry Return"]
        )

        sim_state = build_ppo_state(sim_df)

        sim_action, _ = ppo_model.predict(sim_state, deterministic=True)
        sim_weights = convert_action_to_weights(sim_action)

        sim_expected_returns = sim_df["Expected Total Return"].values
        sim_volatilities = sim_df["Volatility 21D"].values

        sim_return, sim_risk, sim_sharpe = calculate_portfolio_metrics(
            sim_weights,
            sim_expected_returns,
            sim_volatilities
        )

        sim_result = sim_df[[
            "Currency",
            "Predicted FX Return",
            "Carry Return",
            "Expected Total Return",
            "Volatility 21D"
        ]].copy()

        sim_result["Simulated PPO Weight"] = sim_weights

        display_sim = sim_result.copy()

        for col in [
            "Predicted FX Return",
            "Carry Return",
            "Expected Total Return",
            "Volatility 21D",
            "Simulated PPO Weight"
        ]:
            display_sim[col] = display_sim[col].map(lambda x: f"{x:.2%}")

        col1, col2, col3 = st.columns(3)

        col1.metric("Simulated Portfolio Return", f"{sim_return:.2%}")
        col2.metric("Simulated Portfolio Risk", f"{sim_risk:.2%}")
        col3.metric("Simulated Sharpe Ratio", f"{sim_sharpe:.2f}")

        st.dataframe(display_sim, use_container_width=True)

        st.bar_chart(
            pd.DataFrame({
                "Currency": CURRENCIES,
                "Simulated Weight": sim_weights
            }).set_index("Currency")
        )

except Exception as e:
    st.error("App failed to run.")
    st.exception(e)
