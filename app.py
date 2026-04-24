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

st.set_page_config(
    page_title="FX Portfolio Optimisation | LSTM + PPO",
    layout="wide"
)

# =====================================================
# CONFIG
# =====================================================

CURRENCIES = ["EUR", "GBP", "AUD", "JPY", "INR", "CNY"]

TICKERS = {
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "AUD": "AUDUSD=X",
    "JPY": "JPY=X",
    "INR": "INR=X",
    "CNY": "CNY=X",
}

# Default annual interest rates
# Update these when needed
DEFAULT_USD_RATE = 0.0525

DEFAULT_FOREIGN_RATES = {
    "EUR": 0.0400,
    "GBP": 0.0525,
    "AUD": 0.0435,
    "JPY": 0.0010,
    "INR": 0.0650,
    "CNY": 0.0345,
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
# LOAD MODELS
# =====================================================

@st.cache_resource
def extract_lstm_bundle():
    extract_dir = os.path.join(tempfile.gettempdir(), "lstm_fx_models")

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(LSTM_BUNDLE_PATH, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    return extract_dir


def find_file(root_dir, currency, extension):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_lower = file.lower()
            if currency.lower() in file_lower and file_lower.endswith(extension):
                return os.path.join(root, file)

    raise FileNotFoundError(f"Could not find {currency} file ending with {extension}")


@st.cache_resource
def load_lstm_models():
    model_dir = extract_lstm_bundle()

    models = {}
    scalers = {}

    for currency in CURRENCIES:
        model_path = find_file(model_dir, currency, ".keras")
        scaler_path = find_file(model_dir, currency, ".pkl")

        models[currency] = load_model(model_path)
        scalers[currency] = joblib.load(scaler_path)

    return models, scalers


@st.cache_resource
def load_ppo_model():
    return PPO.load(PPO_MODEL_PATH)


# =====================================================
# DATA FUNCTIONS
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
        raise ValueError(f"No FX data downloaded for {ticker}")

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
    latest_window = features.tail(LOOKBACK_DAYS)

    if len(latest_window) < LOOKBACK_DAYS:
        raise ValueError(f"Not enough data for {currency}")

    scaler = scalers[currency]
    model = models[currency]

    scaled_window = scaler.transform(latest_window)
    X = np.expand_dims(scaled_window, axis=0)

    prediction = model.predict(X, verbose=0)

    return float(prediction.flatten()[0])


# =====================================================
# FINANCE FUNCTIONS
# =====================================================

def calculate_carry_return(foreign_rate, usd_rate):
    annual_rate_diff = foreign_rate - usd_rate
    monthly_carry = annual_rate_diff / 12
    return annual_rate_diff, monthly_carry


def build_forecast_table(usd_rate, foreign_rates, lstm_models, lstm_scalers):
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

        current_fx_rate = float(raw_df["close"].iloc[-1])
        volatility_21d = float(features["vol_21"].iloc[-1])
        momentum_21d = float(features["mom_21"].iloc[-1])

        annual_rate_diff, carry_return = calculate_carry_return(
            foreign_rates[currency],
            usd_rate
        )

        expected_total_return = predicted_fx_return + carry_return

        rows.append({
            "Currency": currency,
            "Ticker": TICKERS[currency],
            "Current FX Rate": current_fx_rate,
            "Foreign Interest Rate": foreign_rates[currency],
            "USD Interest Rate": usd_rate,
            "Annual Rate Differential": annual_rate_diff,
            "Predicted FX Return": predicted_fx_return,
            "Carry Return": carry_return,
            "Expected Total Return": expected_total_return,
            "Volatility 21D": volatility_21d,
            "Momentum 21D": momentum_21d,
        })

    return pd.DataFrame(rows)


def build_ppo_state(df):
    predicted_fx = df["Predicted FX Return"].values
    carry = df["Carry Return"].values
    total = df["Expected Total Return"].values
    vol = df["Volatility 21D"].values
    momentum = df["Momentum 21D"].values
    spot = df["Current FX Rate"].values
    rate_diff = df["Annual Rate Differential"].values

    risk_adjusted = total / (vol + 1e-8)

    state = np.concatenate([
        predicted_fx,
        carry,
        total,
        vol,
        momentum,
        spot,
        rate_diff,
        risk_adjusted,
    ]).astype(np.float32)

    if len(state) < PPO_STATE_SIZE:
        state = np.pad(state, (0, PPO_STATE_SIZE - len(state)), constant_values=0)

    if len(state) > PPO_STATE_SIZE:
        state = state[:PPO_STATE_SIZE]

    return state


def action_to_weights(action):
    action = np.array(action).flatten()

    exp_action = np.exp(action - np.max(action))
    weights = exp_action / exp_action.sum()

    return weights


def portfolio_metrics(weights, expected_returns, volatilities):
    portfolio_return = float(np.dot(weights, expected_returns))
    portfolio_risk = float(np.sqrt(np.dot(weights ** 2, volatilities ** 2)))
    sharpe = portfolio_return / portfolio_risk if portfolio_risk != 0 else 0

    return portfolio_return, portfolio_risk, sharpe


def run_ppo_allocation(df, ppo_model):
    state = build_ppo_state(df)
    action, _ = ppo_model.predict(state, deterministic=True)

    weights = action_to_weights(action)

    expected_returns = df["Expected Total Return"].values
    volatilities = df["Volatility 21D"].values

    port_return, port_risk, sharpe = portfolio_metrics(
        weights,
        expected_returns,
        volatilities
    )

    result = df.copy()
    result["PPO Weight"] = weights

    return result, port_return, port_risk, sharpe


def format_percent_columns(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = df[col].map(lambda x: f"{x:.2%}")
    return df


# =====================================================
# APP
# =====================================================

st.title("FX Portfolio Optimisation using LSTM Forecasting + PPO")

st.write(
    "The app predicts next 1-month FX returns using LSTM models, calculates carry return from interest-rate differentials, "
    "and feeds the combined expected return into a trained PPO model to generate optimal currency weights."
)

try:
    ppo_model = load_ppo_model()
    lstm_models, lstm_scalers = load_lstm_models()

    live_df = build_forecast_table(
        usd_rate=DEFAULT_USD_RATE,
        foreign_rates=DEFAULT_FOREIGN_RATES,
        lstm_models=lstm_models,
        lstm_scalers=lstm_scalers
    )

    live_result, live_return, live_risk, live_sharpe = run_ppo_allocation(
        live_df,
        ppo_model
    )

    tab1, tab2, tab3 = st.tabs([
        "Live FX Forecast",
        "PPO Optimal Portfolio",
        "Simulation"
    ])

    # =====================================================
    # TAB 1: LIVE FORECAST
    # =====================================================

    with tab1:
        st.subheader("Live LSTM FX Forecast + Carry Return")

        st.info(
            "Live mode uses default interest-rate assumptions inside the app. "
            "Rate editing is only available in the Simulation tab."
        )

        display_cols = [
            "Currency",
            "Current FX Rate",
            "Foreign Interest Rate",
            "USD Interest Rate",
            "Annual Rate Differential",
            "Predicted FX Return",
            "Carry Return",
            "Expected Total Return",
            "Volatility 21D",
        ]

        display_df = live_df[display_cols]

        display_df = format_percent_columns(
            display_df,
            [
                "Foreign Interest Rate",
                "USD Interest Rate",
                "Annual Rate Differential",
                "Predicted FX Return",
                "Carry Return",
                "Expected Total Return",
                "Volatility 21D",
            ]
        )

        st.dataframe(display_df, use_container_width=True)

    # =====================================================
    # TAB 2: PPO PORTFOLIO
    # =====================================================

    with tab2:
        st.subheader("PPO Optimal Portfolio Allocation")

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected 1-Month Portfolio Return", f"{live_return:.2%}")
        col2.metric("Portfolio Risk", f"{live_risk:.2%}")
        col3.metric("Sharpe Ratio", f"{live_sharpe:.2f}")

        portfolio_df = live_result[[
            "Currency",
            "Predicted FX Return",
            "Carry Return",
            "Expected Total Return",
            "Volatility 21D",
            "PPO Weight"
        ]]

        portfolio_df = format_percent_columns(
            portfolio_df,
            [
                "Predicted FX Return",
                "Carry Return",
                "Expected Total Return",
                "Volatility 21D",
                "PPO Weight",
            ]
        )

        st.dataframe(portfolio_df, use_container_width=True)

        chart_df = live_result[["Currency", "PPO Weight"]].set_index("Currency")
        st.bar_chart(chart_df)

    # =====================================================
    # TAB 3: SIMULATION
    # =====================================================

    with tab3:
        st.subheader("Simulation: Change Interest Rates and Recalculate Portfolio")

        st.write(
            "Here, you can change the USD and foreign interest rates. "
            "The app recalculates the carry return, combines it with the LSTM FX forecast, "
            "and sends the updated state into the PPO model."
        )

        sim_usd_rate = st.number_input(
            "USD Interest Rate",
            value=DEFAULT_USD_RATE,
            step=0.0025,
            format="%.4f"
        )

        sim_foreign_rates = {}

        cols = st.columns(3)

        for i, currency in enumerate(CURRENCIES):
            with cols[i % 3]:
                sim_foreign_rates[currency] = st.number_input(
                    f"{currency} Interest Rate",
                    value=DEFAULT_FOREIGN_RATES[currency],
                    step=0.0025,
                    format="%.4f"
                )

        sim_df = build_forecast_table(
            usd_rate=sim_usd_rate,
            foreign_rates=sim_foreign_rates,
            lstm_models=lstm_models,
            lstm_scalers=lstm_scalers
        )

        sim_result, sim_return, sim_risk, sim_sharpe = run_ppo_allocation(
            sim_df,
            ppo_model
        )

        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.metric("Simulated 1-Month Portfolio Return", f"{sim_return:.2%}")
        col2.metric("Simulated Portfolio Risk", f"{sim_risk:.2%}")
        col3.metric("Simulated Sharpe Ratio", f"{sim_sharpe:.2f}")

        sim_display = sim_result[[
            "Currency",
            "Foreign Interest Rate",
            "USD Interest Rate",
            "Annual Rate Differential",
            "Predicted FX Return",
            "Carry Return",
            "Expected Total Return",
            "Volatility 21D",
            "PPO Weight"
        ]]

        sim_display = format_percent_columns(
            sim_display,
            [
                "Foreign Interest Rate",
                "USD Interest Rate",
                "Annual Rate Differential",
                "Predicted FX Return",
                "Carry Return",
                "Expected Total Return",
                "Volatility 21D",
                "PPO Weight",
            ]
        )

        st.dataframe(sim_display, use_container_width=True)

        sim_chart_df = sim_result[["Currency", "PPO Weight"]].set_index("Currency")
        st.bar_chart(sim_chart_df)

except Exception as e:
    st.error("App failed to run.")
    st.exception(e)
