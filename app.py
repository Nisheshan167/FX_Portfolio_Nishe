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
POSITION_LIMIT = 0.9

PPO_MODEL_PATH = "ppo_fx_final_model_sharpe.zip"
LSTM_BUNDLE_PATH = "lstm_fx_models_bundle.zip"


# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def extract_lstm_bundle():
    extract_dir = os.path.join(tempfile.gettempdir(), "lstm_fx_models")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(LSTM_BUNDLE_PATH, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    return extract_dir


def find_file(root_dir, currency, extension):
    for root, _, files in os.walk(root_dir):
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
        raise ValueError(f"No data downloaded for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].iloc[:, 0]
    else:
        close = data["Close"]

    return pd.DataFrame({"close": close}).dropna()


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

    return df.dropna()[FEATURE_COLUMNS]


def predict_lstm_return(currency, features, models, scalers):
    latest_window = features.tail(LOOKBACK_DAYS)

    if len(latest_window) < LOOKBACK_DAYS:
        raise ValueError(f"Not enough LSTM data for {currency}")

    scaled_window = scalers[currency].transform(latest_window)
    X = np.expand_dims(scaled_window, axis=0)

    prediction = models[currency].predict(X, verbose=0)

    return float(prediction.flatten()[0])


# =====================================================
# PPO-MATCHING FUNCTIONS
# =====================================================

def action_to_weights(action, position_limit=0.9):
    weights = np.clip(action, -1.0, 1.0).astype(np.float64)
    weights = weights * position_limit

    for _ in range(50):
        weights = weights - weights.mean()
        weights = np.clip(weights, -position_limit, position_limit)

    weights = weights - weights.mean()

    return weights.astype(np.float32)


def calculate_carry_return(foreign_rate, usd_rate):
    annual_rate_diff = foreign_rate - usd_rate
    monthly_carry = annual_rate_diff / 12
    return annual_rate_diff, monthly_carry


def build_forecast_table(usd_rate, foreign_rates, lstm_models, lstm_scalers):
    rows = []

    latest_feature_map = {}

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

        latest_feature_map[currency] = features.iloc[-1].to_dict()

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

    return pd.DataFrame(rows), latest_feature_map


def build_state_features(forecast_df, latest_feature_map):
    """
    Recreates PPO-style STATE_FEATURES using latest available market features.

    Training used:
    obs = concatenate([feature_values, prev_weights])

    This app builds feature_values from:
    - LSTM-predicted FX returns
    - rate differentials
    - current technical features
    """

    feature_values = []

    for currency in CURRENCIES:
        tech = latest_feature_map[currency]

        # LSTM forecast replaces target-like next-month FX signal
        feature_values.append(forecast_df.loc[forecast_df["Currency"] == currency, "Predicted FX Return"].iloc[0])

        # Carry/rate-diff feature
        feature_values.append(forecast_df.loc[forecast_df["Currency"] == currency, "Annual Rate Differential"].iloc[0] * 100)

        # Technical features
        for col in FEATURE_COLUMNS:
            feature_values.append(float(tech[col]))

    return np.array(feature_values, dtype=np.float32)


def build_ppo_observation(forecast_df, latest_feature_map, prev_weights=None, expected_obs_dim=None):
    feature_values = build_state_features(forecast_df, latest_feature_map)

    if prev_weights is None:
        prev_weights = np.zeros(len(CURRENCIES), dtype=np.float32)

    obs = np.concatenate([feature_values, prev_weights]).astype(np.float32)

    if expected_obs_dim is not None:
        if len(obs) < expected_obs_dim:
            obs = np.pad(obs, (0, expected_obs_dim - len(obs)), constant_values=0)
        elif len(obs) > expected_obs_dim:
            obs = obs[:expected_obs_dim]

    return obs


def portfolio_metrics(weights, expected_returns):
    portfolio_return = float(np.dot(weights, expected_returns))

    # One-period risk proxy based on dispersion of weighted asset returns
    weighted_asset_returns = weights * expected_returns
    portfolio_risk = float(np.std(weighted_asset_returns))

    if portfolio_risk <= 1e-12:
        sharpe = 0.0
    else:
        sharpe = float((portfolio_return / portfolio_risk) * np.sqrt(12))

    fx_contribution = float(np.dot(weights, expected_returns["fx"])) if isinstance(expected_returns, dict) else None

    return portfolio_return, portfolio_risk, sharpe


def run_ppo_allocation(forecast_df, latest_feature_map, ppo_model):
    expected_obs_dim = int(ppo_model.observation_space.shape[0])

    obs = build_ppo_observation(
        forecast_df=forecast_df,
        latest_feature_map=latest_feature_map,
        prev_weights=np.zeros(len(CURRENCIES), dtype=np.float32),
        expected_obs_dim=expected_obs_dim
    )

    action, _ = ppo_model.predict(obs, deterministic=True)
    weights = action_to_weights(action, position_limit=POSITION_LIMIT)

    fx_returns = forecast_df["Predicted FX Return"].values
    carry_returns = forecast_df["Carry Return"].values
    total_returns = forecast_df["Expected Total Return"].values

    fx_contribution = float(np.dot(weights, fx_returns))
    carry_contribution = float(np.dot(weights, carry_returns))
    total_return = fx_contribution + carry_contribution

    weighted_total = weights * total_returns
    risk = float(np.std(weighted_total))

    sharpe = 0.0 if risk <= 1e-12 else float((total_return / risk) * np.sqrt(12))

    result = forecast_df.copy()
    result["PPO Weight"] = weights
    result["Position Type"] = np.where(result["PPO Weight"] > 0, "Invest / Long", "Borrow / Short")
    result["FX Contribution"] = weights * fx_returns
    result["Carry Contribution"] = weights * carry_returns
    result["Total Contribution"] = weights * total_returns

    return result, total_return, risk, sharpe, fx_contribution, carry_contribution


def format_percent_columns(df, cols):
    df = df.copy()
    for col in cols:
        df[col] = df[col].map(lambda x: f"{x:.2%}")
    return df


# =====================================================
# APP UI
# =====================================================

st.title("FX Portfolio Optimisation using LSTM Forecasting + PPO")

st.write(
    "This app predicts next 1-month FX returns using LSTM models, adds monthly carry from interest-rate differentials, "
    "and passes the resulting state into a trained PPO model that was trained to maximise rolling Sharpe ratio."
)

try:
    ppo_model = load_ppo_model()
    lstm_models, lstm_scalers = load_lstm_models()

    live_df, live_feature_map = build_forecast_table(
        usd_rate=DEFAULT_USD_RATE,
        foreign_rates=DEFAULT_FOREIGN_RATES,
        lstm_models=lstm_models,
        lstm_scalers=lstm_scalers
    )

    (
        live_result,
        live_return,
        live_risk,
        live_sharpe,
        live_fx_contribution,
        live_carry_contribution
    ) = run_ppo_allocation(live_df, live_feature_map, ppo_model)

    tab1, tab2, tab3 = st.tabs([
        "Live FX Forecast",
        "PPO Optimal Portfolio",
        "Simulation"
    ])

    with tab1:
        st.subheader("Live LSTM FX Forecast + Carry")

        st.info(
            "Live mode uses default interest-rate assumptions. "
            "Only the Simulation tab allows rate changes."
        )

        display_df = live_df[[
            "Currency",
            "Current FX Rate",
            "Foreign Interest Rate",
            "USD Interest Rate",
            "Annual Rate Differential",
            "Predicted FX Return",
            "Carry Return",
            "Expected Total Return",
            "Volatility 21D",
        ]]

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

    with tab2:
        st.subheader("PPO Long-Short Portfolio Allocation")

        col1, col2, col3 = st.columns(3)
        col1.metric("Expected 1-Month Portfolio Return", f"{live_return:.2%}")
        col2.metric("Risk Proxy", f"{live_risk:.2%}")
        col3.metric("Sharpe Ratio", f"{live_sharpe:.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("FX Contribution", f"{live_fx_contribution:.2%}")
        col5.metric("Carry Contribution", f"{live_carry_contribution:.2%}")
        col6.metric("Weight Sum", f"{live_result['PPO Weight'].sum():.4f}")

        portfolio_df = live_result[[
            "Currency",
            "Position Type",
            "Predicted FX Return",
            "Carry Return",
            "Expected Total Return",
            "PPO Weight",
            "FX Contribution",
            "Carry Contribution",
            "Total Contribution",
        ]]

        portfolio_df = format_percent_columns(
            portfolio_df,
            [
                "Predicted FX Return",
                "Carry Return",
                "Expected Total Return",
                "PPO Weight",
                "FX Contribution",
                "Carry Contribution",
                "Total Contribution",
            ]
        )

        st.dataframe(portfolio_df, use_container_width=True)

        st.bar_chart(live_result[["Currency", "PPO Weight"]].set_index("Currency"))

    with tab3:
        st.subheader("Simulation: Change Interest Rates")

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

        sim_df, sim_feature_map = build_forecast_table(
            usd_rate=sim_usd_rate,
            foreign_rates=sim_foreign_rates,
            lstm_models=lstm_models,
            lstm_scalers=lstm_scalers
        )

        (
            sim_result,
            sim_return,
            sim_risk,
            sim_sharpe,
            sim_fx_contribution,
            sim_carry_contribution
        ) = run_ppo_allocation(sim_df, sim_feature_map, ppo_model)

        st.divider()

        col1, col2, col3 = st.columns(3)
        col1.metric("Simulated 1-Month Portfolio Return", f"{sim_return:.2%}")
        col2.metric("Risk Proxy", f"{sim_risk:.2%}")
        col3.metric("Sharpe Ratio", f"{sim_sharpe:.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("FX Contribution", f"{sim_fx_contribution:.2%}")
        col5.metric("Carry Contribution", f"{sim_carry_contribution:.2%}")
        col6.metric("Weight Sum", f"{sim_result['PPO Weight'].sum():.4f}")

        sim_display = sim_result[[
            "Currency",
            "Position Type",
            "Foreign Interest Rate",
            "USD Interest Rate",
            "Annual Rate Differential",
            "Predicted FX Return",
            "Carry Return",
            "Expected Total Return",
            "PPO Weight",
            "Total Contribution",
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
                "PPO Weight",
                "Total Contribution",
            ]
        )

        st.dataframe(sim_display, use_container_width=True)

        st.bar_chart(sim_result[["Currency", "PPO Weight"]].set_index("Currency"))

except Exception as e:
    st.error("App failed to run.")
    st.exception(e)
