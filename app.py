import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from stable_baselines3 import PPO
from datetime import datetime

st.set_page_config(page_title="Nishe FX PPO Portfolio", layout="wide")

# =========================================================
# 1. LOAD MODEL + DATA
# =========================================================
@st.cache_resource
def load_model():
    return PPO.load("ppo_fx_final_model_sharpe")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_step2_features.csv", index_col=0, parse_dates=True)

    rename_map = {}
    for col in df.columns:
        if col.endswith("_ret_1m_target"):
            rename_map[col] = col.replace("_ret_1m_target", "_target")

    if rename_map:
        df = df.rename(columns=rename_map)

    return df

model = load_model()
dataset = load_data()

# =========================================================
# 2. CONFIG
# =========================================================
CURRENCIES = ["EUR", "GBP", "AUD", "JPY", "INR", "CNY"]
CURRENCIES = [c for c in CURRENCIES if f"{c}_target" in dataset.columns]

STATE_FEATURES = [col for col in dataset.columns if not col.endswith("_target")]
STATE_FEATURES = [col for col in STATE_FEATURES if pd.api.types.is_numeric_dtype(dataset[col])]

POSITION_LIMIT = 0.9

FX_TICKERS = {
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "AUD": "AUDUSD=X",
    "JPY": "JPY=X",
    "INR": "INR=X",
    "CNY": "CNY=X"
}

RATE_DIFF_COLS = {
    "EUR": "EUR_rate_diff",
    "GBP": "GBP_rate_diff",
    "AUD": "AUD_rate_diff",
    "JPY": "JPY_rate_diff",
    "INR": "INR_rate_diff",
    "CNY": "CNY_rate_diff"
}

# =========================================================
# 3. HELPERS
# =========================================================
def action_to_weights(action: np.ndarray, position_limit: float = 0.9) -> np.ndarray:
    weights = np.clip(action, -1.0, 1.0).astype(np.float32)
    weights = weights * position_limit
    weights = weights - weights.mean()
    weights = np.clip(weights, -position_limit, position_limit)
    return weights

def get_state_from_row(row: pd.Series, prev_weights: np.ndarray | None = None) -> np.ndarray:
    if prev_weights is None:
        prev_weights = np.zeros(len(CURRENCIES), dtype=np.float32)

    feature_values = row[STATE_FEATURES].values.astype(np.float32)
    obs = np.concatenate([feature_values, prev_weights]).astype(np.float32)
    return obs

def get_carry_vector(row: pd.Series) -> np.ndarray:
    carry_vec = []
    for c in CURRENCIES:
        col = RATE_DIFF_COLS[c]
        if col in row.index and pd.notna(row[col]):
            carry_monthly = (row[col] / 100.0) / 12.0
        else:
            carry_monthly = 0.0
        carry_vec.append(carry_monthly)
    return np.array(carry_vec, dtype=np.float32)

def get_fx_vector(row: pd.Series) -> np.ndarray:
    fx_vec = []
    for c in CURRENCIES:
        col = f"{c}_target"
        if col in row.index and pd.notna(row[col]):
            fx_vec.append(row[col])
        else:
            fx_vec.append(0.0)
    return np.array(fx_vec, dtype=np.float32)

def compute_portfolio_metrics(weights: np.ndarray, row: pd.Series):
    fx_vec = get_fx_vector(row)
    carry_vec = get_carry_vector(row)

    fx_contribution = float(np.dot(weights, fx_vec))
    carry_contribution = float(np.dot(weights, carry_vec))
    total_return = fx_contribution + carry_contribution

    returns_hist = []
    tail_df = dataset.tail(12)

    for _, r in tail_df.iterrows():
        fx_tmp = get_fx_vector(r)
        carry_tmp = get_carry_vector(r)
        ret = float(np.dot(weights, fx_tmp + carry_tmp))
        returns_hist.append(ret)

    returns_hist = np.array(returns_hist, dtype=float)
    volatility = float(np.std(returns_hist, ddof=0))

    if volatility <= 1e-12:
        sharpe = 0.0
    else:
        sharpe = float((returns_hist.mean() / volatility) * np.sqrt(12))

    return {
        "fx_contribution": fx_contribution,
        "carry_contribution": carry_contribution,
        "total_return": total_return,
        "volatility": volatility,
        "sharpe": sharpe,
    }

def make_weights_table(weights: np.ndarray, row: pd.Series) -> pd.DataFrame:
    fx_vec = get_fx_vector(row)
    carry_vec = get_carry_vector(row)

    df = pd.DataFrame({
        "Currency": CURRENCIES,
        "Weight": weights,
        "FX_1M": fx_vec,
        "Carry_1M": carry_vec,
        "Total_Asset_1M": fx_vec + carry_vec
    })

    df["Long/Short"] = np.where(df["Weight"] >= 0, "Long", "Short")
    return df.sort_values("Weight", ascending=False).reset_index(drop=True)

@st.cache_data(ttl=3600)
def fetch_live_fx_rates():
    rates = {}
    for ccy, ticker in FX_TICKERS.items():
        try:
            hist = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
            if hist.empty:
                rates[ccy] = np.nan
            else:
                if isinstance(hist.columns, pd.MultiIndex):
                    close_series = hist[("Close", ticker)]
                else:
                    close_series = hist["Close"]
                rates[ccy] = float(close_series.dropna().iloc[-1])
        except Exception:
            rates[ccy] = np.nan
    return rates

def build_today_row():
    """
    Uses latest dataset row as the base state, then updates
    currently visible live inputs such as FX spot and rate differentials.
    """
    row = dataset.iloc[-1].copy()

    live_fx = fetch_live_fx_rates()

    # Put current FX spots into the row if matching feature names exist
    # Adjust these if you later rename features differently.
    fx_feature_map = {
        "EUR": ["EUR_spot", "EUR_fx", "EUR_close"],
        "GBP": ["GBP_spot", "GBP_fx", "GBP_close"],
        "AUD": ["AUD_spot", "AUD_fx", "AUD_close"],
        "JPY": ["JPY_spot", "JPY_fx", "JPY_close"],
        "INR": ["INR_spot", "INR_fx", "INR_close"],
        "CNY": ["CNY_spot", "CNY_fx", "CNY_close"],
    }

    for ccy, value in live_fx.items():
        for feature_name in fx_feature_map.get(ccy, []):
            if feature_name in row.index and pd.notna(value):
                row[feature_name] = value

    return row, live_fx

def show_input_table(live_fx, row):
    rate_vals = []
    for c in CURRENCIES:
        rate_col = RATE_DIFF_COLS[c]
        rate_vals.append(row[rate_col] if rate_col in row.index else np.nan)

    input_df = pd.DataFrame({
        "Currency": CURRENCIES,
        "Current FX Rate": [live_fx.get(c, np.nan) for c in CURRENCIES],
        "Current Rate Differential": rate_vals
    })
    st.dataframe(input_df, use_container_width=True)

# =========================================================
# 4. TITLE
# =========================================================
st.title("FX Portfolio Optimization using PPO")
st.caption("PPO-based FX allocation using FX change + carry, with Sharpe-oriented training.")

today_row, live_fx = build_today_row()
today_label = datetime.now().strftime("%Y-%m-%d %H:%M")

# =========================================================
# 5. TABS
# =========================================================
tab1, tab2 = st.tabs(["Current Optimal Portfolio", "Scenario Simulation"])

# =========================================================
# 6. TAB 1 - CURRENT PORTFOLIO
# =========================================================
with tab1:
    st.subheader("Current PPO Portfolio")
    st.write(f"Live app refresh timestamp: **{today_label}**")

    st.markdown("### Current Market Inputs")
    show_input_table(live_fx, today_row)

    obs = get_state_from_row(today_row)
    action, _ = model.predict(obs, deterministic=True)
    weights = action_to_weights(action, position_limit=POSITION_LIMIT)

    metrics = compute_portfolio_metrics(weights, today_row)
    weights_df = make_weights_table(weights, today_row)

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected 1M Return", f"{metrics['total_return']:.2%}")
    c2.metric("Estimated Volatility", f"{metrics['volatility']:.2%}")
    c3.metric("Estimated Sharpe Ratio", f"{metrics['sharpe']:.2f}")

    st.markdown("### Portfolio Weights")
    st.dataframe(weights_df, use_container_width=True)

    c4, c5 = st.columns(2)
    c4.metric("FX Contribution", f"{metrics['fx_contribution']:.2%}")
    c5.metric("Carry Contribution", f"{metrics['carry_contribution']:.2%}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(weights_df["Currency"], weights_df["Weight"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Current PPO Portfolio Weights")
    ax.set_ylabel("Weight")
    st.pyplot(fig)

# =========================================================
# 7. TAB 2 - SCENARIO SIMULATION
# =========================================================
with tab2:
    st.subheader("Scenario Simulation")
    st.write("Adjust current FX rates and rate differentials, then run the scenario.")

    scenario_row = today_row.copy()

    st.markdown("### Adjust Inputs")
    sim_cols = st.columns(2)

    scenario_fx = {}
    scenario_rates = {}

    with sim_cols[0]:
        st.markdown("#### FX Rates")
        for c in CURRENCIES:
            default_fx = float(live_fx.get(c, np.nan)) if pd.notna(live_fx.get(c, np.nan)) else 0.0
            scenario_fx[c] = st.number_input(
                f"{c} FX Rate",
                value=default_fx,
                step=0.01,
                format="%.4f"
            )

    with sim_cols[1]:
        st.markdown("#### Rate Differentials")
        for c in CURRENCIES:
            rate_col = RATE_DIFF_COLS[c]
            default_rate = float(scenario_row[rate_col]) if rate_col in scenario_row.index and pd.notna(scenario_row[rate_col]) else 0.0
            scenario_rates[c] = st.number_input(
                f"{c} Rate Differential",
                value=default_rate,
                step=0.10,
                format="%.2f"
            )

    run_sim = st.button("Run Simulation", type="primary")

    if run_sim:
        # Update scenario row with user inputs
        fx_feature_map = {
            "EUR": ["EUR_spot", "EUR_fx", "EUR_close"],
            "GBP": ["GBP_spot", "GBP_fx", "GBP_close"],
            "AUD": ["AUD_spot", "AUD_fx", "AUD_close"],
            "JPY": ["JPY_spot", "JPY_fx", "JPY_close"],
            "INR": ["INR_spot", "INR_fx", "INR_close"],
            "CNY": ["CNY_spot", "CNY_fx", "CNY_close"],
        }

        for c in CURRENCIES:
            rate_col = RATE_DIFF_COLS[c]
            if rate_col in scenario_row.index:
                scenario_row[rate_col] = scenario_rates[c]

            for feature_name in fx_feature_map.get(c, []):
                if feature_name in scenario_row.index:
                    scenario_row[feature_name] = scenario_fx[c]

        sim_obs = get_state_from_row(scenario_row)
        sim_action, _ = model.predict(sim_obs, deterministic=True)
        sim_weights = action_to_weights(sim_action, position_limit=POSITION_LIMIT)

        sim_metrics = compute_portfolio_metrics(sim_weights, scenario_row)
        sim_weights_df = make_weights_table(sim_weights, scenario_row)

        st.markdown("### Simulated Portfolio Output")

        s1, s2, s3 = st.columns(3)
        s1.metric("Expected 1M Return", f"{sim_metrics['total_return']:.2%}")
        s2.metric("Estimated Volatility", f"{sim_metrics['volatility']:.2%}")
        s3.metric("Estimated Sharpe Ratio", f"{sim_metrics['sharpe']:.2f}")

        s4, s5 = st.columns(2)
        s4.metric("FX Contribution", f"{sim_metrics['fx_contribution']:.2%}")
        s5.metric("Carry Contribution", f"{sim_metrics['carry_contribution']:.2%}")

        st.dataframe(sim_weights_df, use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.bar(sim_weights_df["Currency"], sim_weights_df["Weight"])
        ax3.axhline(0, linewidth=1)
        ax3.set_title("Scenario Portfolio Weights")
        ax3.set_ylabel("Weight")
        st.pyplot(fig3)
    else:
        st.info("Adjust the inputs and click Run Simulation.")

# =========================================================
# 8. FOOTER
# =========================================================
st.markdown("---")
st.write("Model: PPO | Return definition: FX change + carry | Deployment mode: live input + inference")
