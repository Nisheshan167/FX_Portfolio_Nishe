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

FX_DISPLAY_HELP = {
    "EUR": "EUR/USD",
    "GBP": "GBP/USD",
    "AUD": "AUD/USD",
    "JPY": "USD/JPY",
    "INR": "USD/INR",
    "CNY": "USD/CNY"
}

# Optional feature names in dataset, if present
FX_FEATURE_MAP = {
    "EUR": ["EUR_spot", "EUR_fx", "EUR_close"],
    "GBP": ["GBP_spot", "GBP_fx", "GBP_close"],
    "AUD": ["AUD_spot", "AUD_fx", "AUD_close"],
    "JPY": ["JPY_spot", "JPY_fx", "JPY_close"],
    "INR": ["INR_spot", "INR_fx", "INR_close"],
    "CNY": ["CNY_spot", "CNY_fx", "CNY_close"],
}

# =========================================================
# 3. HELPERS
# =========================================================
def action_to_weights(action: np.ndarray, position_limit: float = 0.9) -> np.ndarray:
    """
    Same self-financing logic as model training:
    - bounded by position_limit
    - weights sum to zero
    """
    weights = np.clip(action, -1.0, 1.0).astype(np.float64)
    weights = weights * position_limit

    for _ in range(50):
        weights = weights - weights.mean()
        weights = np.clip(weights, -position_limit, position_limit)

    weights = weights - weights.mean()
    return weights.astype(np.float32)

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

    # Estimate volatility from the latest 12 historical rows using the same weights
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
            hist = yf.download(
                ticker,
                period="5d",
                interval="1d",
                progress=False,
                auto_adjust=False
            )

            if hist.empty:
                rates[ccy] = np.nan
                continue

            if isinstance(hist.columns, pd.MultiIndex):
                close_series = hist[("Close", ticker)]
            else:
                close_series = hist["Close"]

            close_series = close_series.dropna()
            rates[ccy] = float(close_series.iloc[-1]) if len(close_series) > 0 else np.nan

        except Exception:
            rates[ccy] = np.nan

    return rates

def build_current_row():
    """
    Start from latest dataset row and overwrite any matching FX spot feature columns
    with current live FX, if those columns exist in the trained feature set.
    """
    row = dataset.iloc[-1].copy()
    live_fx = fetch_live_fx_rates()

    for ccy, live_val in live_fx.items():
        if pd.isna(live_val):
            continue
        for feature_name in FX_FEATURE_MAP.get(ccy, []):
            if feature_name in row.index:
                row[feature_name] = live_val

    return row, live_fx

def show_current_inputs_table(live_fx: dict, row: pd.Series):
    data = []
    for c in CURRENCIES:
        rate_col = RATE_DIFF_COLS[c]
        rate_diff = row[rate_col] if rate_col in row.index else np.nan
        data.append({
            "Currency": c,
            "FX Pair": FX_DISPLAY_HELP[c],
            "Current FX Rate": live_fx.get(c, np.nan),
            "Rate Differential vs USD (%)": rate_diff
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

# =========================================================
# 4. TITLE
# =========================================================
st.title("FX Portfolio Optimization using PPO")
st.caption("PPO-based FX allocation using FX change + carry, with self-financing zero-sum weights.")

current_row, live_fx = build_current_row()
timestamp_label = datetime.now().strftime("%Y-%m-%d %H:%M")

# =========================================================
# 5. TABS
# =========================================================
tab1, tab2 = st.tabs(["Current Optimal Portfolio", "Scenario Simulation"])

# =========================================================
# 6. TAB 1 - CURRENT OPTIMAL PORTFOLIO
# =========================================================
with tab1:
    st.subheader("Current PPO Portfolio")
    st.write(f"Refresh timestamp: **{timestamp_label}**")

    st.markdown("### Current Market Inputs")
    show_current_inputs_table(live_fx, current_row)

    obs = get_state_from_row(current_row)
    action, _ = model.predict(obs, deterministic=True)
    weights = action_to_weights(action, position_limit=POSITION_LIMIT)

    metrics = compute_portfolio_metrics(weights, current_row)
    weights_df = make_weights_table(weights, current_row)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected 1M Return", f"{metrics['total_return']:.2%}")
    c2.metric("Estimated Volatility", f"{metrics['volatility']:.2%}")
    c3.metric("Estimated Sharpe Ratio", f"{metrics['sharpe']:.2f}")
    c4.metric("Sum of Weights", f"{weights.sum():.10f}")

    st.markdown("### Portfolio Weights")
    st.dataframe(weights_df, use_container_width=True)

    c5, c6 = st.columns(2)
    c5.metric("FX Contribution", f"{metrics['fx_contribution']:.2%}")
    c6.metric("Carry Contribution", f"{metrics['carry_contribution']:.2%}")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(weights_df["Currency"], weights_df["Weight"])
    ax1.axhline(0, linewidth=1)
    ax1.set_title("Current PPO Portfolio Weights")
    ax1.set_ylabel("Weight")
    st.pyplot(fig1)

    decomp_df = pd.DataFrame({
        "Component": ["FX", "Carry"],
        "Value": [metrics["fx_contribution"], metrics["carry_contribution"]]
    })

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.bar(decomp_df["Component"], decomp_df["Value"])
    ax2.axhline(0, linewidth=1)
    ax2.set_title("1-Month Return Decomposition")
    ax2.set_ylabel("Contribution")
    st.pyplot(fig2)

# =========================================================
# 7. TAB 2 - SCENARIO SIMULATION
# =========================================================
with tab2:
    st.subheader("Scenario Simulation")
    st.write("Adjust current FX rates and rate differentials, then click **Run Simulation**.")

    scenario_row = current_row.copy()

    left, right = st.columns(2)

    scenario_fx = {}
    scenario_rates = {}

    with left:
        st.markdown("### Scenario FX Rates")
        for c in CURRENCIES:
            default_fx = live_fx.get(c, np.nan)
            if pd.isna(default_fx):
                default_fx = 0.0

            scenario_fx[c] = st.number_input(
                f"{c} FX Rate ({FX_DISPLAY_HELP[c]})",
                value=float(default_fx),
                step=0.01,
                format="%.4f"
            )

    with right:
        st.markdown("### Scenario Rate Differentials")
        for c in CURRENCIES:
            rate_col = RATE_DIFF_COLS[c]
            default_rate = float(scenario_row[rate_col]) if rate_col in scenario_row.index and pd.notna(scenario_row[rate_col]) else 0.0

            scenario_rates[c] = st.number_input(
                f"{c} Rate Differential vs USD (%)",
                value=default_rate,
                step=0.10,
                format="%.2f"
            )

    run_sim = st.button("Run Simulation", type="primary")

    if run_sim:
        for c in CURRENCIES:
            rate_col = RATE_DIFF_COLS[c]
            if rate_col in scenario_row.index:
                scenario_row[rate_col] = scenario_rates[c]

            for feature_name in FX_FEATURE_MAP.get(c, []):
                if feature_name in scenario_row.index:
                    scenario_row[feature_name] = scenario_fx[c]

        sim_obs = get_state_from_row(scenario_row)
        sim_action, _ = model.predict(sim_obs, deterministic=True)
        sim_weights = action_to_weights(sim_action, position_limit=POSITION_LIMIT)

        sim_metrics = compute_portfolio_metrics(sim_weights, scenario_row)
        sim_weights_df = make_weights_table(sim_weights, scenario_row)

        st.markdown("### Simulated Portfolio Output")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Expected 1M Return", f"{sim_metrics['total_return']:.2%}")
        s2.metric("Estimated Volatility", f"{sim_metrics['volatility']:.2%}")
        s3.metric("Estimated Sharpe Ratio", f"{sim_metrics['sharpe']:.2f}")
        s4.metric("Sum of Weights", f"{sim_weights.sum():.10f}")

        s5, s6 = st.columns(2)
        s5.metric("FX Contribution", f"{sim_metrics['fx_contribution']:.2%}")
        s6.metric("Carry Contribution", f"{sim_metrics['carry_contribution']:.2%}")

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
st.write("Model: PPO | Return definition: FX change + carry | Constraint: self-financing (weights sum to zero)")
