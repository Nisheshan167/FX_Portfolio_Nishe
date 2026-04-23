import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

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
        col = f"{c}_rate_diff"
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

    # Approximate volatility using recent PPO-style rolling realized returns
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
        "fx_vec": fx_vec,
        "carry_vec": carry_vec
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

# =========================================================
# 4. TITLE
# =========================================================
st.title("FX Portfolio Optimization using PPO")
st.caption("PPO-based FX allocation using FX change + carry, with Sharpe-oriented training.")

latest_row = dataset.iloc[-1].copy()
latest_date = dataset.index[-1]

# =========================================================
# 5. TABS
# =========================================================
tab1, tab2 = st.tabs(["Current Optimal Portfolio", "Scenario Simulation"])

# =========================================================
# 6. TAB 1 - CURRENT PORTFOLIO
# =========================================================
with tab1:
    st.subheader("Current PPO Portfolio")
    st.write(f"Latest observation date: **{latest_date.date()}**")

    obs = get_state_from_row(latest_row)
    action, _ = model.predict(obs, deterministic=True)
    weights = action_to_weights(action, position_limit=POSITION_LIMIT)

    metrics = compute_portfolio_metrics(weights, latest_row)
    weights_df = make_weights_table(weights, latest_row)

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
    ax.set_title("PPO Portfolio Weights")
    ax.set_ylabel("Weight")
    st.pyplot(fig)

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
    st.write("Adjust selected macro inputs and inspect how the PPO allocation changes.")

    scenario_row = latest_row.copy()

    slider_cols = st.columns(3)

    editable_features = [
        "EUR_rate_diff", "GBP_rate_diff", "AUD_rate_diff",
        "JPY_rate_diff", "INR_rate_diff", "CNY_rate_diff",
        "EU_cpi_diff", "UK_cpi_diff", "AU_cpi_diff",
        "JP_cpi_diff", "IN_cpi_diff", "CN_cpi_diff"
    ]

    existing_editable = [col for col in editable_features if col in scenario_row.index]

    for i, feature in enumerate(existing_editable):
        with slider_cols[i % 3]:
            current_val = float(scenario_row[feature]) if pd.notna(scenario_row[feature]) else 0.0
            scenario_row[feature] = st.slider(
                feature,
                min_value=float(current_val - 5),
                max_value=float(current_val + 5),
                value=float(current_val),
                step=0.1
            )

    st.markdown("### Simulated Portfolio")

    sim_obs = get_state_from_row(scenario_row)
    sim_action, _ = model.predict(sim_obs, deterministic=True)
    sim_weights = action_to_weights(sim_action, position_limit=POSITION_LIMIT)

    sim_metrics = compute_portfolio_metrics(sim_weights, scenario_row)
    sim_weights_df = make_weights_table(sim_weights, scenario_row)

    s1, s2, s3 = st.columns(3)
    s1.metric("Expected 1M Return", f"{sim_metrics['total_return']:.2%}")
    s2.metric("Estimated Volatility", f"{sim_metrics['volatility']:.2%}")
    s3.metric("Estimated Sharpe Ratio", f"{sim_metrics['sharpe']:.2f}")

    st.dataframe(sim_weights_df, use_container_width=True)

    s4, s5 = st.columns(2)
    s4.metric("FX Contribution", f"{sim_metrics['fx_contribution']:.2%}")
    s5.metric("Carry Contribution", f"{sim_metrics['carry_contribution']:.2%}")

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.bar(sim_weights_df["Currency"], sim_weights_df["Weight"])
    ax3.axhline(0, linewidth=1)
    ax3.set_title("Scenario Portfolio Weights")
    ax3.set_ylabel("Weight")
    st.pyplot(fig3)

# =========================================================
# 8. FOOTER
# =========================================================
st.markdown("---")
st.write("Model: PPO | Return definition: FX change + carry | Deployment mode: inference only")
