import re
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
ALL_CURRENCIES = ["EUR", "GBP", "AUD", "JPY", "INR", "CNY"]
CURRENCIES = [c for c in ALL_CURRENCIES if f"{c}_target" in dataset.columns]

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

FX_FEATURE_CANDIDATES = {
    "EUR": ["EUR_spot", "EUR_fx", "EUR_close", "EURUSD", "EUR_price"],
    "GBP": ["GBP_spot", "GBP_fx", "GBP_close", "GBPUSD", "GBP_price"],
    "AUD": ["AUD_spot", "AUD_fx", "AUD_close", "AUDUSD", "AUD_price"],
    "JPY": ["JPY_spot", "JPY_fx", "JPY_close", "JPYUSD", "JPY_price"],
    "INR": ["INR_spot", "INR_fx", "INR_close", "INRUSD", "INR_price"],
    "CNY": ["CNY_spot", "CNY_fx", "CNY_close", "CNYUSD", "CNY_price"],
}


# =========================================================
# 3. GENERIC HELPERS
# =========================================================
def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def normalize_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", col.lower())


def is_rate_col(col: str) -> bool:
    c = col.lower()
    return "rate" in c or "yield" in c or "policy" in c or "interest" in c


def is_diff_like(col: str) -> bool:
    c = col.lower()
    return ("diff" in c) or ("spread" in c) or ("carry" in c)


def is_usd_like(col: str) -> bool:
    c = col.lower()
    return ("usd" in c) or ("us_" in c) or c.startswith("us") or ("fed" in c)


def find_first_existing(candidates, columns):
    for c in candidates:
        if c in columns:
            return c
    return None


# =========================================================
# 4. RATE COLUMN DISCOVERY
# =========================================================
def discover_rate_structure(columns):
    """
    Auto-detect:
    - USD/base rate column
    - local absolute rate columns
    - rate differential columns
    """
    cols = list(columns)

    usd_candidates = []
    for col in cols:
        cl = col.lower()
        if is_rate_col(col) and not is_diff_like(col) and is_usd_like(col):
            score = 0
            if "usd" in cl:
                score += 5
            if "rate" in cl:
                score += 3
            if "policy" in cl:
                score += 2
            if "yield" in cl:
                score += 1
            usd_candidates.append((score, col))

    usd_rate_col = max(usd_candidates, default=(None, None))[1]

    local_abs_cols = {}
    diff_cols = {}

    for ccy in CURRENCIES:
        abs_candidates = []
        diff_candidates = []

        for col in cols:
            cl = col.lower()

            if ccy.lower() not in cl:
                continue

            if not is_rate_col(col):
                continue

            if is_diff_like(col):
                score = 0
                if "rate" in cl:
                    score += 3
                if "diff" in cl:
                    score += 4
                if "spread" in cl:
                    score += 2
                if "carry" in cl:
                    score += 1
                diff_candidates.append((score, col))
            else:
                score = 0
                if "rate" in cl:
                    score += 4
                if "policy" in cl:
                    score += 2
                if "yield" in cl:
                    score += 1
                abs_candidates.append((score, col))

        local_abs_cols[ccy] = max(abs_candidates, default=(None, None))[1]
        diff_cols[ccy] = max(diff_candidates, default=(None, None))[1]

    return {
        "usd_rate_col": usd_rate_col,
        "local_abs_cols": local_abs_cols,
        "diff_cols": diff_cols,
    }


RATE_INFO = discover_rate_structure(dataset.columns)


# =========================================================
# 5. FX / RATE ACCESSORS
# =========================================================
def get_fx_vector(row: pd.Series) -> np.ndarray:
    fx_vec = []
    for c in CURRENCIES:
        target_col = f"{c}_target"
        fx_vec.append(safe_float(row[target_col], 0.0) if target_col in row.index else 0.0)
    return np.array(fx_vec, dtype=np.float32)


def get_usd_rate(row: pd.Series) -> float:
    usd_col = RATE_INFO["usd_rate_col"]
    if usd_col and usd_col in row.index:
        return safe_float(row[usd_col], 0.0)
    return 0.0


def get_absolute_rate_for_currency(row: pd.Series, ccy: str) -> float:
    """
    Priority:
    1. use explicit local absolute rate col if present
    2. else derive from USD/base + diff
    3. else 0
    """
    local_abs_col = RATE_INFO["local_abs_cols"].get(ccy)
    diff_col = RATE_INFO["diff_cols"].get(ccy)

    if local_abs_col and local_abs_col in row.index and pd.notna(row[local_abs_col]):
        return safe_float(row[local_abs_col], 0.0)

    usd_rate = get_usd_rate(row)

    if diff_col and diff_col in row.index and pd.notna(row[diff_col]):
        return usd_rate + safe_float(row[diff_col], 0.0)

    return 0.0


def get_absolute_rate_dict(row: pd.Series) -> dict:
    return {c: get_absolute_rate_for_currency(row, c) for c in CURRENCIES}


def get_carry_vector(row: pd.Series) -> np.ndarray:
    """
    Monthly carry from absolute local rate - absolute USD/base rate.
    """
    usd_rate = get_usd_rate(row)
    carry_vec = []

    for c in CURRENCIES:
        local_rate = get_absolute_rate_for_currency(row, c)
        carry_annual = (local_rate - usd_rate) / 100.0
        carry_monthly = carry_annual / 12.0
        carry_vec.append(carry_monthly)

    return np.array(carry_vec, dtype=np.float32)


# =========================================================
# 6. ACTION / STATE / METRICS
# =========================================================
def action_to_weights(action: np.ndarray, position_limit: float = 0.9) -> np.ndarray:
    weights = np.asarray(action, dtype=np.float64).flatten()
    weights = np.clip(weights, -1.0, 1.0) * position_limit

    for _ in range(100):
        weights = weights - weights.mean()
        weights = np.clip(weights, -position_limit, position_limit)

    weights = weights - weights.mean()

    if len(weights) > 0:
        weights[-1] = -weights[:-1].sum()

    weights = np.clip(weights, -position_limit, position_limit)
    weights = weights - weights.mean()

    return weights.astype(np.float32)


def get_state_from_row(row: pd.Series, prev_weights: np.ndarray | None = None) -> np.ndarray:
    if prev_weights is None:
        prev_weights = np.zeros(len(CURRENCIES), dtype=np.float32)

    feature_values = row[STATE_FEATURES].values.astype(np.float32)
    obs = np.concatenate([feature_values, prev_weights]).astype(np.float32)
    return obs


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
    abs_rates = get_absolute_rate_dict(row)
    usd_rate = get_usd_rate(row)

    df = pd.DataFrame({
        "Currency": CURRENCIES,
        "Weight": weights,
        "Absolute Rate (%)": [abs_rates[c] for c in CURRENCIES],
        "USD Rate (%)": [usd_rate] * len(CURRENCIES),
        "FX_1M": fx_vec,
        "Carry_1M": carry_vec,
        "Total_Asset_1M": fx_vec + carry_vec
    })

    df["Long/Short"] = np.where(df["Weight"] >= 0, "Long", "Short")
    return df.sort_values("Weight", ascending=False).reset_index(drop=True)


def run_model_on_row(row: pd.Series):
    obs = get_state_from_row(row)
    action, _ = model.predict(obs, deterministic=True)
    weights = action_to_weights(action, position_limit=POSITION_LIMIT)
    metrics = compute_portfolio_metrics(weights, row)
    weights_df = make_weights_table(weights, row)
    return obs, weights, metrics, weights_df


# =========================================================
# 7. LIVE FX + FEATURE UPDATES
# =========================================================
@st.cache_data(ttl=3600)
def fetch_live_fx_rates():
    rates = {}
    for ccy, ticker in FX_TICKERS.items():
        try:
            hist = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)

            if hist.empty:
                rates[ccy] = np.nan
                continue

            if isinstance(hist.columns, pd.MultiIndex):
                if ("Close", ticker) in hist.columns:
                    close_series = hist[("Close", ticker)]
                else:
                    close_series = hist.xs("Close", axis=1, level=0).iloc[:, 0]
            else:
                close_series = hist["Close"]

            close_series = close_series.dropna()
            rates[ccy] = float(close_series.iloc[-1]) if len(close_series) > 0 else np.nan

        except Exception:
            rates[ccy] = np.nan

    return rates


def update_fx_features_in_row(row: pd.Series, fx_inputs: dict) -> pd.Series:
    row = row.copy()

    for c, val in fx_inputs.items():
        for feature_name in FX_FEATURE_CANDIDATES.get(c, []):
            if feature_name in row.index:
                row[feature_name] = val

    return row


def update_rate_features_in_row(row: pd.Series, scenario_abs_rates: dict) -> pd.Series:
    """
    Updates whichever rate features actually exist in the dataset:
    - local absolute rate col if present
    - rate differential col if present, derived from local abs - USD/base
    """
    row = row.copy()
    usd_rate = get_usd_rate(row)

    for c in CURRENCIES:
        local_rate = float(scenario_abs_rates[c])

        local_abs_col = RATE_INFO["local_abs_cols"].get(c)
        diff_col = RATE_INFO["diff_cols"].get(c)

        if local_abs_col and local_abs_col in row.index:
            row[local_abs_col] = local_rate

        if diff_col and diff_col in row.index:
            row[diff_col] = local_rate - usd_rate

    return row


def update_fx_change_features(base_row: pd.Series, scenario_row: pd.Series, scenario_fx: dict) -> pd.Series:
    row = scenario_row.copy()

    change_templates = [
        "{c}_spot_change",
        "{c}_fx_change",
        "{c}_return_1m",
        "{c}_mom_1m",
    ]

    for c, new_fx in scenario_fx.items():
        base_spot = None

        for feat in FX_FEATURE_CANDIDATES.get(c, []):
            if feat in base_row.index and pd.notna(base_row[feat]):
                base_spot = safe_float(base_row[feat], None)
                break

        if base_spot is None or base_spot == 0:
            continue

        pct_change = (new_fx / base_spot) - 1.0

        for template in change_templates:
            col = template.format(c=c)
            if col in row.index:
                row[col] = pct_change

    return row


def build_today_row():
    row = dataset.iloc[-1].copy()
    live_fx = fetch_live_fx_rates()
    row = update_fx_features_in_row(row, live_fx)
    return row, live_fx


def apply_scenario_to_row(base_row: pd.Series, scenario_fx: dict, scenario_abs_rates: dict) -> pd.Series:
    row = base_row.copy()
    row = update_fx_features_in_row(row, scenario_fx)
    row = update_rate_features_in_row(row, scenario_abs_rates)
    row = update_fx_change_features(base_row, row, scenario_fx)
    return row


# =========================================================
# 8. DISPLAY HELPERS
# =========================================================
def show_input_table(live_fx, row):
    abs_rates = get_absolute_rate_dict(row)

    input_df = pd.DataFrame({
        "Currency": CURRENCIES,
        "Current FX Rate": [live_fx.get(c, np.nan) for c in CURRENCIES],
        "Absolute Interest Rate (%)": [abs_rates[c] for c in CURRENCIES]
    })
    st.dataframe(input_df, use_container_width=True)

    usd_col = RATE_INFO["usd_rate_col"]
    if usd_col:
        st.caption(f"Base/USD rate column detected: {usd_col}")
    else:
        st.caption("No explicit USD/base rate column detected. Absolute rates may depend only on available local rate columns.")


def plot_weights(df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["Currency"], df["Weight"])
    ax.axhline(0, linewidth=1)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    st.pyplot(fig)


# =========================================================
# 9. TITLE
# =========================================================
st.title("FX Portfolio Optimization using PPO")
st.caption("PPO-based FX allocation using FX change + carry, with Sharpe-oriented training.")

today_row, live_fx = build_today_row()
today_label = datetime.now().strftime("%Y-%m-%d %H:%M")

# =========================================================
# 10. TABS
# =========================================================
tab1, tab2 = st.tabs(["Current Optimal Portfolio", "Scenario Simulation"])

# =========================================================
# 11. TAB 1 - CURRENT PORTFOLIO
# =========================================================
with tab1:
    st.subheader("Current PPO Portfolio")
    st.write(f"Live app refresh timestamp: **{today_label}**")

    st.markdown("### Current Market Inputs")
    show_input_table(live_fx, today_row)

    current_obs, weights, metrics, weights_df = run_model_on_row(today_row)

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected 1M Return", f"{metrics['total_return']:.2%}")
    c2.metric("Estimated Volatility", f"{metrics['volatility']:.2%}")
    c3.metric("Estimated Sharpe Ratio", f"{metrics['sharpe']:.2f}")

    st.markdown("### Portfolio Weights")
    st.dataframe(weights_df, use_container_width=True)

    c4, c5, c6 = st.columns(3)
    c4.metric("FX Contribution", f"{metrics['fx_contribution']:.2%}")
    c5.metric("Carry Contribution", f"{metrics['carry_contribution']:.2%}")
    c6.metric("Sum of Weights", f"{weights.sum():.6f}")

    plot_weights(weights_df, "Current PPO Portfolio Weights")

# =========================================================
# 12. TAB 2 - SCENARIO SIMULATION
# =========================================================
with tab2:
    st.subheader("Scenario Simulation")
    st.write("Adjust FX rates and absolute local interest rates, then run the scenario.")

    base_abs_rates = get_absolute_rate_dict(today_row)

    st.markdown("### Adjust Inputs")
    sim_cols = st.columns(2)

    scenario_fx = {}
    scenario_abs_rates = {}

    with sim_cols[0]:
        st.markdown("#### FX Rates")
        for c in CURRENCIES:
            default_fx = safe_float(live_fx.get(c, np.nan), 0.0)
            scenario_fx[c] = st.number_input(
                f"{c} FX Rate",
                value=default_fx,
                step=0.01,
                format="%.4f",
                key=f"fx_{c}"
            )

    with sim_cols[1]:
        st.markdown("#### Absolute Interest Rates (%)")
        for c in CURRENCIES:
            default_rate = safe_float(base_abs_rates.get(c, 0.0), 0.0)
            scenario_abs_rates[c] = st.number_input(
                f"{c} Interest Rate (%)",
                value=default_rate,
                step=0.10,
                format="%.2f",
                key=f"rate_{c}"
            )

    run_sim = st.button("Run Simulation", type="primary")

    if run_sim:
        scenario_row = apply_scenario_to_row(today_row, scenario_fx, scenario_abs_rates)

        sim_obs, sim_weights, sim_metrics, sim_weights_df = run_model_on_row(scenario_row)
        current_obs = get_state_from_row(today_row)

        st.markdown("### Simulated Portfolio Output")

        s1, s2, s3 = st.columns(3)
        s1.metric("Expected 1M Return", f"{sim_metrics['total_return']:.2%}")
        s2.metric("Estimated Volatility", f"{sim_metrics['volatility']:.2%}")
        s3.metric("Estimated Sharpe Ratio", f"{sim_metrics['sharpe']:.2f}")

        s4, s5, s6 = st.columns(3)
        s4.metric("FX Contribution", f"{sim_metrics['fx_contribution']:.2%}")
        s5.metric("Carry Contribution", f"{sim_metrics['carry_contribution']:.2%}")
        s6.metric("Sum of Weights", f"{sim_weights.sum():.6f}")

        st.markdown("### Simulated Portfolio Weights")
        st.dataframe(sim_weights_df, use_container_width=True)

        plot_weights(sim_weights_df, "Scenario Portfolio Weights")

        st.markdown("### Scenario Debug")
        debug_df = pd.DataFrame({
            "Metric": [
                "Observation shift vs current",
                "Max absolute observation difference"
            ],
            "Value": [
                float(np.abs(sim_obs - current_obs).sum()),
                float(np.max(np.abs(sim_obs - current_obs)))
            ]
        })
        st.dataframe(debug_df, use_container_width=True)

        st.caption(
            "If the observation shift is still tiny after changing rates, "
            "the trained PPO policy likely depends more on other features than the rate fields alone."
        )
    else:
        st.info("Adjust the inputs and click Run Simulation.")

# =========================================================
# 13. FOOTER
# =========================================================
st.markdown("---")
st.write("Model: PPO | Return definition: FX change + carry | Deployment mode: live input + inference")
