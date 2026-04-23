import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr
from stable_baselines3 import PPO

st.set_page_config(page_title="Nishe FX PPO Portfolio", layout="wide")

# =========================================================
# 1. CONFIG
# =========================================================
MODEL_PATH = "ppo_fx_final_model_sharpe"
TEMPLATE_DATA_PATH = "dataset_step2_features.csv"

ALL_CURRENCIES = ["EUR", "GBP", "AUD", "JPY", "INR", "CNY"]

FX_TICKERS = {
    "EUR": "EURUSD=X",
    "GBP": "GBPUSD=X",
    "AUD": "AUDUSD=X",
    "JPY": "JPY=X",
    "INR": "INR=X",
    "CNY": "CNY=X",
}

# FRED rate series:
# USD / EUR / GBP are solid
# others are best-effort short-term proxies available on FRED;
# if unavailable, the code falls back to the template dataset values.
FRED_RATE_SERIES = {
    "USD": "FEDFUNDS",
    "EUR": "ECBDFR",
    "GBP": "BOERUKM",
    "AUD": "IR3TIB01AUM156N",   # 3M interbank proxy (best-effort)
    "JPY": "IR3TIB01JPM156N",   # 3M interbank proxy (best-effort)
    "INR": "IR3TIB01INM156N",   # 3M interbank proxy (best-effort)
    "CNY": "IR3TIB01CNM156N",   # 3M interbank proxy
}

LOOKBACK_YEARS = 5
POSITION_LIMIT = 0.9

# =========================================================
# 2. LOAD MODEL + TEMPLATE DATA
# =========================================================
@st.cache_resource
def load_model():
    return PPO.load(MODEL_PATH)


@st.cache_data
def load_template_data():
    df = pd.read_csv(TEMPLATE_DATA_PATH, index_col=0, parse_dates=True)

    rename_map = {}
    for col in df.columns:
        if col.endswith("_ret_1m_target"):
            rename_map[col] = col.replace("_ret_1m_target", "_target")

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


model = load_model()
template_dataset = load_template_data()

CURRENCIES = [c for c in ALL_CURRENCIES if f"{c}_target" in template_dataset.columns]
STATE_FEATURES = [c for c in template_dataset.columns if not c.endswith("_target")]
STATE_FEATURES = [c for c in STATE_FEATURES if pd.api.types.is_numeric_dtype(template_dataset[c])]

# =========================================================
# 3. HELPERS
# =========================================================
def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def last_valid(series: pd.Series, default=np.nan):
    s = series.dropna()
    if len(s) == 0:
        return default
    return float(s.iloc[-1])


def second_last_valid(series: pd.Series, default=np.nan):
    s = series.dropna()
    if len(s) < 2:
        return default
    return float(s.iloc[-2])


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


def set_if_exists(row: pd.Series, candidates: list[str], value):
    for col in candidates:
        if col in row.index:
            row[col] = value


def compute_metrics(weights: np.ndarray, row: pd.Series):
    fx_vec = []
    carry_vec = []

    for c in CURRENCIES:
        fx_col = f"{c}_target"
        rate_diff_col = f"{c}_rate_diff"

        fx_val = safe_float(row[fx_col], 0.0) if fx_col in row.index else 0.0
        rate_diff = safe_float(row[rate_diff_col], 0.0) if rate_diff_col in row.index else 0.0
        carry_1m = (rate_diff / 100.0) / 12.0

        fx_vec.append(fx_val)
        carry_vec.append(carry_1m)

    fx_vec = np.array(fx_vec, dtype=float)
    carry_vec = np.array(carry_vec, dtype=float)

    fx_contribution = float(np.dot(weights, fx_vec))
    carry_contribution = float(np.dot(weights, carry_vec))
    total_return = fx_contribution + carry_contribution

    hist_returns = []
    tail_df = template_dataset.tail(12)

    for _, r in tail_df.iterrows():
        fx_tmp = np.array([safe_float(r.get(f"{c}_target", 0.0), 0.0) for c in CURRENCIES], dtype=float)
        carry_tmp = np.array([(safe_float(r.get(f"{c}_rate_diff", 0.0), 0.0) / 100.0) / 12.0 for c in CURRENCIES], dtype=float)
        hist_returns.append(float(np.dot(weights, fx_tmp + carry_tmp)))

    hist_returns = np.array(hist_returns, dtype=float)
    vol = float(np.std(hist_returns, ddof=0))

    if vol <= 1e-12:
        sharpe = 0.0
    else:
        sharpe = float((hist_returns.mean() / vol) * np.sqrt(12))

    return {
        "fx_contribution": fx_contribution,
        "carry_contribution": carry_contribution,
        "total_return": total_return,
        "volatility": vol,
        "sharpe": sharpe,
    }


def make_weights_table(weights: np.ndarray, row: pd.Series, current_abs_rates: dict, usd_rate: float) -> pd.DataFrame:
    records = []
    for i, c in enumerate(CURRENCIES):
        fx_val = safe_float(row.get(f"{c}_target", 0.0), 0.0)
        rate_diff = safe_float(row.get(f"{c}_rate_diff", 0.0), 0.0)
        carry_1m = (rate_diff / 100.0) / 12.0

        records.append({
            "Currency": c,
            "Weight": float(weights[i]),
            "Absolute Rate (%)": float(current_abs_rates.get(c, np.nan)),
            "USD Rate (%)": float(usd_rate),
            "Rate Differential (%)": float(rate_diff),
            "FX_1M": float(fx_val),
            "Carry_1M": float(carry_1m),
            "Total_Asset_1M": float(fx_val + carry_1m),
            "Long/Short": "Long" if weights[i] >= 0 else "Short"
        })

    df = pd.DataFrame(records).sort_values("Weight", ascending=False).reset_index(drop=True)
    return df


# =========================================================
# 4. DATA PULLS: YAHOO + FRED
# =========================================================
@st.cache_data(ttl=3600)
def fetch_fx_history_yahoo(start_date: str, end_date: str):
    fx_hist = {}

    for ccy, ticker in FX_TICKERS.items():
        if ccy not in CURRENCIES:
            continue

        try:
            hist = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )

            if hist.empty:
                fx_hist[ccy] = pd.Series(dtype=float)
                continue

            if isinstance(hist.columns, pd.MultiIndex):
                if ("Close", ticker) in hist.columns:
                    close_series = hist[("Close", ticker)]
                else:
                    close_series = hist.xs("Close", axis=1, level=0).iloc[:, 0]
            else:
                close_series = hist["Close"]

            fx_hist[ccy] = close_series.dropna().astype(float)

        except Exception:
            fx_hist[ccy] = pd.Series(dtype=float)

    return fx_hist


@st.cache_data(ttl=3600)
def fetch_fred_history(start_date: str, end_date: str):
    out = {}

    for name, series_id in FRED_RATE_SERIES.items():
        try:
            s = pdr.DataReader(series_id, "fred", start_date, end_date)
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            out[name] = s.dropna().astype(float)
        except Exception:
            out[name] = pd.Series(dtype=float)

    return out


def daily_to_monthly_last(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=float)
    s = series.copy().sort_index()
    return s.resample("ME").last().dropna()


def align_monthly_histories(fx_hist_daily: dict, rate_hist: dict):
    fx_monthly = {}
    for c, s in fx_hist_daily.items():
        fx_monthly[c] = daily_to_monthly_last(s)

    rate_monthly = {}
    for k, s in rate_hist.items():
        if s.empty:
            rate_monthly[k] = pd.Series(dtype=float)
        else:
            s2 = s.copy().sort_index()
            rate_monthly[k] = s2.resample("ME").last().ffill().dropna()

    return fx_monthly, rate_monthly


# =========================================================
# 5. BUILD CURRENT FEATURE ROW FROM 5Y HISTORY
# =========================================================
def get_template_last_row():
    return template_dataset.iloc[-1].copy()


def derive_current_absolute_rates(rate_monthly: dict, template_row: pd.Series) -> tuple[dict, float]:
    usd_rate = last_valid(rate_monthly.get("USD", pd.Series(dtype=float)), np.nan)

    if pd.isna(usd_rate):
        usd_rate = 0.0

    current_abs_rates = {}
    for c in CURRENCIES:
        live_rate = last_valid(rate_monthly.get(c, pd.Series(dtype=float)), np.nan)

        if pd.isna(live_rate):
            # fallback 1: if template has explicit local rate
            explicit_rate_col = f"{c}_rate"
            if explicit_rate_col in template_row.index and pd.notna(template_row[explicit_rate_col]):
                live_rate = safe_float(template_row[explicit_rate_col], 0.0)
            else:
                # fallback 2: reconstruct from usd + diff
                diff_col = f"{c}_rate_diff"
                diff_val = safe_float(template_row.get(diff_col, 0.0), 0.0)
                live_rate = usd_rate + diff_val

        current_abs_rates[c] = float(live_rate)

    return current_abs_rates, float(usd_rate)


def build_live_feature_row(template_row: pd.Series, fx_monthly: dict, rate_monthly: dict):
    row = template_row.copy()

    current_abs_rates, usd_rate = derive_current_absolute_rates(rate_monthly, template_row)

    for c in CURRENCIES:
        fx_series = fx_monthly.get(c, pd.Series(dtype=float))
        if fx_series.empty:
            continue

        spot_t = last_valid(fx_series, np.nan)
        spot_t1 = second_last_valid(fx_series, np.nan)

        if pd.notna(spot_t):
            set_if_exists(row, [f"{c}_spot", f"{c}_fx", f"{c}_close", f"{c}_price"], spot_t)

        if pd.notna(spot_t) and pd.notna(spot_t1) and spot_t1 != 0:
            ret_1m = (spot_t / spot_t1) - 1.0
            # current realized-style feature proxy used for latest target-style estimate
            if f"{c}_target" in row.index:
                row[f"{c}_target"] = ret_1m

            set_if_exists(
                row,
                [f"{c}_ret_1m", f"{c}_return_1m", f"{c}_fx_change", f"{c}_spot_change", f"{c}_mom_1m"],
                ret_1m
            )

        # 3M momentum
        if len(fx_series.dropna()) >= 4:
            spot_t3 = float(fx_series.dropna().iloc[-4])
            if spot_t3 != 0:
                mom_3m = (spot_t / spot_t3) - 1.0
                set_if_exists(row, [f"{c}_mom_3m", f"{c}_return_3m", f"{c}_ret_3m"], mom_3m)

        # 12M rolling vol on monthly returns
        monthly_rets = fx_series.pct_change().dropna()
        if len(monthly_rets) >= 12:
            vol_12m = float(monthly_rets.iloc[-12:].std(ddof=0))
            set_if_exists(row, [f"{c}_vol_12m", f"{c}_rolling_vol", f"{c}_fx_vol"], vol_12m)

        # absolute rates + differential
        local_rate = current_abs_rates[c]
        rate_diff = local_rate - usd_rate

        set_if_exists(row, [f"{c}_rate", f"{c}_abs_rate", f"{c}_interest_rate"], local_rate)
        if f"{c}_rate_diff" in row.index:
            row[f"{c}_rate_diff"] = rate_diff

        # delta rate diff
        local_series = rate_monthly.get(c, pd.Series(dtype=float))
        usd_series = rate_monthly.get("USD", pd.Series(dtype=float))

        if not local_series.empty and not usd_series.empty:
            aligned = pd.concat([local_series.rename("local"), usd_series.rename("usd")], axis=1).ffill().dropna()
            if len(aligned) >= 2:
                aligned["diff"] = aligned["local"] - aligned["usd"]
                delta_diff = float(aligned["diff"].iloc[-1] - aligned["diff"].iloc[-2])
                set_if_exists(row, [f"{c}_delta_rate_diff", f"delta_{c}_rate_diff"], delta_diff)

    return row, current_abs_rates, usd_rate


def build_scenario_feature_row(
    template_row: pd.Series,
    fx_monthly: dict,
    rate_monthly: dict,
    scenario_fx: dict,
    scenario_abs_rates: dict
):
    row = template_row.copy()

    # copy histories so scenario only changes the latest point
    fx_monthly_sim = {k: v.copy() for k, v in fx_monthly.items()}
    rate_monthly_sim = {k: v.copy() for k, v in rate_monthly.items()}

    # determine latest month-end present in histories
    all_dates = []
    for s in fx_monthly_sim.values():
        if len(s) > 0:
            all_dates.append(s.index.max())
    for s in rate_monthly_sim.values():
        if len(s) > 0:
            all_dates.append(s.index.max())

    if len(all_dates) == 0:
        latest_date = pd.Timestamp.today().normalize() + pd.offsets.MonthEnd(0)
    else:
        latest_date = max(all_dates)

    # update latest FX and rate observations in the 5Y histories
    for c in CURRENCIES:
        if c not in fx_monthly_sim or fx_monthly_sim[c].empty:
            fx_monthly_sim[c] = pd.Series(dtype=float)

        fx_monthly_sim[c].loc[latest_date] = float(scenario_fx[c])

        if c not in rate_monthly_sim or rate_monthly_sim[c].empty:
            rate_monthly_sim[c] = pd.Series(dtype=float)

        rate_monthly_sim[c].loc[latest_date] = float(scenario_abs_rates[c])

    # USD remains whatever latest FRED value is
    if "USD" not in rate_monthly_sim or rate_monthly_sim["USD"].empty:
        rate_monthly_sim["USD"] = pd.Series([0.0], index=[latest_date])

    current_abs_rates, usd_rate = derive_current_absolute_rates(rate_monthly_sim, template_row)

    # rebuild latest row features from the modified 5Y histories
    for c in CURRENCIES:
        fx_series = fx_monthly_sim.get(c, pd.Series(dtype=float)).sort_index()
        if len(fx_series.dropna()) == 0:
            continue

        spot_t = last_valid(fx_series, np.nan)
        spot_t1 = second_last_valid(fx_series, np.nan)

        if pd.notna(spot_t):
            set_if_exists(row, [f"{c}_spot", f"{c}_fx", f"{c}_close", f"{c}_price"], spot_t)

        if pd.notna(spot_t) and pd.notna(spot_t1) and spot_t1 != 0:
            ret_1m = (spot_t / spot_t1) - 1.0
            if f"{c}_target" in row.index:
                row[f"{c}_target"] = ret_1m

            set_if_exists(
                row,
                [f"{c}_ret_1m", f"{c}_return_1m", f"{c}_fx_change", f"{c}_spot_change", f"{c}_mom_1m"],
                ret_1m
            )

        if len(fx_series.dropna()) >= 4:
            spot_t3 = float(fx_series.dropna().iloc[-4])
            if spot_t3 != 0:
                mom_3m = (spot_t / spot_t3) - 1.0
                set_if_exists(row, [f"{c}_mom_3m", f"{c}_return_3m", f"{c}_ret_3m"], mom_3m)

        monthly_rets = fx_series.pct_change().dropna()
        if len(monthly_rets) >= 12:
            vol_12m = float(monthly_rets.iloc[-12:].std(ddof=0))
            set_if_exists(row, [f"{c}_vol_12m", f"{c}_rolling_vol", f"{c}_fx_vol"], vol_12m)

        local_rate = current_abs_rates[c]
        rate_diff = local_rate - usd_rate

        set_if_exists(row, [f"{c}_rate", f"{c}_abs_rate", f"{c}_interest_rate"], local_rate)
        if f"{c}_rate_diff" in row.index:
            row[f"{c}_rate_diff"] = rate_diff

        local_series = rate_monthly_sim.get(c, pd.Series(dtype=float)).sort_index()
        usd_series = rate_monthly_sim.get("USD", pd.Series(dtype=float)).sort_index()

        aligned = pd.concat([local_series.rename("local"), usd_series.rename("usd")], axis=1).ffill().dropna()
        if len(aligned) >= 2:
            aligned["diff"] = aligned["local"] - aligned["usd"]
            delta_diff = float(aligned["diff"].iloc[-1] - aligned["diff"].iloc[-2])
            set_if_exists(row, [f"{c}_delta_rate_diff", f"delta_{c}_rate_diff"], delta_diff)

    return row, current_abs_rates, usd_rate


def run_model_on_row(row: pd.Series):
    obs = get_state_from_row(row)
    action, _ = model.predict(obs, deterministic=True)
    weights = action_to_weights(action, position_limit=POSITION_LIMIT)
    metrics = compute_metrics(weights, row)
    return obs, weights, metrics


# =========================================================
# 6. LOAD LIVE 5Y INPUT DATA
# =========================================================
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=365 * LOOKBACK_YEARS + 30)

fx_hist_daily = fetch_fx_history_yahoo(start_date.isoformat(), end_date.isoformat())
fred_hist = fetch_fred_history(start_date.isoformat(), end_date.isoformat())
fx_monthly, rate_monthly = align_monthly_histories(fx_hist_daily, fred_hist)

template_row = get_template_last_row()
live_row, current_abs_rates, current_usd_rate = build_live_feature_row(template_row, fx_monthly, rate_monthly)
current_obs, current_weights, current_metrics = run_model_on_row(live_row)
current_table = make_weights_table(current_weights, live_row, current_abs_rates, current_usd_rate)

# live FX display = latest daily value
live_fx_display = {c: last_valid(fx_hist_daily.get(c, pd.Series(dtype=float)), np.nan) for c in CURRENCIES}

# =========================================================
# 7. UI
# =========================================================
st.title("FX Portfolio Optimization using PPO")
st.caption("5Y Yahoo Finance FX history + FRED rates → latest feature row → PPO weights")

tab1, tab2 = st.tabs(["Current Optimal Portfolio", "Scenario Simulation"])

with tab1:
    st.subheader("Current PPO Portfolio")
    st.write(f"App refresh time: **{datetime.now().strftime('%Y-%m-%d %H:%M')}**")

    st.markdown("### Current Market Inputs")
    current_input_df = pd.DataFrame({
        "Currency": CURRENCIES,
        "Current FX Rate": [live_fx_display.get(c, np.nan) for c in CURRENCIES],
        "Absolute Interest Rate (%)": [current_abs_rates.get(c, np.nan) for c in CURRENCIES],
        "Rate Differential vs USD (%)": [current_abs_rates.get(c, 0.0) - current_usd_rate for c in CURRENCIES],
    })
    st.dataframe(current_input_df, use_container_width=True)
    st.write(f"**USD Rate (%):** {current_usd_rate:.2f}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected 1M Return", f"{current_metrics['total_return']:.2%}")
    c2.metric("Estimated Volatility", f"{current_metrics['volatility']:.2%}")
    c3.metric("Estimated Sharpe Ratio", f"{current_metrics['sharpe']:.2f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("FX Contribution", f"{current_metrics['fx_contribution']:.2%}")
    c5.metric("Carry Contribution", f"{current_metrics['carry_contribution']:.2%}")
    c6.metric("Sum of Weights", f"{current_weights.sum():.6f}")

    st.markdown("### Current Portfolio Weights")
    st.dataframe(current_table, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(current_table["Currency"], current_table["Weight"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Current PPO Portfolio Weights")
    ax.set_ylabel("Weight")
    st.pyplot(fig)

with tab2:
    st.subheader("Scenario Simulation")
    st.write(
        "This simulation still uses the same 5-year Yahoo/FRED history, "
        "but replaces the latest FX / interest-rate point with your scenario, "
        "recomputes rate differentials, rebuilds the latest feature row, and then reruns PPO."
    )

    st.markdown("### Scenario Inputs")
    left, right = st.columns(2)

    scenario_fx = {}
    scenario_abs_rates = {}

    with left:
        st.markdown("#### FX Rates")
        for c in CURRENCIES:
            scenario_fx[c] = st.number_input(
                f"{c} FX Rate",
                value=float(live_fx_display.get(c, 0.0)) if pd.notna(live_fx_display.get(c, np.nan)) else 0.0,
                step=0.01,
                format="%.4f",
                key=f"fx_{c}"
            )

    with right:
        st.markdown("#### Absolute Interest Rates (%)")
        for c in CURRENCIES:
            scenario_abs_rates[c] = st.number_input(
                f"{c} Interest Rate (%)",
                value=float(current_abs_rates.get(c, 0.0)),
                step=0.10,
                format="%.2f",
                key=f"rate_{c}"
            )

    run_sim = st.button("Run Simulation", type="primary")

    if run_sim:
        sim_row, sim_abs_rates, sim_usd_rate = build_scenario_feature_row(
            template_row=template_row,
            fx_monthly=fx_monthly,
            rate_monthly=rate_monthly,
            scenario_fx=scenario_fx,
            scenario_abs_rates=scenario_abs_rates,
        )

        sim_obs, sim_weights, sim_metrics = run_model_on_row(sim_row)
        sim_table = make_weights_table(sim_weights, sim_row, sim_abs_rates, sim_usd_rate)

        st.markdown("### Simulated Portfolio Output")
        s1, s2, s3 = st.columns(3)
        s1.metric("Expected 1M Return", f"{sim_metrics['total_return']:.2%}")
        s2.metric("Estimated Volatility", f"{sim_metrics['volatility']:.2%}")
        s3.metric("Estimated Sharpe Ratio", f"{sim_metrics['sharpe']:.2f}")

        s4, s5, s6 = st.columns(3)
        s4.metric("FX Contribution", f"{sim_metrics['fx_contribution']:.2%}")
        s5.metric("Carry Contribution", f"{sim_metrics['carry_contribution']:.2%}")
        s6.metric("Sum of Weights", f"{sim_weights.sum():.6f}")

        st.markdown("### Simulated Market Inputs")
        sim_input_df = pd.DataFrame({
            "Currency": CURRENCIES,
            "Scenario FX Rate": [scenario_fx[c] for c in CURRENCIES],
            "Scenario Absolute Rate (%)": [sim_abs_rates[c] for c in CURRENCIES],
            "Scenario Rate Differential vs USD (%)": [sim_abs_rates[c] - sim_usd_rate for c in CURRENCIES],
        })
        st.dataframe(sim_input_df, use_container_width=True)
        st.write(f"**USD Rate (%):** {sim_usd_rate:.2f}")

        st.markdown("### Scenario Portfolio Weights")
        st.dataframe(sim_table, use_container_width=True)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(sim_table["Currency"], sim_table["Weight"])
        ax2.axhline(0, linewidth=1)
        ax2.set_title("Scenario PPO Portfolio Weights")
        ax2.set_ylabel("Weight")
        st.pyplot(fig2)

        debug_df = pd.DataFrame({
            "Metric": [
                "Observation shift vs current",
                "Max abs observation difference"
            ],
            "Value": [
                float(np.abs(sim_obs - current_obs).sum()),
                float(np.max(np.abs(sim_obs - current_obs)))
            ]
        })
        st.markdown("### Debug")
        st.dataframe(debug_df, use_container_width=True)
    else:
        st.info("Adjust FX / rates and click Run Simulation.")

st.markdown("---")
st.write("Model: PPO | Current tab = live 5Y rebuild | Simulation = latest point override on same 5Y histories")
