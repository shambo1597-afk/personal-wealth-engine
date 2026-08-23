# ====================================================================================================
# MASTER QUANTITATIVE WEALTH OPERATING SYSTEM (STREAMLIT PRODUCTION SUITE v12.0)
# Modular Architecture: config | database | quant_engine | tax_engine | app (Orchestration)
# ====================================================================================================

import os
import io
import time
import json
import datetime
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from config import (
    DB_FILE, RISK_FREE_RATE, MAX_RETAIL_CAP, MAX_SECTOR_CAP, STATUTORY_FEE_BUFFER, MIN_DISPOSAL_VALUE,
    MIN_ADTV_INR, TOP_N_SELECTED_EQUITIES, BENCHMARK_TICKER,
    EQUITY_DELIVERY_FRICTION, ETF_DEBT_GOLD_FRICTION,
    SOVEREIGN_BOND_TICKER, get_market_time_horizons
)
from database import (
    get_db_connection, init_db, clean_tax_lots_df, clean_trade_ledger_df,
    reconcile_corporate_actions_and_splits, parse_and_import_broker_csv, compile_xirr_cash_flows,
    create_database_backup, backup_database
)
from quant_engine import (
    compute_portfolio_xirr, fetch_master_market_data, solve_portfolio_in_memory,
    ingest_and_run_dual_pipeline, fetch_latest_prices, fetch_recent_news,
    fetch_fundamental_scorecard, fetch_live_indian_risk_free_rate, compute_vectorized_monte_carlo_frontier,
    analyze_ticker_sentiment, compute_news_sentiment_for_universe,
    compute_lifecycle_asset_allocation,
    sync_screener_fundamentals_for_universe, get_fundamentals_last_synced,
    SECTOR_MAP, get_asset_sector
)
from tax_engine import (
    compute_realized_tax_summary, compute_unrealized_tax_lots_analysis, build_schedule_112a_records,
    recommend_tax_loss_harvesting_trades
)

warnings.filterwarnings('ignore')

def get_asset_friction(ticker, asset_class=''):
    """Resolves asset-specific statutory exchange friction (0.03% for debt/gold vs 0.15% for equities)."""
    t_upper = str(ticker).upper()
    cls_upper = str(asset_class).upper()
    is_debt_or_gold = (
        t_upper == SOVEREIGN_BOND_TICKER.upper() or
        'GOLDBEES' in t_upper or
        'SILVERBEES' in t_upper or
        'DEBT' in cls_upper or
        'GOLD' in cls_upper or
        '50AA' in cls_upper
    )
    return ETF_DEBT_GOLD_FRICTION if is_debt_or_gold else EQUITY_DELIVERY_FRICTION

# Dynamically resolve live market date anchors and time horizons
dates = get_market_time_horizons()
TODAY = dates['TODAY']
today_naive = TODAY.tz_localize(None) if getattr(TODAY, 'tzinfo', None) is not None else TODAY
TODAY_STR = dates['TODAY_STR']
TODAY_ISO = dates['TODAY_ISO']
OOS_SPLIT_STR = dates['OOS_SPLIT_STR']
BACKTEST_TRAIN_START_STR = dates['BACKTEST_TRAIN_START_STR']
LIVE_TRAIN_START_STR = dates['LIVE_TRAIN_START_STR']

# ----------------------------------------------------------------------------------------------------
# 1. STREAMLIT PAGE CONFIGURATION & DATABASE INITIALIZATION
# ----------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Personal Wealth Engine | Quantitative OS",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_db()

if 'committed_today' not in st.session_state:
    st.session_state.committed_today = False
if 'last_commit_info' not in st.session_state:
    st.session_state.last_commit_info = None

# ----------------------------------------------------------------------------------------------------
# 2. SIDEBAR CONTROLS & BATCHED FORM INPUTS
# ----------------------------------------------------------------------------------------------------
st.sidebar.title("⚙️ Engine Controls")
st.sidebar.markdown("Configure personal wealth targets, investor profile, broker liquidity, and optimization models.")

with st.sidebar.form("engine_settings_form"):
    dob = st.date_input(
        "Date of Birth (Risk Profiler)",
        value=datetime.date(1997, 10, 15),
        min_value=datetime.date(1940, 1, 1),
        max_value=datetime.date(2010, 1, 1),
        help="Determines equity glide path: Target Equity Wt = (110 - Age)%."
    )

    total_available_broker_cash = st.number_input(
        "Available Broker Wallet Cash (₹)",
        min_value=0.0,
        max_value=10000000.0,
        value=150000.0,
        step=10000.0,
        format="%.2f",
        help="Uninvested cash in trading account available for fresh purchases."
    )

    turbo_mode = st.toggle(
        "⚡ Turbo / Presentation Mode (Instant Offline)", 
        value=True, 
        help="Bypasses all live network checks and executes purely in-memory from local Parquet data (<0.05s)."
    )

    recalculate_btn = st.form_submit_button("⚡ Run & Recalculate Portfolio", type="primary")

live_rf_val, is_live, source_name = fetch_live_indian_risk_free_rate(turbo_mode=turbo_mode)
RISK_FREE_RATE = live_rf_val if live_rf_val <= 1.0 else live_rf_val / 100.0
rf_disp = live_rf_val * 100.0 if live_rf_val <= 1.0 else live_rf_val
retirement_age = 60.0

if is_live:
    st.sidebar.caption(f"🟢 **Live Sovereign Rate:** `{rf_disp:.2f}%` *(Source: {source_name})*")
else:
    st.sidebar.caption(f"🟡 **Fallback Rate:** `{rf_disp:.2f}%` *(Network Offline)*")

# Real-time Retirement Clock & Horizon Resolver
exact_age_sb = (TODAY - pd.to_datetime(dob)).days / 365.25
years_to_goal_sb = max(0.0, float(retirement_age - exact_age_sb))
if years_to_goal_sb <= 3.0:
    st.sidebar.caption(f"🛡️ **LDI Shield Active:** `{years_to_goal_sb:.1f}Y` to Goal (Age `{exact_age_sb:.1f}` → `{retirement_age:.0f}`)")
else:
    st.sidebar.caption(f"⏳ **Retirement Clock:** `{years_to_goal_sb:.1f}Y` to Goal (Age `{exact_age_sb:.1f}` → `{retirement_age:.0f}`)")

# Hardcoded quant optimization mode locked to Ledoit-Wolf Max-Sharpe tangency
opt_mode = 'Max-Sharpe (Ledoit-Wolf)'

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Tradebook Importer")

with st.sidebar.expander("Import Zerodha / Groww CSV", expanded=False):
    import_mode = st.radio(
        "Import Mode:",
        options=['Merge New Trades (Incremental)', 'Full Ledger Reset & Rebuild from CSV'],
        index=0,
        help="Incremental mode appends new trades. Reset & Rebuild completely reconstructs portfolio from tradebook CSV."
    )

    confirm_reset = True
    if import_mode == 'Full Ledger Reset & Rebuild from CSV':
        confirm_reset = st.checkbox("⚠️ Confirm complete overwrite of existing lots", value=False)

    uploaded_tradebook = st.file_uploader(
        "Upload Tradebook CSV",
        type=['csv'],
        key="demat_csv_uploader",
        help="Upload standard Zerodha Console tradebook or Groww order report CSV."
    )
    if uploaded_tradebook is not None:
        if import_mode == 'Full Ledger Reset & Rebuild from CSV' and not confirm_reset:
            st.warning("⚠️ Please check confirmation box above to proceed.")
        else:
            file_bytes = uploaded_tradebook.getvalue()
            import_sig = f"{uploaded_tradebook.name}_{len(file_bytes)}_{import_mode}"
            if st.session_state.get('last_imported_sig') != import_sig:
                with get_db_connection() as conn:
                    import_res = parse_and_import_broker_csv(file_bytes, conn, mode=import_mode)
                st.session_state.last_imported_sig = import_sig
                st.session_state.last_import_res = import_res
                if import_res.get('success'):
                    st.cache_data.clear()
                    st.rerun()

            import_res = st.session_state.get('last_import_res')
            if import_res and import_res.get('success'):
                st.success(f"✓ Detected: **{import_res['broker']}**")
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric("Lots Imported", import_res['imported_count'])
                    st.metric("Duplicates Skipped", import_res['skipped_count'])
                with m_col2:
                    st.metric("Invested Capital", f"₹{import_res['total_invested']:,.2f}")
                    st.metric("Net Active Lots", import_res['active_lots_count'])
                if import_res.get('unmatched_sells_count', 0) > 0 and 'Incremental' in import_mode:
                    st.warning("⚠️ Note: Sells detected. For full historical FIFO matching, select 'Full Ledger Reset & Rebuild'.")
            elif import_res and not import_res.get('success'):
                st.error(f"❌ {import_res['error']}")

st.sidebar.markdown("---")
st.sidebar.subheader("ACID Database Execution")
sb_force_commit = st.sidebar.checkbox(
    "⚡ Allow Same-Day Additional Tranche",
    value=False,
    key="sb_force_commit",
    help="Check to commit an additional trade tranche or rebalance on the same day."
)
sb_confirm_commit = st.sidebar.checkbox(
    "🔒 Confirm accuracy of broker orders before updating SQLite ledger",
    value=False,
    key="sb_confirm_commit"
)
sb_execute_button = st.sidebar.button(
    "💾 Commit Trades to SQLite Database",
    type="primary",
    disabled=not sb_confirm_commit,
    key="sb_execute_btn"
)

if st.sidebar.button("🔄 Clear Market Cache & Recalculate"):
    st.cache_data.clear()
    st.session_state.committed_today = False
    st.session_state.last_commit_info = None
    st.sidebar.success("Market cache cleared!")

# ----------------------------------------------------------------------------------------------------
# 3. MAIN DASHBOARD EXECUTION & TWO-TIER QUANT PIPELINE
# ----------------------------------------------------------------------------------------------------
st.title("🏛️ Master Quantitative Wealth Operating System")

progress_placeholder = st.empty()
with progress_placeholder.container():
    p_bar = st.progress(5, text="⚡ Initializing Quantitative Wealth Engine...")

# Stage 1: Master Market Ingestion
p_bar.progress(20, text="📥 Ingesting Nifty 200 Universe & Sovereign Risk-Free Benchmarks...")
master_data = fetch_master_market_data(turbo_mode=turbo_mode)

# Data-vintage disclosure: the price/volume history actually driving today's recommendations
# may not be today's data, especially with Turbo Mode's Parquet cache -- surface it plainly
# rather than let a stale run look identical to a fresh one.
_valid_assets_for_vintage = master_data.get('valid_assets')
if _valid_assets_for_vintage is not None and not _valid_assets_for_vintage.empty:
    _data_asof_ts = pd.to_datetime(_valid_assets_for_vintage.index.max())
    _data_age_days = (TODAY.normalize() - _data_asof_ts.normalize()).days
    if _data_age_days <= 1:
        st.sidebar.caption(f"🟢 **Price Data As Of:** `{_data_asof_ts.strftime('%d-%b-%Y')}` (current)")
    elif _data_age_days <= 5:
        st.sidebar.caption(f"🟡 **Price Data As Of:** `{_data_asof_ts.strftime('%d-%b-%Y')}` ({_data_age_days}d old)")
    else:
        st.sidebar.caption(f"🔴 **Price Data As Of:** `{_data_asof_ts.strftime('%d-%b-%Y')}` ({_data_age_days}d old — stale, refresh before trading)")
if turbo_mode:
    st.sidebar.caption("⚡ **Turbo Mode ON:** reading from local Parquet cache, not a fresh live pull. Toggle it off and rerun before you rely on these numbers for a real order.")

# Stage 2: Multi-Factor Scoring & Quality Gates
p_bar.progress(45, text="📊 Scoring 3-Stage DuPont Quality, 12M Momentum & Piotroski Safety Gates...")
live_sector_map = master_data.get('sector_map', {})

# Stage 3: Markowitz Ledoit-Wolf Tangency Optimization
p_bar.progress(65, text="🧮 Solving Ledoit-Wolf Markowitz Tangency & Bayesian Sector Simplex...")
# Solve Live & Backtest allocations purely in-memory in < 20ms
live_quant = solve_portfolio_in_memory(
    master_data['live_returns_df'],
    mode=opt_mode,
    risk_free_rate=RISK_FREE_RATE,
    max_retail_cap=MAX_RETAIL_CAP,
    shrunk_cov=master_data.get('live_shrunk_cov'),
    mean_returns=master_data.get('live_mean_returns'),
    sector_map=SECTOR_MAP
)
live_quant['bench_returns'] = master_data['bench_live_rets']
live_quant['multifactor_scorecard'] = master_data['live_scorecard']
live_quant['top_20_equities'] = master_data['top_20_equities']
live_quant['universe_fundamentals'] = master_data['universe_fundamentals']
live_quant['sma_200_series'] = master_data['sma_200_series']
live_quant['adtv_90d_series'] = master_data['adtv_90d_series']
live_quant['valid_assets'] = master_data['valid_assets']

bt_quant = solve_portfolio_in_memory(
    master_data['backtest_train_df'],
    mode=opt_mode,
    risk_free_rate=RISK_FREE_RATE,
    max_retail_cap=MAX_RETAIL_CAP,
    shrunk_cov=master_data.get('bt_shrunk_cov'),
    mean_returns=master_data.get('bt_mean_returns'),
    sector_map=SECTOR_MAP
)
bt_quant['multifactor_scorecard'] = master_data['bt_scorecard']
bt_quant['top_20_equities'] = master_data['top_20_bt_equities']
bt_quant['sma_200_series'] = master_data['valid_assets'].rolling(200).mean().iloc[-1]

tickers = live_quant['tickers']
clean_tickers = live_quant['clean_tickers']
w_optimal_live = live_quant['w_optimal']
active_returns_df = live_quant['active_returns_df']
active_cov_matrix = live_quant['active_cov_matrix']
active_mean_vector = live_quant['active_mean_vector']
bench_returns = live_quant['bench_returns']
OPTIMAL_K = live_quant['optimal_k']
multifactor_scorecard_df = master_data['live_scorecard']
top_20_equities = master_data['top_20_equities']
sma_200_series = master_data['sma_200_series']
oos_assets = master_data['oos_assets']
oos_bench = master_data['oos_bench']
universe_fundamentals = master_data.get('universe_fundamentals')
news_triggers = master_data.get('triggers', {})
news_data = master_data.get('news_data', {})

# Dynamic Regime-Switching Volatility Targeting
if bench_returns is not None and not bench_returns.empty:
    rolling_60d_vol = float(bench_returns.tail(60).std() * np.sqrt(252))
else:
    rolling_60d_vol = 0.15

dob_timestamp = pd.to_datetime(dob).tz_localize(None) if getattr(pd.to_datetime(dob), 'tzinfo', None) is not None else pd.to_datetime(dob)
exact_age = (today_naive - dob_timestamp).days / 365.25
years_to_goal = max(0.0, float(retirement_age - exact_age))

lifecycle_alloc = compute_lifecycle_asset_allocation(
    exact_age=exact_age,
    years_to_goal=years_to_goal,
    rolling_60d_vol=rolling_60d_vol
)
w_risky = lifecycle_alloc['w_risky']
w_safe  = lifecycle_alloc['w_safe']
is_ldi_active = lifecycle_alloc['is_ldi_active']
regime_tag = lifecycle_alloc['regime_tag']
regime_desc = lifecycle_alloc['regime_desc']

st.caption(f"Execution Date: **{TODAY.strftime('%d-%B-%Y')}** | Engine: **{opt_mode}** | Active Market Regime: **{regime_tag}** (60d Realized Vol: **{rolling_60d_vol*100:.1f}%**)")

# Stage 4: Valuations, 200-DMA Trend Filter & FIFO Tax Lots
p_bar.progress(85, text="🛡️ Applying 200-DMA Trend Filter, LDI Cash Shield & Tax FIFO Engine...")

target_w_series = pd.Series(w_optimal_live, index=tickers)

with get_db_connection() as conn:
    tax_lots_df = clean_tax_lots_df(pd.read_sql("""
        SELECT lot_id, ticker, asset_name, asset_class, buy_date, quantity, buy_price 
        FROM tax_lots WHERE quantity > 0 ORDER BY buy_date ASC
    """, conn))

owned_summary = {}
for t in tax_lots_df['ticker'].unique():
    sub_lots = tax_lots_df[tax_lots_df['ticker'] == t]
    total_q = float(sub_lots['quantity'].sum())
    ltcg_q = 0.0
    for _, lot in sub_lots.iterrows():
        lot_buy_dt = pd.to_datetime(lot['buy_date']).tz_localize(None) if getattr(pd.to_datetime(lot['buy_date']), 'tzinfo', None) is not None else pd.to_datetime(lot['buy_date'])
        if (today_naive - lot_buy_dt).days >= 365:
            ltcg_q += float(lot['quantity'])
    
    a_class = str(sub_lots['asset_class'].iloc[0])
    if t == SOVEREIGN_BOND_TICKER:
        a_class = 'Sovereign Debt ETF (Sec 50AA)'

    owned_summary[t] = {
        'total_shares': total_q, 'ltcg_shares': ltcg_q, 'stcg_shares': total_q - ltcg_q,
        'asset_name': str(sub_lots['asset_name'].iloc[0]), 'asset_class': a_class
    }

owned_tickers = list(owned_summary.keys())
all_relevant_tickers = list(set(tickers + owned_tickers + [SOVEREIGN_BOND_TICKER]))

latest_prices_series = fetch_latest_prices(all_relevant_tickers, turbo_mode=turbo_mode)
news_data = master_data.get('news_data') or fetch_recent_news(all_relevant_tickers, turbo_mode=turbo_mode)
sent_series, gov_flags, triggers = compute_news_sentiment_for_universe(all_relevant_tickers, news_data, turbo_mode=turbo_mode)

# Stage 5: Finalization & Demat Ticket Generation
p_bar.progress(100, text="✅ Portfolio Optimization & Demat Trade Tickets Synchronized!")
time.sleep(0.15)
progress_placeholder.empty()

if recalculate_btn:
    st.toast("⚡ Portfolio successfully re-calculated and optimized!", icon="🚀")

current_risky_wealth, current_safe_wealth = 0.0, 0.0
for t in all_relevant_tickers:
    if t in owned_summary:
        try:
            p = float(latest_prices_series.get(t, 0.0))
            if pd.isna(p) or p < 0: p = 0.0
        except Exception: p = 0.0
        val = owned_summary[t]['total_shares'] * p
        if t == SOVEREIGN_BOND_TICKER: current_safe_wealth += val
        else: current_risky_wealth += val

total_portfolio_wealth = current_risky_wealth + current_safe_wealth + total_available_broker_cash
target_risky_wealth = total_portfolio_wealth * w_risky
target_safe_wealth  = total_portfolio_wealth * w_safe

# ----------------------------------------------------------------------------------------------------
# 4. HARDENED 4-PASS QUANT REBALANCING WITH 200-DMA TREND FILTER & DP FEE SHIELD
# ----------------------------------------------------------------------------------------------------
drift_tolerance_pct = 25 if total_available_broker_cash > 0 else 20
tolerance_decimal = drift_tolerance_pct / 100.0

# Build universe Piotroski F-Score & Institutional Inflow lookup maps for Tab 1 & Tab 3.
# 'safe' is the actual buy-gate decision: True only for a real F-Score > 3, or a Financial
# Institution the score is explicitly exempted for -- missing/failed/unsynced data is UNSAFE
# (blocks fresh capital) by default, never a silent neutral pass.
_UNSYNCED_F_INFO = {'score': None, 'badge': '⚪ Not Yet Synced', 'display': 'N/A', 'safe': False, 'source': 'Not Yet Synced'}

def _f_info_from_row(row) -> dict:
    raw_score = row.get('piotroski_f_score', row.get('Piotroski_F_Score', row.get('f_score_num')))
    score = None if pd.isna(raw_score) else int(raw_score)
    source = str(row.get('Data Source', row.get('Fundamentals Source', 'Not Yet Synced')))
    is_exempt_financial = (source == 'Not Applicable (Financial Institution)')
    return {
        'score': score,
        'badge': str(row.get('piotroski_badge', row.get('f_score_label', row.get('F_Score_Label', '⚪ Not Yet Synced')))),
        'display': str(row.get('Piotroski F-Score', row.get('f_score_display', row.get('Piotroski_Score', 'N/A')))),
        'safe': is_exempt_financial or (score is not None and score > 3),
        'source': source,
    }

f_score_lookup = {}
if universe_fundamentals is not None and not universe_fundamentals.empty and 'Ticker' in universe_fundamentals.columns:
    for _, f_row in universe_fundamentals.iterrows():
        f_score_lookup[f_row['Ticker']] = _f_info_from_row(f_row)

inflow_lookup = {}
if not multifactor_scorecard_df.empty and 'Ticker' in multifactor_scorecard_df.columns:
    for _, sc_row in multifactor_scorecard_df.iterrows():
        tk = sc_row['Ticker']
        if tk not in f_score_lookup:
            f_score_lookup[tk] = _f_info_from_row(sc_row)

        ratio_val = float(sc_row.get('Accumulation_Ratio', sc_row.get('Vol_Accum_Ratio', 1.0)))
        badge_val = str(sc_row.get('Institutional Inflow', sc_row.get('Institutional_Inflow_Badge', '⚪ Neutral Flow')))
        inflow_lookup[tk] = {'ratio': ratio_val, 'badge': badge_val}

order_tbl_data = []
for t in all_relevant_tickers:
    if t == SOVEREIGN_BOND_TICKER: continue
    try:
        p = float(latest_prices_series.get(t, 0.0))
        if pd.isna(p) or p < 0: p = 0.0
    except Exception: p = 0.0
        
    q = owned_summary[t]['total_shares'] if t in owned_summary else 0
    q_ltcg = owned_summary[t]['ltcg_shares'] if t in owned_summary else 0
    curr_val = q * p
    t_weight = float(target_w_series.get(t, 0.0))
    target_val = target_risky_wealth * t_weight
    gap = target_val - curr_val
    rel_drift = (curr_val - target_val) / target_val if target_val > 0 else (999.0 if curr_val > 0 else 0.0)
    
    # 200-DMA Trend Filter Metrics
    sma_val = float(sma_200_series.get(t, np.nan))
    if pd.isna(sma_val) or sma_val <= 0:
        trend_tag = "⚪ N/A (Hedge/New)"
        sma_display = np.nan
    elif p >= sma_val:
        trend_tag = "🟢 UPTREND (P ≥ SMA200)"
        sma_display = np.round(sma_val, 2)
    else:
        trend_tag = "🔴 DOWNTREND (P < SMA200)"
        sma_display = np.round(sma_val, 2)

    is_red_flag = (gov_flags.get(t) == 'RED_FLAG')
    red_trigger = triggers.get(t, '')
    
    # Piotroski F-Score Information
    f_info = f_score_lookup.get(t, _UNSYNCED_F_INFO)
    f_sc = f_info['score']
    f_badge = f_info['badge']
    f_disp = f_info['display']

    # Institutional Inflow Information
    inflow_info = inflow_lookup.get(t, {'ratio': 1.0, 'badge': '⚪ Neutral Flow'})
    inflow_badge = inflow_info['badge']
    inflow_ratio = inflow_info['ratio']

    sec = get_asset_sector(t)
    if is_red_flag:
        init_action = "🔴 NEWS CIRCUIT BREAKER (GOVERNANCE ALERT)"
        init_rationale = f"Governance Alert: {red_trigger}" if red_trigger else "High-Risk News Catalyst / Regulatory Flag"
    elif not f_info['safe']:
        init_action = "⛔ FUNDAMENTALS NOT SYNCED (BUY BLOCKED)" if f_sc is None else "⛔ F-SCORE DETERIORATING (BUY BLOCKED)"
        init_rationale = (
            "No audited Screener.in fundamentals synced for this ticker — Buy Blocked pending sync"
            if f_sc is None else
            f"F-Score: {f_disp} ({f_badge}) — Balance Sheet Deteriorating (Buy Blocked)"
        )
    elif t_weight > 0:
        init_action = '⚪ HOLD'
        if inflow_ratio >= 2.0:
            init_rationale = f"Max-Sharpe Conviction ({t_weight*100:.1f}%) | 🔥 Heavy Inflow ({inflow_ratio:.1f}x Vol)"
        elif inflow_ratio >= 1.3:
            init_rationale = f"Max-Sharpe Tangency ({t_weight*100:.1f}%) | 🟢 Steady Inflow ({inflow_ratio:.1f}x Vol)"
        else:
            init_rationale = f"Max-Sharpe Tangency Weight ({t_weight*100:.1f}%)"
    else:
        init_action = '⚪ HOLD'
        init_rationale = "Excess/Purged Lot"

    order_tbl_data.append({
        'Asset': owned_summary[t]['asset_name'] if t in owned_summary else t.replace('.NS',''),
        'Sector': sec,
        'Class': owned_summary[t]['asset_class'] if t in owned_summary else ('Real Estate (REIT)' if 'EMBASSY' in t else ('Gold ETF' if 'GOLD' in t else 'Large-Cap Equity')),
        'Live Price (₹)': p,
        'SMA-200 (₹)': sma_display,
        'Trend (200 DMA)': trend_tag,
        'Institutional Inflow': inflow_badge,
        'Accumulation Ratio': f"{inflow_ratio:.2f}x",
        'Shares Owned': q,
        'LTCG Shares': q_ltcg,
        'Current Value (₹)': curr_val,
        'Target Value (₹)': np.round(target_val, 2),
        'Mathematical Gap (₹)': np.round(gap, 2),
        'Relative Drift (%)': np.round(rel_drift * 100, 2),
        'Target Wt (%)': np.round(t_weight * w_risky * 100, 2),
        'Shares to Buy': 0,
        'Shares to Sell': 0,
        'Action': init_action,
        'Quantitative Rationale': init_rationale
    })

order_tbl = pd.DataFrame(order_tbl_data, index=[t for t in all_relevant_tickers if t != SOVEREIGN_BOND_TICKER])

# PASS 0: TAX-OPTIMIZED CAPITAL HARVESTING WITH DP FEE SHIELD & STCG TAX SHIELD
total_harvested_from_sells = 0.0

for t in order_tbl.index:
    p = order_tbl.loc[t, 'Live Price (₹)']
    target_val = order_tbl.loc[t, 'Target Value (₹)']
    curr_val = order_tbl.loc[t, 'Current Value (₹)']
    q_total = order_tbl.loc[t, 'Shares Owned']
    q_ltcg = order_tbl.loc[t, 'LTCG Shares']
    f_info = f_score_lookup.get(t, _UNSYNCED_F_INFO)
    f_disp = f_info['display']
    if p <= 0: continue
        
    if target_val == 0.0 and q_total > 0:
        order_tbl.loc[t, 'Shares to Sell'] = q_total
        order_tbl.loc[t, 'Action'] = f"🔴 LIQUIDATE {q_total} Share(s)"
        order_tbl.loc[t, 'Quantitative Rationale'] = f"Liquidate Purged Asset (0% Target Weight) | F-Score: {f_disp}"
        total_harvested_from_sells += (q_total * p * 0.9975)
    elif q_total > 0 and target_val > 0 and ((curr_val - target_val) / target_val) > tolerance_decimal:
        desired_trim = int((curr_val - target_val) // p)
        allowed_trim = min(desired_trim, q_ltcg)
        if allowed_trim > 0:
            disposal_val = allowed_trim * p
            if disposal_val >= MIN_DISPOSAL_VALUE:
                order_tbl.loc[t, 'Shares to Sell'] = allowed_trim
                order_tbl.loc[t, 'Action'] = f"🔴 TRIM {allowed_trim} Share(s)"
                order_tbl.loc[t, 'Quantitative Rationale'] = f"LTCG Rebalance Trim (>{drift_tolerance_pct}% Drift: +{((curr_val - target_val) / target_val)*100:.1f}%) | F-Score: {f_disp}"
                total_harvested_from_sells += (allowed_trim * p * 0.9975)
            else:
                order_tbl.loc[t, 'Shares to Sell'] = 0
                order_tbl.loc[t, 'Action'] = "🟡 MICRO-TRIM SKIPPED (DP FEE SHIELD)"
                order_tbl.loc[t, 'Quantitative Rationale'] = f"Disposal value (₹{disposal_val:.2f}) < Min Floor (₹{MIN_DISPOSAL_VALUE:.0f}) | F-Score: {f_disp}"
        else:
            order_tbl.loc[t, 'Shares to Sell'] = 0
            order_tbl.loc[t, 'Action'] = "🟡 TAX SHIELD: PROTECTED (STCG < 1Y)"
            order_tbl.loc[t, 'Quantitative Rationale'] = f"STCG Protection: Holding < 365 days locked to avoid 20% tax penalty | F-Score: {f_disp}"

# HOLISTIC CAPITAL REALLOCATION
total_deployable_capital = total_available_broker_cash + total_harvested_from_sells
sold_risky_val = sum(order_tbl['Shares to Sell'] * order_tbl['Live Price (₹)'])
post_sell_risky_wealth = current_risky_wealth - sold_risky_val
post_sell_safe_wealth = current_safe_wealth

total_post_sell_wealth = post_sell_risky_wealth + post_sell_safe_wealth + total_deployable_capital
final_target_risky_wealth = total_post_sell_wealth * w_risky
final_target_safe_wealth  = total_post_sell_wealth * w_safe

risky_deficit = max(0.0, final_target_risky_wealth - post_sell_risky_wealth)
safe_deficit  = max(0.0, final_target_safe_wealth - post_sell_safe_wealth)
total_combined_deficit = risky_deficit + safe_deficit

if total_combined_deficit > 0:
    fresh_risky_cash = total_deployable_capital * (risky_deficit / total_combined_deficit)
    fresh_safe_cash  = total_deployable_capital * (safe_deficit / total_combined_deficit)
else:
    fresh_risky_cash = total_deployable_capital * w_risky
    fresh_safe_cash  = total_deployable_capital * w_safe

# PASS 1 & 2: INFLOW-FIRST FRICTION-ADJUSTED BUY ALLOCATION WITH 200-DMA BINARY TREND FILTER & PIOTROSKI QUALITY GUARD
cash_pool = float(fresh_risky_cash)

for t in order_tbl.index:
    t_weight = float(target_w_series.get(t, 0.0))
    t_target_val = final_target_risky_wealth * t_weight
    p = order_tbl.loc[t, 'Live Price (₹)']
    curr_remaining_q = order_tbl.loc[t, 'Shares Owned'] - order_tbl.loc[t, 'Shares to Sell']
    curr_remaining_val = curr_remaining_q * p
    new_gap = max(0.0, t_target_val - curr_remaining_val)
    order_tbl.loc[t, 'Target Value (₹)'] = np.round(t_target_val, 2)
    order_tbl.loc[t, 'Mathematical Gap (₹)'] = np.round(new_gap, 2)
    order_tbl.loc[t, 'Relative Drift (%)'] = np.round(
        ((curr_remaining_val - t_target_val) / t_target_val * 100.0) if t_target_val > 0 else (999.0 if curr_remaining_val > 0 else 0.0),
        2
    )

# Pass 1: Prioritized buy allocation for constituents with largest negative drift (underweight assets)
# Governed by Governance Circuit Breaker, Piotroski F-Score Gate, & 200-DMA Trend Filter Check
pass1_priority_order = order_tbl.sort_values(
    by=['Relative Drift (%)', 'Mathematical Gap (₹)'],
    ascending=[True, False]
).index

for t in pass1_priority_order:
    p = order_tbl.loc[t, 'Live Price (₹)']
    if p <= 0: continue
    p_gross = p * (1.0 + get_asset_friction(t, order_tbl.loc[t, 'Class']))
    gap = order_tbl.loc[t, 'Mathematical Gap (₹)']
    sma_val = order_tbl.loc[t, 'SMA-200 (₹)']
    is_red_flag = (gov_flags.get(t) == 'RED_FLAG')
    red_trigger = triggers.get(t, '')
    f_info = f_score_lookup.get(t, _UNSYNCED_F_INFO)
    f_sc = f_info['score']
    f_badge = f_info['badge']
    f_disp = f_info['display']

    # NLP News Sentiment & Governance Circuit Breaker: Block new BUY allocations
    if is_red_flag:
        order_tbl.loc[t, 'Shares to Buy'] = 0
        if gap > 0 or order_tbl.loc[t, 'Shares Owned'] > 0:
            order_tbl.loc[t, 'Action'] = "🔴 NEWS CIRCUIT BREAKER (GOVERNANCE ALERT)"
            order_tbl.loc[t, 'Quantitative Rationale'] = f"Governance Alert: {red_trigger}" if red_trigger else "High-Risk News Catalyst / Regulatory Flag"
        continue

    # Piotroski F-Score Safety Gate: Block new BUY allocations if F-Score <= 3, OR if there is
    # no verified audited data at all (never a silent neutral pass for missing/failed sync).
    # Financial Institutions are explicitly exempted (f_info['safe']=True) since the 9-point
    # score doesn't apply to them -- see FINANCIAL_SECTOR_NAMES in quant_engine.py.
    if not f_info['safe']:
        order_tbl.loc[t, 'Shares to Buy'] = 0
        if gap > 0 or order_tbl.loc[t, 'Shares Owned'] > 0:
            if f_sc is None:
                order_tbl.loc[t, 'Action'] = "⛔ FUNDAMENTALS NOT SYNCED (BUY BLOCKED)"
                order_tbl.loc[t, 'Quantitative Rationale'] = "No audited Screener.in fundamentals synced for this ticker — Fresh Capital Blocked pending sync"
            else:
                order_tbl.loc[t, 'Action'] = "⛔ F-SCORE DETERIORATING (BUY BLOCKED)"
                order_tbl.loc[t, 'Quantitative Rationale'] = f"F-Score Decay ({f_disp} — {f_badge}) | Decaying Balance Sheet — Fresh Capital Blocked"
        continue

    # 200-DMA Binary Trend Filter: Block new BUY allocations if Live Price < SMA_200
    if pd.notna(sma_val) and sma_val > 0 and p < sma_val:
        order_tbl.loc[t, 'Shares to Buy'] = 0
        if gap > 0:
            order_tbl.loc[t, 'Action'] = "🟡 SMA-200 DOWNTREND FILTER (BUY BLOCKED)"
            order_tbl.loc[t, 'Quantitative Rationale'] = f"200-DMA Downtrend: Price (₹{p:.2f}) < SMA-200 (₹{sma_val:.2f}) | F-Score: {f_disp} — Buy Blocked"
        continue

    if order_tbl.loc[t, 'Target Value (₹)'] > 0 and gap > 0 and cash_pool >= p_gross:
        desired_buy_q = int(gap // p_gross)
        max_possible_q = int(cash_pool // p_gross)
        buy_q = min(desired_buy_q, max_possible_q)
        if buy_q > 0:
            order_tbl.loc[t, 'Shares to Buy'] += buy_q
            cash_pool -= (buy_q * p_gross)

# Pass 2: Greedy Reallocation of unspent cash pool prioritizing largest remaining negative drift
while cash_pool > 0:
    eligible_mask = (
        (order_tbl['Target Value (₹)'] > 0) & 
        (order_tbl.index.map(lambda sym: gov_flags.get(sym, 'NORMAL') != 'RED_FLAG')) &
        (order_tbl.index.map(lambda sym: f_score_lookup.get(sym, _UNSYNCED_F_INFO)['safe'])) &
        ((order_tbl['Live Price (₹)'] >= order_tbl['SMA-200 (₹)']) | order_tbl['SMA-200 (₹)'].isna()) &
        (order_tbl['Live Price (₹)'] > 0)
    )
    eligible_df = order_tbl[eligible_mask].copy()
    if eligible_df.empty:
        break
    
    eligible_df['rem_gap'] = eligible_df['Mathematical Gap (₹)'] - (eligible_df['Shares to Buy'] * eligible_df['Live Price (₹)'])
    eligible_df['rem_drift'] = (
        (eligible_df['Shares Owned'] - eligible_df['Shares to Sell'] + eligible_df['Shares to Buy']) * eligible_df['Live Price (₹)'] - eligible_df['Target Value (₹)']
    ) / eligible_df['Target Value (₹)']
    
    underweight_candidates = eligible_df[eligible_df['rem_gap'] > 0].sort_values(
        by=['rem_drift', 'rem_gap'],
        ascending=[True, False]
    )
    if underweight_candidates.empty:
        break
        
    bought_any = False
    for t in underweight_candidates.index:
        p = order_tbl.loc[t, 'Live Price (₹)']
        p_gross = p * (1.0 + get_asset_friction(t, order_tbl.loc[t, 'Class']))
        rem_gap = order_tbl.loc[t, 'Mathematical Gap (₹)'] - (order_tbl.loc[t, 'Shares to Buy'] * p)
        if cash_pool >= p_gross and p_gross <= (rem_gap * 1.35):
            order_tbl.loc[t, 'Shares to Buy'] += 1
            cash_pool -= p_gross
            bought_any = True
    if not bought_any:
        break

for t in order_tbl.index:
    b = int(order_tbl.loc[t, 'Shares to Buy'])
    if b > 0:
        f_info = f_score_lookup.get(t, _UNSYNCED_F_INFO)
        order_tbl.loc[t, 'Action'] = f"🟢 BUY {b} Share(s)"
        order_tbl.loc[t, 'Quantitative Rationale'] = f"F-Score: {f_info['display']} ({f_info['badge']}) | Gap: ₹{order_tbl.loc[t, 'Mathematical Gap (₹)']:,.0f} | 200-DMA Uptrend"

# PASS 3: PHYSICAL SOVEREIGN G-SEC BOND ALLOCATION (SECTION 50AA)
try:
    p_bond = float(latest_prices_series.get(SOVEREIGN_BOND_TICKER, 55.0))
    if pd.isna(p_bond) or p_bond <= 0: p_bond = 55.0
except Exception: p_bond = 55.0
    
p_bond_gross = p_bond * (1.0 + ETF_DEBT_GOLD_FRICTION)
total_safe_cash_pool = fresh_safe_cash + cash_pool
bonds_to_buy = int(total_safe_cash_pool // p_bond_gross)
unspent_bond_cash = total_safe_cash_pool - (bonds_to_buy * p_bond_gross)

bond_rationale = (
    f"🛡️ Endpoint LDI Cash Shield (T={years_to_goal:.1f}Y ≤ 3Y) | Sovereign Anchor (Sec 50AA | 7.20% YTM)"
    if years_to_goal <= 3.0 else
    "Sovereign Fixed Income Anchor (Sec 50AA | 7.20% YTM)"
)

bond_row = pd.DataFrame([{
    'Asset': 'GILT5YBEES (5-Yr G-Sec)', 'Sector': 'Sovereign Fixed Income (Sec 50AA)',
    'Class': 'Sovereign Debt ETF (Sec 50AA)', 'Live Price (₹)': p_bond,
    'SMA-200 (₹)': np.nan, 'Trend (200 DMA)': '🟢 Sovereign Anchor',
    'Institutional Inflow': '⚪ Neutral Flow',
    'Accumulation Ratio': '1.00x',
    'Shares Owned': owned_summary.get(SOVEREIGN_BOND_TICKER, {}).get('total_shares', 0),
    'LTCG Shares': owned_summary.get(SOVEREIGN_BOND_TICKER, {}).get('ltcg_shares', 0),
    'Current Value (₹)': owned_summary.get(SOVEREIGN_BOND_TICKER, {}).get('total_shares', 0) * p_bond,
    'Target Value (₹)': np.round(final_target_safe_wealth, 2),
    'Mathematical Gap (₹)': np.round(final_target_safe_wealth - (owned_summary.get(SOVEREIGN_BOND_TICKER, {}).get('total_shares', 0) * p_bond), 2),
    'Relative Drift (%)': 0.0, 'Target Wt (%)': np.round(w_safe * 100, 2), 'Shares to Buy': bonds_to_buy, 'Shares to Sell': 0,
    'Action': f"🟢 BUY {bonds_to_buy} Unit(s)" if bonds_to_buy > 0 else "⚪ HOLD",
    'Quantitative Rationale': bond_rationale
}], index=[SOVEREIGN_BOND_TICKER])

complete_order_tbl = pd.concat([order_tbl, bond_row])

total_invested_capital = float((tax_lots_df['quantity'] * tax_lots_df['buy_price']).sum())
current_holding_wealth = current_risky_wealth + current_safe_wealth
unrealized_pl = current_holding_wealth - total_invested_capital
unrealized_pl_pct = (unrealized_pl / total_invested_capital * 100) if total_invested_capital > 0 else 0.0

# Calculate exact annualized money-weighted XIRR from SQLite database
with get_db_connection() as conn:
    xirr_res = compile_xirr_cash_flows(conn, current_holding_wealth, xirr_solver_func=compute_portfolio_xirr)

xirr_value = xirr_res['xirr_float']
xirr_display = xirr_res['xirr_text']
xirr_cf_df = xirr_res['cf_df']

actual_spent_on_buys = sum(complete_order_tbl['Shares to Buy'] * complete_order_tbl['Live Price (₹)'])
estimated_fees       = sum(complete_order_tbl.loc[t, 'Shares to Buy'] * complete_order_tbl.loc[t, 'Live Price (₹)'] * get_asset_friction(t, complete_order_tbl.loc[t, 'Class']) for t in complete_order_tbl.index)
harvested_from_sells = sum(complete_order_tbl['Shares to Sell'] * complete_order_tbl['Live Price (₹)'])
total_residual_cash  = unspent_bond_cash

# ----------------------------------------------------------------------------------------------------
# 5. SUMMARY KPIS & TOP METRIC CARDS
# ----------------------------------------------------------------------------------------------------
kpi_row1_col1, kpi_row1_col2, kpi_row1_col3, kpi_row1_col4, kpi_row1_col5 = st.columns(5)
with kpi_row1_col1:
    st.metric("Total Net Worth", f"₹{total_portfolio_wealth:,.2f}")
with kpi_row1_col2:
    st.metric("Total Capital Invested", f"₹{total_invested_capital:,.2f}")
with kpi_row1_col3:
    st.metric("Unrealized P/L", f"₹{unrealized_pl:,.2f}", f"{unrealized_pl_pct:+.2f}%")
with kpi_row1_col4:
    st.metric("Personal Money-Weighted IRR (XIRR)", xirr_display, "Annualized Compounded" if pd.notna(xirr_value) else None)
with kpi_row1_col5:
    alloc_delta = f"Age {exact_age:.0f} (T={years_to_goal:.1f}Y) | {'🛡️ LDI Shield' if years_to_goal <= 3.0 else regime_tag}"
    st.metric("Target Allocation", f"{w_risky*100:.0f}% Eq / {w_safe*100:.0f}% Bond", alloc_delta)

kpi_row2_col1, kpi_row2_col2, kpi_row2_col3, kpi_row2_col4 = st.columns(4)
with kpi_row2_col1:
    st.metric("Fresh Cash Deployed", f"₹{actual_spent_on_buys:,.2f}", f"{(actual_spent_on_buys/total_available_broker_cash)*100:.1f}%" if total_available_broker_cash > 0 else "0.0%")
with kpi_row2_col2:
    st.metric("Est. Statutory Fees (STT)", f"₹{estimated_fees:,.2f}", "Exchange Friction")
with kpi_row2_col3:
    st.metric("Capital Harvested", f"₹{harvested_from_sells:,.2f}")
with kpi_row2_col4:
    st.metric("Residual Wallet Cash", f"₹{total_residual_cash:,.2f}")

# ----------------------------------------------------------------------------------------------------
# 6. ACID TRANSACTION COMMIT WITH PRE-COMMIT SAFETY GUARD & AUDIT LEDGER
# ----------------------------------------------------------------------------------------------------

# Persistent Post-Execution Feedback Banner
if st.session_state.get('last_commit_info'):
    info = st.session_state.last_commit_info
    lot_list_str = ", ".join(f"Lot #{lid}" for lid in info['all_lots']) if info['all_lots'] else "None"
    st.success(
        f"✅ **Persistent Execution Log** | Exact Timestamp: `{info['timestamp']}` | "
        f"Updated / Created Lot IDs: **{lot_list_str}** | "
        f"Total Orders: {info['total_buys']} Buy(s) (₹{info['total_spent']:,.2f}), {info['total_sells']} Sell(s) (₹{info['total_harvested']:,.2f}) committed to `{DB_FILE}`."
    )

buy_mask = complete_order_tbl['Shares to Buy'] > 0
sell_mask = complete_order_tbl['Shares to Sell'] > 0

total_buys_count = int(buy_mask.sum())
total_inr_to_deploy = float(sum(complete_order_tbl.loc[buy_mask, 'Shares to Buy'] * complete_order_tbl.loc[buy_mask, 'Live Price (₹)']))

total_sells_count = int(sell_mask.sum())
capital_harvested_val = float(sum(complete_order_tbl.loc[sell_mask, 'Shares to Sell'] * complete_order_tbl.loc[sell_mask, 'Live Price (₹)']))

estimated_friction_val = float(
    sum(complete_order_tbl.loc[t, 'Shares to Buy'] * complete_order_tbl.loc[t, 'Live Price (₹)'] * get_asset_friction(t, complete_order_tbl.loc[t, 'Class']) for t in complete_order_tbl.index if complete_order_tbl.loc[t, 'Shares to Buy'] > 0) +
    sum(complete_order_tbl.loc[t, 'Shares to Sell'] * complete_order_tbl.loc[t, 'Live Price (₹)'] * get_asset_friction(t, complete_order_tbl.loc[t, 'Class']) for t in complete_order_tbl.index if complete_order_tbl.loc[t, 'Shares to Sell'] > 0)
)
remaining_unallocated_cash = float(total_residual_cash)

with st.expander("🛡️ Pre-Commit Execution Summary & Confirmation Safety Guard", expanded=True):
    st.markdown("#### 📋 Pre-Commit Execution Summary")
    st.caption("Review calculated trade quantities, cash deployment, and tax friction before committing to the immutable SQLite ledger.")
    
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    with sum_col1:
        st.metric("🟢 Total Buy Orders", f"{total_buys_count} Order(s)", f"₹{total_inr_to_deploy:,.2f} to Deploy")
    with sum_col2:
        st.metric("🔴 Total Sell Orders", f"{total_sells_count} Order(s)", f"₹{capital_harvested_val:,.2f} Harvested")
    with sum_col3:
        st.metric("⚡ Est. STT & Friction", f"₹{estimated_friction_val:,.2f}", "Brokerage & Statutory")
    with sum_col4:
        st.metric("💵 Remaining Cash", f"₹{remaining_unallocated_cash:,.2f}", "Unallocated Wallet Cash")

    st.markdown("---")
    force_commit = st.checkbox(
        "⚡ Force Commit Additional Orders (Allow Same-Day Tranche)",
        value=False,
        key="main_force_commit",
        help="Check this if you are intentionally executing an additional trade tranche or rebalance on the same calendar day."
    )
    confirm_commit = st.checkbox(
        "🔒 Confirm accuracy of broker orders before updating SQLite ledger",
        value=False,
        key="main_confirm_commit"
    )
    execute_button = st.button(
        "💾 Commit Trades to SQLite Database",
        type="primary",
        disabled=not confirm_commit,
        key="main_execute_btn"
    )

final_confirm = confirm_commit or sb_confirm_commit
final_execute = execute_button or sb_execute_button
allow_same_day = force_commit or sb_force_commit

if final_execute:
    if not final_confirm:
        st.warning("⚠️ Please check 'Confirm accuracy of broker orders before updating SQLite ledger' before committing.")
    elif st.session_state.get('committed_today', False) and not allow_same_day:
        st.warning("⚠️ **Double-Click Safety Guard Active:** This order basket was already committed. If you intend to execute an additional trade tranche today, please check **'Force Commit Additional Orders'**.")
    else:
        backup_database()
        affected_lot_ids = []
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Execute Sells (FIFO disposal matching)
            for t in complete_order_tbl.index:
                sells_needed = float(complete_order_tbl.loc[t, 'Shares to Sell'])
                if sells_needed > 0:
                    p_sell = float(complete_order_tbl.loc[t, 'Live Price (₹)'])
                    lots_cursor = conn.cursor()
                    lots_cursor.execute("SELECT lot_id, buy_date, quantity, buy_price, asset_class FROM tax_lots WHERE ticker = ? AND quantity > 0 ORDER BY buy_date ASC, lot_id ASC", [t])
                    for lot_id, buy_date_str, lot_qty, lot_buy_price, lot_asset_class in lots_cursor.fetchall():
                        if isinstance(lot_qty, bytes):
                            try: lot_qty = float(int.from_bytes(lot_qty, 'little'))
                            except Exception: lot_qty = 0.0
                        lot_qty = round(float(lot_qty), 4)
                        lot_buy_price = float(lot_buy_price)
                        if sells_needed <= 1e-6: break
                        
                        shares_from_lot = min(sells_needed, lot_qty)
                        realized_gain = round((p_sell - lot_buy_price) * shares_from_lot, 2)
                        
                        buy_dt = pd.to_datetime(buy_date_str).tz_localize(None) if getattr(pd.to_datetime(buy_date_str), 'tzinfo', None) is not None else pd.to_datetime(buy_date_str)
                        days_held = (today_naive - buy_dt).days
                        
                        is_50aa = (t == SOVEREIGN_BOND_TICKER) or ('50AA' in str(lot_asset_class)) or ('Debt' in str(lot_asset_class))
                        if is_50aa:
                            gain_type = 'Section 50AA (Debt STCG)'
                        elif days_held >= 365:
                            gain_type = 'LTCG'
                        else:
                            gain_type = 'STCG'
                            
                        fees_est = round(shares_from_lot * p_sell * get_asset_friction(t, lot_asset_class), 2)

                        if lot_qty <= shares_from_lot + 1e-6:
                            cursor.execute("DELETE FROM tax_lots WHERE lot_id = ?", [int(lot_id)])
                        else:
                            rem_qty = round(float(lot_qty - shares_from_lot), 4)
                            cursor.execute("UPDATE tax_lots SET quantity = ? WHERE lot_id = ?", [rem_qty, int(lot_id)])
                        
                        sells_needed -= shares_from_lot

                        cursor.execute("""
                            INSERT INTO trade_ledger (execution_date, ticker, action, quantity, price, realized_pnl, gain_type, fees_estimated)
                            VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?)
                        """, [TODAY_ISO, str(t), float(round(shares_from_lot, 4)), round(p_sell, 2), realized_gain, gain_type, fees_est])
                        affected_lot_ids.append(int(lot_id))

            # 2. Execute Buys
            for t in complete_order_tbl.index:
                buys_count = float(complete_order_tbl.loc[t, 'Shares to Buy'])
                if buys_count > 0:
                    p_buy = float(complete_order_tbl.loc[t, 'Live Price (₹)'])
                    fees_est = round(buys_count * p_buy * get_asset_friction(t, complete_order_tbl.loc[t, 'Class']), 2)
                    
                    cursor.execute("""
                        INSERT INTO tax_lots (ticker, asset_name, asset_class, buy_date, quantity, buy_price, last_split_date)
                        VALUES (?, ?, ?, ?, ?, ?, '')
                    """, [str(t), str(complete_order_tbl.loc[t, 'Asset']), str(complete_order_tbl.loc[t, 'Class']), TODAY_ISO, float(round(buys_count, 4)), round(p_buy, 2)])
                    
                    new_id = cursor.lastrowid
                    if new_id:
                        affected_lot_ids.append(int(new_id))

                    cursor.execute("""
                        INSERT INTO trade_ledger (execution_date, ticker, action, quantity, price, realized_pnl, gain_type, fees_estimated)
                        VALUES (?, ?, 'BUY', ?, ?, 0.0, 'N/A', ?)
                    """, [TODAY_ISO, str(t), float(round(buys_count, 4)), round(p_buy, 2), float(fees_est)])
                    
            conn.commit()

        # Reconcile corporate actions / stock splits automatically upon trade execution
        split_msgs = reconcile_corporate_actions_and_splits()
        if split_msgs:
            for msg in split_msgs:
                st.info(msg)

        commit_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.committed_today = True
        st.session_state.last_commit_info = {
            'timestamp': commit_timestamp,
            'all_lots': sorted(list(set(affected_lot_ids))),
            'total_buys': total_buys_count,
            'total_sells': total_sells_count,
            'total_spent': total_inr_to_deploy,
            'total_harvested': capital_harvested_val
        }
        lot_list_str = ", ".join(f"Lot #{lid}" for lid in st.session_state.last_commit_info['all_lots']) if st.session_state.last_commit_info['all_lots'] else "None"
        st.success(f"💾 **Trades Successfully Committed!** | Timestamp: `{commit_timestamp}` | Affected Lots: **{lot_list_str}** | Relational FIFO tax lots & Immutable Audit Ledger committed to SQLite ('{DB_FILE}') with ACID safety!")
        st.balloons()
        st.cache_data.clear()
        st.rerun()

# ----------------------------------------------------------------------------------------------------
# 7. TABS INTERFACE: DEMAT TICKET, VISUAL SUITE, MULTI-FACTOR SCORECARD, TAX HARVESTING & LEDGER
# ----------------------------------------------------------------------------------------------------
tab_ticket, tab_visuals, tab_dupont, tab_harvest, tab_db = st.tabs([
    "📋 Actionable Demat Ticket",
    "📈 Quantitative Visual Suite",
    "📊 Multi-Factor Quality & Momentum Scorecard",
    "💎 Section 112A Tax Harvester",
    "📂 SQLite Audit & Tax Ledger"
])

# --- TAB 1: DEMAT TICKET ---
with tab_ticket:
    st.subheader("📋 Actionable Demat Trade Ticket (Quantamental Wealth Engine v12.0)")
    if years_to_goal <= 3.0:
        st.warning("🛡️ **Endpoint LDI Shield Active (T ≤ 3Y):** 3-Year cash needs are locked into Sovereign Debt (GILT5YBEES). Portfolio is 100% insulated from sudden market crashes / pandemics on your target encashment date.")
    if total_available_broker_cash > 0:
        st.info("💡 **Tax-Free Cash Rebalancing Active:** Fresh cash is automatically routed to underweight assets to restore optimal tangency with zero sell friction.")
    st.info("💡 **Instructions for Zerodha / Groww:** Place Delivery (CNC) Market Orders for the 🟢 BUY signals below, then click '💾 Commit Trades to SQLite Database' in the sidebar to lock your ledger.")
    st.warning(
        "⚠️ **Before you place any order:** the Piotroski F-Score gate now runs on audited financials "
        "extracted from Screener.in (see the 'Multi-Factor Quality & Momentum Scorecard' tab → **🔄 Sync "
        "Audited Filings** and the **Fundamentals Source** column), but this extraction has never been "
        "verified against a live Screener.in page -- run a sync yourself and spot-check a few tickers' "
        "numbers in your browser before trusting a single order below. A ⛔ **FUNDAMENTALS NOT SYNCED** "
        "action means this ticker has no verified data at all and is blocked until you sync."
    )
    
    with st.expander("⚡ 1-Click Zerodha Kite Basket Order Exporter", expanded=False):
        basket_mode = st.radio(
            "Basket Export Mode:",
            options=["Export Full Basket (Buys + Sells)", "Buys Only"],
            index=0,
            horizontal=True,
            key="kite_basket_mode_toggle"
        )
        basket_items = []
        if "Buys + Sells" in basket_mode:
            for t in complete_order_tbl.index:
                s_qty = int(complete_order_tbl.loc[t, 'Shares to Sell'])
                if s_qty > 0:
                    sym = t.replace('.NS', '')
                    basket_items.append({
                        "variety": "regular",
                        "tradingsymbol": sym,
                        "exchange": "NSE",
                        "transaction_type": "SELL",
                        "order_type": "MARKET",
                        "quantity": s_qty,
                        "product": "CNC",
                        "validity": "DAY"
                    })
        for t in complete_order_tbl.index:
            b_qty = int(complete_order_tbl.loc[t, 'Shares to Buy'])
            if b_qty > 0:
                sym = t.replace('.NS', '')
                basket_items.append({
                    "variety": "regular",
                    "tradingsymbol": sym,
                    "exchange": "NSE",
                    "transaction_type": "BUY",
                    "order_type": "MARKET",
                    "quantity": b_qty,
                    "product": "CNC",
                    "validity": "DAY"
                })
        basket_json = json.dumps(basket_items, indent=2)
        st.code(basket_json, language='json')
        st.download_button(
            "📥 Download Zerodha Kite Basket (JSON)",
            data=basket_json,
            file_name=f"Kite_Basket_{TODAY_STR}.json",
            mime="application/json"
        )
        st.markdown("*Copy the JSON above or upload the JSON directly into Zerodha Kite Web -> Orders -> Baskets to execute all trades in one click.*")

    display_cols = [
        'Asset', 'Sector', 'Class', 'Live Price (₹)', 'SMA-200 (₹)', 'Trend (200 DMA)',
        'Institutional Inflow', 'Shares Owned',
        'Current Value (₹)', 'Target Value (₹)', 'Action', 'Quantitative Rationale'
    ]
    ticket_view = complete_order_tbl[display_cols].set_index('Asset')
    st.dataframe(ticket_view.style.format({
        'Live Price (₹)': lambda x: f"₹{x:.2f}" if pd.notna(x) else "-",
        'SMA-200 (₹)': lambda x: f"₹{x:.2f}" if pd.notna(x) else "-",
        'Current Value (₹)': '₹{:,.2f}',
        'Target Value (₹)': '₹{:,.2f}'
    }), use_container_width=True)

    with st.expander('📰 Live News & Catalyst Monitor', expanded=False):
        st.caption("Review recent news headlines and corporate catalysts for your selected portfolio constituents prior to placing orders.")
        if turbo_mode:
            st.info("⚡ **Presentation / Turbo Mode Active:** Live news scraping bypassed for instantaneous in-memory execution (<0.05s).")
            news_data = {}
        else:
            with st.spinner("Fetching latest news catalysts..."):
                news_data = fetch_recent_news(tickers, turbo_mode=False)
        for t in tickers:
            clean_name = t.replace('.NS', '')
            t_headlines = news_data.get(t, [])
            st.markdown(f"##### **{clean_name}** (`{t}`)")
            if t_headlines:
                for item in t_headlines:
                    title = item['title']
                    link = item['link']
                    publisher = item['publisher']
                    pub_date = item['pub_date']
                    date_info = f" • *{pub_date}*" if pub_date else ""
                    if link and link != '#':
                        st.markdown(f"- [{title}]({link}) — **{publisher}**{date_info}")
                    else:
                        st.markdown(f"- **{title}** — *{publisher}*{date_info}")
            else:
                st.info(f"No recent news articles found for {clean_name}.")
            st.markdown("---")

# --- TAB 2: VISUAL SUITE ---
with tab_visuals:
    st.subheader("📈 Quantitative Visual Suite")
    
    vis_row1_col1, vis_row1_col2 = st.columns(2)
    
    # FIGURE 1: MULTI-ASSET CORRELATION HEATMAP (PLOTLY)
    with vis_row1_col1:
        st.markdown("#### Figure 1: Multi-Asset Correlation Heatmap")
        corr_active = active_returns_df.corr()
        corr_active.index = clean_tickers
        corr_active.columns = clean_tickers
        fig1 = px.imshow(
            corr_active,
            text_auto='.2f',
            aspect='auto',
            color_continuous_scale='RdBu_r',
            range_color=[-1, 1],
            title="Empirical Correlation Matrix (Shrunk via Ledoit-Wolf)"
        )
        fig1.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig1, use_container_width=True)

    # FIGURE 2: OPTIMAL ALLOCATION PIE CHART
    with vis_row1_col2:
        st.markdown("#### Figure 2: Target Portfolio Allocation")
        fig4 = px.pie(
            values=w_optimal_live,
            names=clean_tickers,
            title=f"Target Capital Allocation ({opt_mode})",
            hole=0.4
        )
        fig4.update_traces(textposition='inside', textinfo='percent+label')
        fig4.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    
    # FIGURE 3: EFFICIENT FRONTIER
    st.markdown("#### Figure 3: Modern Portfolio Theory: Efficient Frontier & Capital Market Line (CML)")
    def get_p_metrics(w):
        ret = np.dot(w, active_mean_vector)
        vol = np.sqrt(np.dot(w.T, np.dot(active_cov_matrix, w)))
        sr = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0
        return ret, vol, sr

    # Ultra-fast vectorized Dirichlet Monte Carlo (< 0.05s)
    _, mc_rets, mc_vols, mc_srs = compute_vectorized_monte_carlo_frontier(
        active_cov_matrix, active_mean_vector, rf_rate=RISK_FREE_RATE, n_sim=3000
    )

    act_bounds = tuple((0.0, MAX_RETAIL_CAP) for _ in range(OPTIMAL_K))
    act_cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    act_init = np.ones(OPTIMAL_K) / OPTIMAL_K

    opt_sr_res = minimize(lambda w: -get_p_metrics(w)[2], act_init, method='SLSQP', bounds=act_bounds, constraints=act_cons)
    opt_mv_res = minimize(lambda w: np.dot(w.T, np.dot(active_cov_matrix, w)), act_init, method='SLSQP', bounds=act_bounds, constraints=act_cons)

    sr_ret_val, sr_vol_val, sr_sharpe_val = get_p_metrics(opt_sr_res.x)
    mv_ret_val, mv_vol_val, mv_sharpe_val = get_p_metrics(opt_mv_res.x)

    t_rets_trace = np.linspace(mv_ret_val, sr_ret_val * 1.05, 30)
    eff_vols_trace = []
    for tr in t_rets_trace:
        c = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}, {'type': 'eq', 'fun': lambda w: np.dot(w, active_mean_vector) - tr})
        res_tr = minimize(lambda w: np.dot(w.T, np.dot(active_cov_matrix, w)), act_init, method='SLSQP', bounds=act_bounds, constraints=c)
        eff_vols_trace.append(np.sqrt(res_tr.fun) if res_tr.success else np.nan)

    client_cml_ret = (w_safe * RISK_FREE_RATE * 100) + (w_risky * sr_ret_val * 100)
    client_cml_vol = w_risky * sr_vol_val * 100

    bench_ann_ret = bench_returns.mean() * 252
    bench_ann_vol = bench_returns.std() * np.sqrt(252)

    cml_x = np.linspace(0, sr_vol_val * 1.4, 50)
    cml_y = RISK_FREE_RATE + ((sr_ret_val - RISK_FREE_RATE) / sr_vol_val) * cml_x

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=mc_vols * 100,
        y=mc_rets * 100,
        mode='markers',
        marker=dict(
            color=mc_srs,
            colorscale='Viridis',
            size=5,
            opacity=0.35,
            showscale=True,
            colorbar=dict(title=dict(text=f'Sharpe Ratio<br>(Rf={RISK_FREE_RATE*100:.1f}%)', side='top'), x=1.02)
        ),
        name='Simulated Portfolios',
        hovertemplate='Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br>Sharpe: %{marker.color:.2f}<extra></extra>'
    ))

    fig2.add_trace(go.Scatter(
        x=np.array(eff_vols_trace) * 100,
        y=t_rets_trace * 100,
        mode='lines',
        line=dict(color='deepskyblue', width=3),
        name='Efficient Frontier (Markowitz)'
    ))

    fig2.add_trace(go.Scatter(
        x=cml_x * 100,
        y=cml_y * 100,
        mode='lines',
        line=dict(color='gold', width=2.5, dash='dash'),
        name='Capital Market Line (CML)'
    ))

    fig2.add_trace(go.Scatter(
        x=[sr_vol_val * 100],
        y=[sr_ret_val * 100],
        mode='markers',
        marker=dict(color='crimson', size=14, symbol='star', line=dict(color='white', width=1.5)),
        name=f'Optimal Risky Basket (Sharpe={sr_sharpe_val:.2f})'
    ))

    fig2.add_trace(go.Scatter(
        x=[mv_vol_val * 100],
        y=[mv_ret_val * 100],
        mode='markers',
        marker=dict(color='blue', size=12, symbol='circle', line=dict(color='white', width=1.5)),
        name=f'Minimum Variance Basket (Vol={mv_vol_val*100:.1f}%)'
    ))

    cml_target_name = (
        f'YOUR TARGET ALLOCATION ({w_risky*100:.0f}% Eq / {w_safe*100:.0f}% Bond) [🛡️ LDI Shield Active]'
        if years_to_goal <= 3.0 else
        f'YOUR TARGET ALLOCATION ({w_risky*100:.0f}% Eq / {w_safe*100:.0f}% Bond)'
    )
    fig2.add_trace(go.Scatter(
        x=[client_cml_vol],
        y=[client_cml_ret],
        mode='markers',
        marker=dict(color='lime', size=16, symbol='diamond', line=dict(color='black', width=2)),
        name=cml_target_name
    ))

    fig2.add_trace(go.Scatter(
        x=[bench_ann_vol * 100],
        y=[bench_ann_ret * 100],
        mode='markers',
        marker=dict(color='black', size=12, symbol='square', line=dict(color='white', width=1.5)),
        name=f'Nifty 50 Index (Vol={bench_ann_vol*100:.1f}%)'
    ))

    fig2.update_layout(
        title=dict(text=f'Efficient Frontier & Your CML Target (Live K* = {OPTIMAL_K} Assets)', x=0.5),
        xaxis_title='Annualized Volatility / Risk (%)',
        yaxis_title='Annualized Expected Return (%)',
        height=550,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.35, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # FIGURE 4: LIVE OUT-OF-SAMPLE BACKTEST
    st.markdown("#### Figure 4: Live Out-of-Sample Forward Backtest")
    st.caption(f"🛡️ **Unbiased Quant Rigor:** In-sample optimization window: **{BACKTEST_TRAIN_START_STR} to {OOS_SPLIT_STR}** | Strict Out-of-Sample Forward Evaluation: **{OOS_SPLIT_STR} to {TODAY_STR}**.")

    test_assets_raw = oos_assets
    test_bench_raw = oos_bench
    test_returns = test_assets_raw.pct_change().dropna()
    # Ensure test_bench_ret is a Series (not DataFrame) for scalar cumprod
    test_bench_raw_series = test_bench_raw
    if isinstance(test_bench_raw_series, pd.DataFrame):
        test_bench_raw_series = test_bench_raw_series.iloc[:, 0] if test_bench_raw_series.shape[1] >= 1 else test_bench_raw_series.mean(axis=1)
    test_bench_ret = test_bench_raw_series.pct_change().dropna()
    if isinstance(test_bench_ret, pd.DataFrame):
        test_bench_ret = test_bench_ret.iloc[:, 0]

    t_dates = test_returns.index.intersection(test_bench_ret.index)
    test_assets_raw = test_assets_raw.loc[t_dates]
    test_returns = test_returns.loc[t_dates]
    test_bench_ret = test_bench_ret.loc[t_dates]

    w_bt_series = pd.Series(bt_quant['w_optimal'], index=bt_quant['tickers']).reindex(test_assets_raw.columns).fillna(0.0)
    w_bt_sum = w_bt_series.sum()
    w_bt_optimal = (w_bt_series / w_bt_sum).values if w_bt_sum > 1e-10 else (np.ones(len(w_bt_series)) / len(w_bt_series))

    r_dates = test_assets_raw.index.to_series().resample('QE').max().dropna().tolist()
    rebal_w = pd.Series(index=test_assets_raw.index, dtype='float64')
    rebal_w.iloc[0] = 1.0
    cur_h = w_bt_optimal.copy()

    for i in range(1, len(test_assets_raw)):
        c_date = test_assets_raw.index[i]
        d_ratio = (test_assets_raw.iloc[i] / test_assets_raw.iloc[i-1]).values
        cur_h = cur_h * d_ratio
        t_val = np.sum(cur_h)
        rebal_w.iloc[i] = t_val
        if c_date in r_dates:
            cur_h = (t_val * 0.999) * w_bt_optimal

    rebal_daily_ret = rebal_w.pct_change().dropna()
    # Normalize rebal_w to start at 1.0
    rebal_w_init = rebal_w.iloc[0]
    if rebal_w_init is not None and rebal_w_init > 1e-10:
        rebal_w = rebal_w / rebal_w_init
    daily_rf = (1 + RISK_FREE_RATE)**(1/252) - 1
    user_strategy_ret = (w_risky * rebal_daily_ret) + (w_safe * daily_rf)

    # Use strictly in-sample training covariance for the MVP comparative benchmark (Zero OOS leakage)
    bt_train_df = master_data['backtest_train_df']
    valid_train_cols = [c for c in test_assets_raw.columns if c in bt_train_df.columns]
    if valid_train_cols:
        lw_in_sample = LedoitWolf().fit(bt_train_df[valid_train_cols])
        shrunk_cov_in_sample = lw_in_sample.covariance_ * 252
        bt_K = len(valid_train_cols)
        bt_init = np.ones(bt_K) / bt_K
        bt_bounds = tuple((0.0, MAX_RETAIL_CAP) for _ in range(bt_K))
        bt_cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bt_mv_res = minimize(lambda w: np.dot(w.T, np.dot(shrunk_cov_in_sample, w)), bt_init, method='SLSQP', bounds=bt_bounds, constraints=bt_cons)
        mv_weights = bt_mv_res.x if bt_mv_res.success else bt_init
        mv_oos_returns = test_returns[valid_train_cols].dot(mv_weights)
    else:
        mv_oos_returns = test_returns.mean(axis=1)

    strat_col_name = (
        f'★ Your Personal Strategy (Age {exact_age:.0f} | 🛡️ LDI T={years_to_goal:.1f}Y)'
        if years_to_goal <= 3.0 else
        f'★ Your Personal Strategy (Age {exact_age:.0f} CML)'
    )
    oos_df = pd.DataFrame({
        strat_col_name: (1 + user_strategy_ret).cumprod(),
        'Optimal Multi-Asset (Quarterly Rebalanced)': rebal_w,
        'Minimum Variance (MVP - Defensive)': (1 + mv_oos_returns).cumprod(),
        'Nifty 50 Benchmark': (1 + test_bench_ret).cumprod()
    }).dropna()

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=oos_df.index,
        y=oos_df[strat_col_name],
        mode='lines',
        line=dict(color='forestgreen', width=3),
        name=f'★ Your Personal Strategy ({w_risky*100:.0f}% Risky / {w_safe*100:.0f}% Safe)'
    ))
    fig3.add_trace(go.Scatter(
        x=oos_df.index,
        y=oos_df['Minimum Variance (MVP - Defensive)'],
        mode='lines',
        line=dict(color='royalblue', width=2, dash='dash'),
        name='Minimum Variance (MVP - Defensive)'
    ))
    fig3.add_trace(go.Scatter(
        x=oos_df.index,
        y=oos_df['Optimal Multi-Asset (Quarterly Rebalanced)'],
        mode='lines',
        line=dict(color='crimson', width=1.5, dash='dot'),
        name='100% Multi-Asset Engine (Rebalanced)'
    ))
    fig3.add_trace(go.Scatter(
        x=oos_df.index,
        y=oos_df['Nifty 50 Benchmark'],
        mode='lines',
        line=dict(color='black', width=2),
        name='Nifty 50 Benchmark'
    ))

    fig3.update_layout(
        title=dict(text=f'Live Out-of-Sample Forward Backtest ({OOS_SPLIT_STR} to {TODAY_STR})', x=0.5),
        xaxis_title='Date',
        yaxis_title='Portfolio Wealth Multiplier (Base = ₹1.0)',
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig3, use_container_width=True)

# --- TAB 3: DUPONT & MULTI-FACTOR SCORECARD ---
with tab_dupont:
    st.subheader("📊 Multi-Factor Intelligence & DuPont Analytics Hub (Nifty 200 Universe)")
    st.caption("ℹ️ **Integrated Quantitative Screening Engine:** Combines 90-Day ADTV Liquidity Gate (≥₹10 Cr), 3-Stage DuPont Quality (Margin × Turnover × Leverage), 12M-1M Momentum, 60D Realized Volatility Penalty, and Institutional Volume Accumulation ($Z_{accum}$) with Forensic Governance Circuit Breakers.")
    st.warning(
        "⚠️ **Fundamentals Data Provenance:** DuPont ROE, Net Margin, Debt/Equity, and the Piotroski F-Score "
        "are extracted from Screener.in's audited financial-statement tables (Profit & Loss, Balance Sheet, "
        "Cash Flow) via the sync below -- not a static in-code snapshot any more. That said, **this scraper has "
        "never been run against a live Screener.in page during development** (the environment it was built in "
        "blocks that site); before trusting a specific position's F-Score gate, sync and then spot-check its "
        "numbers against the actual screener.in page in your browser. A ticker with **Fundamentals Source = "
        "'Sync Failed'** means the parser didn't recognize that page's layout -- treat that as blocked, not "
        "as a passing score. Banks/NBFCs show **'Not Applicable (Financial Institution)'**: the 9-point "
        "Piotroski test doesn't transfer to their statement structure, so only ROE/D-E are shown for them."
    )

    # -------------------------------------------------------------------------
    # 0. AUDITED FILINGS SYNC -- the only way the fundamentals store gets populated or refreshed
    # -------------------------------------------------------------------------
    _last_synced = get_fundamentals_last_synced()
    sync_col1, sync_col2 = st.columns([1, 2])
    with sync_col1:
        sync_clicked = st.button("🔄 Sync Audited Filings from Screener.in", type="primary", use_container_width=True)
    with sync_col2:
        if _last_synced:
            st.caption(f"🕒 **Last Synced:** `{_last_synced}`")
        else:
            st.caption("🔴 **Never synced** — the Piotroski/DuPont Quality Gate has no data yet and will block all fresh BUY allocations until you sync.")

    # A prior sync's summary survives the st.rerun() below via session_state, so the user
    # actually gets to see the result instead of it being wiped by the immediate rerun.
    _last_sync_summary = st.session_state.get('last_screener_sync_summary')
    if _last_sync_summary:
        st.success(_last_sync_summary['message'])
        if _last_sync_summary.get('failed_rows'):
            with st.expander(f"⛔ {len(_last_sync_summary['failed_rows'])} tickers failed to sync last run — click to see why"):
                st.dataframe(pd.DataFrame(_last_sync_summary['failed_rows']), hide_index=True, use_container_width=True)
        if _last_sync_summary.get('mostly_failed'):
            st.error(
                "🔴 **Over half of all tickers failed to sync.** This almost certainly means the "
                "scraper's assumptions about Screener.in's page layout are wrong (site redesign, "
                "anti-bot page, or a changed URL structure) rather than 200 individual failures -- "
                "do not trust any 'Screener.in (Audited)' rows from this run until that's fixed."
            )
        del st.session_state['last_screener_sync_summary']  # show it exactly once

    if sync_clicked:
        # The full Nifty 200 + multi-asset candidate universe -- NOT `tickers` (the optimizer's
        # already-selected top-N), since syncing must cover the whole pool the screen picks from.
        _full_universe = list(live_sector_map.keys()) if live_sector_map else list(multifactor_scorecard_df['Ticker']) if not multifactor_scorecard_df.empty else tickers
        sync_universe = list(dict.fromkeys(list(_full_universe) + owned_tickers + [SOVEREIGN_BOND_TICKER]))
        sync_progress = st.progress(0, text="Starting Screener.in sync...")

        def _on_sync_progress(done: int, total: int, current_ticker: str) -> None:
            sync_progress.progress(done / total, text=f"Syncing {current_ticker} ({done}/{total})...")

        with st.spinner("Fetching audited filings from Screener.in — this is deliberately rate-limited and will take several minutes for the full universe..."):
            synced_df = sync_screener_fundamentals_for_universe(
                sync_universe, sector_map=SECTOR_MAP, progress_callback=_on_sync_progress
            )
        sync_progress.progress(1.0, text="Sync complete.")

        if not synced_df.empty:
            n_ok = int((synced_df['Data Source'] == 'Screener.in (Audited)').sum())
            n_na_fin = int((synced_df['Data Source'] == 'Not Applicable (Financial Institution)').sum())
            n_failed = int((synced_df['Data Source'] == 'Sync Failed').sum())
            failed_rows = (
                synced_df[synced_df['Data Source'] == 'Sync Failed'][['Ticker', 'sync_error']].to_dict('records')
                if n_failed > 0 else []
            )
            st.session_state['last_screener_sync_summary'] = {
                'message': (
                    f"✅ Synced {len(synced_df)} tickers: **{n_ok} audited**, **{n_na_fin} financial institutions "
                    f"(F-Score N/A)**, **{n_failed} failed**."
                ),
                'failed_rows': failed_rows,
                'mostly_failed': n_failed > len(synced_df) * 0.5,
            }
        st.cache_data.clear()
        st.rerun()

    # -------------------------------------------------------------------------
    # 1. GOVERNANCE & RED-FLAG CIRCUIT BREAKER ALERT BANNER
    # -------------------------------------------------------------------------
    red_flags = pd.DataFrame()
    if not multifactor_scorecard_df.empty:
        if 'Governance' in multifactor_scorecard_df.columns:
            gov_mask = multifactor_scorecard_df['Governance'].astype(str).str.contains('RED FLAG', na=False)
        elif 'Governance_Flag' in multifactor_scorecard_df.columns:
            gov_mask = multifactor_scorecard_df['Governance_Flag'] == 'RED_FLAG'
        else:
            gov_mask = pd.Series(False, index=multifactor_scorecard_df.index)
        
        sent_mask = (multifactor_scorecard_df['Sentiment_Raw'] < -0.30) if 'Sentiment_Raw' in multifactor_scorecard_df.columns else pd.Series(False, index=multifactor_scorecard_df.index)
        red_flags = multifactor_scorecard_df[gov_mask | sent_mask]

    if not red_flags.empty:
        flagged_names = ", ".join(red_flags['Asset'].tolist())
        st.error(
            f"🚨 **News Circuit Breaker Active:** {len(red_flags)} constituent(s) flagged for negative governance / adverse news catalysts (**{flagged_names}**). Fresh capital allocations automatically blocked."
        )
        with st.expander("🔍 View Red-Flagged Constituent Details & News Triggers", expanded=False):
            for _, rf_row in red_flags.iterrows():
                tk = rf_row['Ticker']
                as_name = rf_row['Asset']
                trig = news_triggers.get(tk, "Adverse news sentiment / forensic flag detected in live NLP feed.")
                sent_val = float(rf_row.get('Sentiment_Raw', 0.0))
                gov_tag = str(rf_row.get('Governance', '🚨 RED FLAG'))
                st.markdown(f"- **{as_name}** (`{tk}`): **Governance Flag:** `{gov_tag}` | **Sentiment Score:** `{sent_val:+.2f}` | **Trigger:** *{trig}*")
    else:
        st.success("🟢 **Governance Circuit Breaker Clean:** All 200 Nifty constituents passed NLP sentiment & forensic governance screens with zero active red flags.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. INTERACTIVE 4-QUADRANT FACTOR SCATTER MAP (PLOTLY)
    # -------------------------------------------------------------------------
    st.markdown("#### 1. Interactive 4-Quadrant Factor Matrix: Quality vs. Momentum")
    st.caption("Plots constituent DuPont ROE against 12M-1M Momentum with bubble size proportional to 90-Day ADTV (Liquidity). Dashed lines denote universe median thresholds.")

    if not multifactor_scorecard_df.empty and 'Momentum_Raw' in multifactor_scorecard_df.columns:
        scatter_df = multifactor_scorecard_df.copy()
        scatter_df['Momentum_Pct'] = pd.to_numeric(scatter_df['Momentum_Raw'], errors='coerce') * 100.0
        
        # Ensure robust numeric ROE extraction for complete horizontal dispersion across all 4 quadrants
        if 'roe_num' in scatter_df.columns:
            scatter_df['ROE_Pct'] = pd.to_numeric(scatter_df['roe_num'], errors='coerce')
        elif 'ROE_Clean' in scatter_df.columns:
            scatter_df['ROE_Pct'] = pd.to_numeric(scatter_df['ROE_Clean'], errors='coerce')
        else:
            scatter_df['ROE_Pct'] = scatter_df['DuPont ROE (%)'].astype(str).str.rstrip('%').astype(float)
            
        scatter_df['ROE_Pct'] = scatter_df['ROE_Pct'].fillna(16.0)
        scatter_df['ADTV_Cr'] = pd.to_numeric(scatter_df.get('ADTV_90d_Cr', 10.0), errors='coerce').fillna(10.0)
        scatter_df['Bubble_Size'] = np.clip(np.sqrt(scatter_df['ADTV_Cr']) * 3.5, 8, 30)
        
        med_roe = float(scatter_df['ROE_Pct'].median())
        med_mom = float(scatter_df['Momentum_Pct'].median())

        fig_quad = go.Figure()

        # Categorize by selection status
        status_groups = [
            (f'🟢 Selected (Top {TOP_N_SELECTED_EQUITIES})', '#00C853', 12, 'circle'),
            ('⚪ Liquid Candidate', '#00B0FF', 8, 'circle'),
            ('⛔ F-Score Deteriorating', '#FF1744', 9, 'diamond'),
            ('⛔ Illiquid Gate Blocked', '#78909C', 6, 'x')
        ]

        for status_label, color, base_sz, marker_sym in status_groups:
            sub_pts = scatter_df[scatter_df['Selection_Status'].str.startswith(status_label[:3])]
            if not sub_pts.empty:
                is_top = 'Top' in status_label
                fig_quad.add_trace(go.Scatter(
                    x=sub_pts['ROE_Pct'],
                    y=sub_pts['Momentum_Pct'],
                    mode='markers',
                    marker=dict(
                        size=sub_pts['Bubble_Size'],
                        color=color,
                        opacity=0.90 if is_top else 0.55,
                        symbol=marker_sym,
                        line=dict(width=1.5 if is_top else 0.5, color='#FFFFFF' if is_top else 'rgba(255,255,255,0.3)')
                    ),
                    name=status_label,
                    customdata=np.stack((
                        sub_pts['Asset'],
                        sub_pts['Ticker'],
                        sub_pts['ADTV_Cr'],
                        sub_pts.get('Accumulation_Ratio', sub_pts.get('Vol_Accum_Ratio', pd.Series(1.0, index=sub_pts.index))),
                        sub_pts.get('Institutional Inflow', sub_pts.get('Institutional_Inflow_Badge', pd.Series('⚪ Neutral Flow', index=sub_pts.index))),
                        sub_pts['Selection_Status'],
                        sub_pts.get('Piotroski_Score', sub_pts.get('Piotroski F-Score', pd.Series('7/9', index=sub_pts.index)))
                    ), axis=-1)
                ))

        # Add quadrant median reference lines
        fig_quad.add_vline(x=med_roe, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1.5)
        fig_quad.add_hline(y=med_mom, line_dash="dash", line_color="rgba(255,255,255,0.4)", line_width=1.5)

        # -------------------------------------------------------------------------
        # HIGH-CONTRAST 4-QUADRANT BADGES (Bold, High-Legibility Dark Text)
        # -------------------------------------------------------------------------

        # 1. Top-Right: Optimal Compounders (High ROE, High Momentum)
        fig_quad.add_annotation(
            x=0.98, y=0.98, xref="paper", yref="paper",
            text="<b>★ OPTIMAL COMPOUNDERS</b><br><span style='font-size:10px;'>High Quality + High Momentum</span>",
            showarrow=False,
            align="center",
            font=dict(size=12, color="#064e3b", family="Arial Black, sans-serif"),
            bgcolor="rgba(220, 252, 231, 0.95)", # High-contrast light emerald
            bordercolor="#16a34a",
            borderwidth=2,
            borderpad=6
        )

        # 2. Top-Left: Speculative Momentum (Low ROE, High Momentum)
        fig_quad.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text="<b>⚡ SPECULATIVE MOMENTUM</b><br><span style='font-size:10px;'>High Momentum, Lower Quality</span>",
            showarrow=False,
            align="center",
            font=dict(size=12, color="#78350f", family="Arial Black, sans-serif"),
            bgcolor="rgba(254, 243, 199, 0.95)", # High-contrast light amber
            bordercolor="#d97706",
            borderwidth=2,
            borderpad=6
        )

        # 3. Bottom-Right: Quality / Value Traps (High ROE, Lagging Momentum)
        fig_quad.add_annotation(
            x=0.98, y=0.02, xref="paper", yref="paper",
            text="<b>💎 QUALITY / VALUE TRAPS</b><br><span style='font-size:10px;'>High Quality, Lagging Momentum</span>",
            showarrow=False,
            align="center",
            font=dict(size=12, color="#1e3a8a", family="Arial Black, sans-serif"),
            bgcolor="rgba(219, 234, 254, 0.95)", # High-contrast light blue
            bordercolor="#2563eb",
            borderwidth=2,
            borderpad=6
        )

        # 4. Bottom-Left: Avoid / Deteriorating (Low ROE, Low Momentum)
        fig_quad.add_annotation(
            x=0.02, y=0.02, xref="paper", yref="paper",
            text="<b>⚠️ AVOID / DETERIORATING</b><br><span style='font-size:10px;'>Low Quality + Low Momentum</span>",
            showarrow=False,
            align="center",
            font=dict(size=12, color="#7f1d1d", family="Arial Black, sans-serif"),
            bgcolor="rgba(254, 226, 226, 0.95)", # High-contrast light red
            bordercolor="#dc2626",
            borderwidth=2,
            borderpad=6
        )

        # Improve Hover Tooltip & Remove Cluttered In-Chart Text
        fig_quad.update_traces(
            hovertemplate="<b>%{customdata[0]}</b> (%{customdata[1]})<br>• <b>Piotroski F-Score:</b> %{customdata[6]}<br>• <b>Institutional Inflow:</b> %{customdata[4]} (%{customdata[3]}x Vol)<br>• <b>DuPont ROE:</b> %{x:.1f}%<br>• <b>12M-1M Momentum:</b> %{y:+.1f}%<br>• <b>90D ADTV:</b> ₹%{customdata[2]:.1f} Cr<br>• <b>Status:</b> %{customdata[5]}<extra></extra>"
        )

        fig_quad.update_layout(
            title=dict(text="Constituent 4-Quadrant Factor Matrix (Quality ROE vs. 12M Momentum)", x=0.5),
            xaxis=dict(title="Quality Factor: DuPont ROE (%)", range=[0, 55], gridcolor='rgba(255,255,255,0.08)'),
            yaxis=dict(title="Momentum Factor: 12M-1M Momentum (%)", range=[-60, 130], gridcolor='rgba(255,255,255,0.08)'),
            height=600,
            margin=dict(l=20, r=20, t=50, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_quad, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 3. SINGLE-STOCK DUPONT 3-STAGE INTERACTIVE DRILLDOWN
    # -------------------------------------------------------------------------
    st.markdown("#### 2. Single-Stock DuPont 3-Stage Factor Decomposition & Piotroski Health")
    st.caption("Deconstructs Return on Equity (ROE) into operating efficiency, asset productivity, and capital structure leverage, paired with the 9-point Piotroski F-Score.")

    if not multifactor_scorecard_df.empty:
        stock_options = multifactor_scorecard_df['Asset'].tolist()
        col_sel, _ = st.columns([2, 1])
        with col_sel:
            selected_asset = st.selectbox(
                "🔍 Select Constituent for 3-Stage DuPont ROE & Piotroski Deep-Dive:",
                options=stock_options,
                index=0,
                help="Select any Nifty 200 constituent to inspect its DuPont operating margin, asset turnover, leverage, and Piotroski F-Score."
            )

        selected_row = multifactor_scorecard_df[multifactor_scorecard_df['Asset'] == selected_asset].iloc[0]
        sel_ticker = selected_row['Ticker']

        # Extract DuPont & Piotroski metrics from universe_fundamentals or scorecard. None/NaN
        # is preserved as None throughout -- never silently defaulted to a plausible-looking
        # number -- since this panel exists specifically to let you audit what's driving the
        # buy/block decision.
        def _num_or_none(v):
            return float(v) if pd.notna(v) else None

        npm_val = _num_or_none(selected_row.get('NPM_Clean'))
        de_val = _num_or_none(selected_row.get('DE_Clean'))
        roe_val = _num_or_none(selected_row.get('ROE_Clean'))
        accum_ratio_val = _num_or_none(selected_row.get('Accumulation_Ratio', selected_row.get('Vol_Accum_Ratio', 1.0))) or 1.0
        inflow_badge_val = str(selected_row.get('Institutional Inflow', selected_row.get('Institutional_Inflow_Badge', '⚪ Neutral Flow')))

        fund_row = universe_fundamentals[universe_fundamentals['Ticker'] == sel_ticker].iloc[0] if (universe_fundamentals is not None and not universe_fundamentals.empty and sel_ticker in universe_fundamentals['Ticker'].values) else None
        source_row = fund_row if fund_row is not None else selected_row

        turnover_val = _num_or_none(source_row.get('turnover_num'))
        leverage_val = _num_or_none(source_row.get('leverage_num'))
        pe_val = _num_or_none(source_row.get('pe_num'))
        health_val = str(source_row.get('Fundamental Health', '⚪ Not Yet Synced'))
        f_score_raw = source_row.get('piotroski_f_score', source_row.get('f_score_num', source_row.get('Piotroski_F_Score')))
        f_score_val = None if pd.isna(f_score_raw) else int(f_score_raw)
        f_badge_val = str(source_row.get('piotroski_badge', source_row.get('f_score_label', source_row.get('F_Score_Label', '⚪ Not Yet Synced'))))
        f_disp_val = str(source_row.get('Piotroski F-Score', source_row.get('f_score_display', 'N/A')))
        data_source_val = str(source_row.get('Data Source', source_row.get('Fundamentals Source', 'Not Yet Synced')))

        with st.container(border=True):
            st.markdown(f"##### 🏢 **{selected_asset}** (`{sel_ticker}`) — DuPont 3-Stage & Piotroski Breakdown")
            if data_source_val == 'Screener.in (Audited)':
                st.caption("📌 **Fundamentals Source: Screener.in (Audited)** — extracted from this company's own audited P&L/Balance Sheet/Cash Flow tables. This scraper is unverified against a live page (see the Tab 3 banner above) -- spot-check these numbers in your browser before trusting them.")
            elif data_source_val == 'Not Applicable (Financial Institution)':
                st.caption("🏦 **Fundamentals Source: Not Applicable (Financial Institution)** — the 9-point Piotroski F-Score doesn't transfer to bank/NBFC statement structures, so it's intentionally not computed here. DuPont ROE/D-E above remain real, audited figures.")
            elif data_source_val == 'Sync Failed':
                st.caption("⛔ **Fundamentals Source: Sync Failed** — the scraper could not parse this ticker's Screener.in page. Treat every number below as unavailable, not as a passing score.")
            else:
                st.caption("⚪ **Fundamentals Source: Not Yet Synced** — click '🔄 Sync Audited Filings' above to fetch this ticker's real financials. Fresh capital allocation is blocked until then.")
            st.latex(r"\text{DuPont ROE} = \underbrace{\left(\frac{\text{Net Income}}{\text{Revenue}}\right)}_{\text{Net Profit Margin}} \times \underbrace{\left(\frac{\text{Revenue}}{\text{Total Assets}}\right)}_{\text{Asset Turnover}} \times \underbrace{\left(\frac{\text{Total Assets}}{\text{Total Equity}}\right)}_{\text{Financial Leverage}}")

            d_c1, d_c2, d_c3, d_c4, d_c5, d_c6 = st.columns(6)
            with d_c1:
                st.metric(
                    "1. Net Margin (%)",
                    f"{npm_val:.1f}%" if npm_val is not None else "N/A",
                    help="Operating Efficiency: Percentage of revenue converted into bottom-line net profit."
                )
                st.caption("*(Operating Efficiency)*")
            with d_c2:
                st.metric(
                    "2. Asset Turnover (x)",
                    f"{turnover_val:.2f}x" if turnover_val is not None else "N/A",
                    help="Asset Utilization: Revenue generated per unit of total assets deployed."
                )
                st.caption("*(Asset Utilization)*")
            with d_c3:
                st.metric(
                    "3. Financial Leverage",
                    f"{leverage_val:.2f}x" if leverage_val is not None else "N/A",
                    f"D/E: {de_val:.2f}x" if de_val is not None else "D/E: N/A",
                    delta_color="off",
                    help="Capital Structure: Total assets divided by shareholders' equity."
                )
                st.caption("*(Solvency / Leverage)*")
            with d_c4:
                st.metric(
                    "= 3-Stage DuPont ROE",
                    f"{roe_val:.1f}%" if roe_val is not None else "N/A",
                    f"Health: {health_val}",
                    delta_color="normal" if (roe_val is not None and roe_val > 15.0) else "inverse",
                    help="Calculated DuPont Return on Equity."
                )
                st.caption("*(Total Return on Equity)*")
            with d_c5:
                st.metric(
                    "Piotroski F-Score",
                    f_disp_val,
                    f_badge_val,
                    delta_color="normal" if (f_score_val is not None and f_score_val >= 8) else ("inverse" if (f_score_val is not None and f_score_val <= 4) else "off"),
                    help="9-Point Discrete Fundamental Health Score (0-9): Profitability, Solvency, Operating Efficiency."
                )
                st.caption("*(Balance Sheet Health)*")
            with d_c6:
                st.metric(
                    "Institutional Inflow",
                    f"{accum_ratio_val:.2f}x",
                    inflow_badge_val,
                    delta_color="normal" if accum_ratio_val >= 1.3 else "off",
                    help="5-Day Mean Volume / 90-Day Median Volume (Institutional Accumulation Z_accum Factor)."
                )
                st.caption("*(Volume Accumulation)*")

            # Contextual diagnosis
            if accum_ratio_val >= 2.0:
                st.success(f"🔥 **Heavy Institutional Accumulation ({accum_ratio_val:.2f}x Volume):** {selected_asset} is experiencing aggressive institutional capital inflows driven by structural catalysts.")
            elif accum_ratio_val >= 1.3:
                st.info(f"🟢 **Steady Institutional Inflows ({accum_ratio_val:.2f}x Volume):** {selected_asset} exhibits consistent buying demand above its 90-day baseline.")

            if f_score_val is None:
                st.warning(f"⚪ **Piotroski F-Score Unavailable:** {f_badge_val}. Fresh capital allocation is blocked for {selected_asset} until real audited data is available (sync, or this is a Financial Institution the score doesn't apply to).")
            elif f_score_val >= 8:
                st.success(f"🟢 **Piotroski F-Score ({f_score_val}/9 — Strong):** {selected_asset} exhibits pristine financial strength across profitability, positive cash flows (CFO > Net Income), improving asset efficiency, and zero share dilution.")
            elif f_score_val <= 3:
                st.error(f"⛔ **Piotroski F-Score ({f_score_val}/9 — Decaying / Distressed):** {selected_asset} shows deteriorating balance sheet quality, weak accruals, or elevated leverage. Fresh capital allocation blocked by Quality Gate.")
            elif f_score_val <= 4:
                st.warning(f"⚠️ **Piotroski F-Score ({f_score_val}/9 — Low Quality):** {selected_asset} demonstrates sub-optimal earnings accruals or compressed operating margins.")
            else:
                st.info(f"🔵 **Piotroski F-Score ({f_score_val}/9 — Moderate Stability):** {selected_asset} maintains stable financial operations with sound solvency.")

            if leverage_val is not None and leverage_val > 3.0:
                st.warning(f"⚠️ **High Financial Leverage ({leverage_val:.2f}x):** {selected_asset}'s ROE is heavily boosted by debt rather than high operating profit margins.")
            elif npm_val is not None and npm_val > 20.0:
                st.info(f"💎 **Strong Pricing Power ({npm_val:.1f}% Margin):** {selected_asset} demonstrates a competitive moat with high net profitability.")
            elif turnover_val is not None and turnover_val > 1.2:
                st.info(f"⚡ **High Asset Velocity ({turnover_val:.2f}x Turnover):** {selected_asset} operates with lean, highly productive asset turnover.")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 4. STREAMLIT COLUMN_CONFIG UPGRADE ON MASTER SCORECARD TABLE
    # -------------------------------------------------------------------------
    st.markdown("#### 3. Full Multi-Factor Ranking Scorecard (All 200 Constituents)")
    st.caption("Composite Multi-Factor Score: `35% Quality (DuPont ROE & Solvency & Piotroski F-Score) + 35% 12M-1M Momentum + 15% Volatility Penalty + 15% Institutional Volume Accumulation (Z_accum)`.")

    if not multifactor_scorecard_df.empty:
        scorecard_display_df = multifactor_scorecard_df.copy()
        scorecard_display_df['Momentum_Pct'] = scorecard_display_df['Momentum_Raw'] * 100.0
        scorecard_display_df['Vol_60d_Pct'] = scorecard_display_df['Vol_60d_Raw'] * 100.0

        table_cols = [
            'Rank', 'Asset', 'Selection_Status', 'ADTV_90d_Cr', 'Institutional Inflow', 'Accumulation_Ratio',
            'Piotroski_Score', 'ROE_Clean', 'Momentum_Pct', 'DE_Clean', 'NPM_Clean', 'Vol_60d_Pct',
            'Governance', 'Fundamental Health', 'Fundamentals Source'
        ]
        valid_table_cols = [c for c in table_cols if c in scorecard_display_df.columns]

        st.dataframe(
            scorecard_display_df[valid_table_cols],
            column_config={
                "Rank": st.column_config.NumberColumn("Rank", format="#%d", width="small"),
                "Asset": st.column_config.TextColumn("Asset", width="medium"),
                "Selection_Status": st.column_config.TextColumn("Status", width="medium"),
                "ADTV_90d_Cr": st.column_config.NumberColumn("ADTV 90D", format="₹%.1f Cr", width="small"),
                "Institutional Inflow": st.column_config.TextColumn(
                    "Institutional Inflow",
                    help="Volume accumulation status: 🔥 Heavy Accumulation (≥2x Volume), 🟢 Steady Inflows (≥1.3x Volume), ⚪ Neutral Flow",
                    width="medium"
                ),
                "Accumulation_Ratio": st.column_config.NumberColumn(
                    "Accum. Ratio",
                    format="%.2fx",
                    help="5-Day Mean Volume / 90-Day Median Volume (Z_accum Factor)",
                    width="small"
                ),
                "Piotroski_Score": st.column_config.TextColumn(
                    "Piotroski F-Score",
                    help="9-Point Discrete Fundamental Health Score (0-9): Profitability, Solvency, Operating Efficiency.",
                    width="small"
                ),
                "ROE_Clean": st.column_config.ProgressColumn(
                    "DuPont ROE (%)",
                    format="%.1f%%",
                    min_value=0.0,
                    max_value=40.0,
                    width="medium"
                ),
                "Momentum_Pct": st.column_config.ProgressColumn(
                    "12M-1M Momentum",
                    format="%+.1f%%",
                    min_value=-50.0,
                    max_value=100.0,
                    width="medium"
                ),
                "DE_Clean": st.column_config.NumberColumn("Debt/Equity", format="%.2fx", width="small"),
                "NPM_Clean": st.column_config.NumberColumn("Net Margin", format="%.1f%%", width="small"),
                "Vol_60d_Pct": st.column_config.NumberColumn("60D Realized Vol", format="%.1f%%", width="small"),
                "Governance": st.column_config.TextColumn("Governance Flag", width="small"),
                "Fundamental Health": st.column_config.TextColumn("Health Classification", width="medium"),
                "Fundamentals Source": st.column_config.TextColumn(
                    "Fundamentals Source",
                    help="'Screener.in (Audited)' = extracted from this company's own audited filings via "
                         "Sync (unverified scraper -- spot-check it). 'Not Applicable (Financial Institution)' "
                         "= a bank/NBFC the 9-point F-Score doesn't apply to; ROE/D-E are still real. "
                         "'Sync Failed' / 'Not Yet Synced' = no verified data at all; fresh BUYs are blocked.",
                    width="medium"
                ),
            },
            hide_index=True,
            use_container_width=True
        )

# --- TAB 4: TAX HARVESTING & CAPITAL GAINS CENTER ---
with tab_harvest:
    st.subheader("💎 Annual Tax-Harvesting & Capital Gains Center (Section 112A / 111A / 50AA & Budget 2024–2026)")
    st.caption("Enforces statutory Section 70 & 74 intra-head loss set-off rules: STCL offsets both STCG/LTCG; LTCL is strictly restricted to LTCG.")

    current_fy_year = TODAY.year if TODAY.month >= 4 else TODAY.year - 1
    current_fy_str = f"FY {current_fy_year}-{str(current_fy_year+1)[-2:]}"
    fy_start_dt = f"{current_fy_year}-04-01"
    fy_end_dt = f"{current_fy_year+1}-03-31"

    with get_db_connection() as conn:
        fy_sells_df = clean_trade_ledger_df(pd.read_sql("""
            SELECT * FROM trade_ledger 
            WHERE action = 'SELL' 
              AND execution_date >= ? 
              AND execution_date <= ?
            ORDER BY execution_date ASC
        """, conn, params=[fy_start_dt, fy_end_dt]))

    tax_summary = compute_realized_tax_summary(fy_sells_df)
    unrealized_tax_analysis = compute_unrealized_tax_lots_analysis(tax_lots_df, latest_prices_series)
    harvest_recommendations_df = recommend_tax_loss_harvesting_trades(tax_summary, unrealized_tax_analysis)

    # 5 KPI Metric Summary Cards
    cg_col1, cg_col2, cg_col3, cg_col4, cg_col5 = st.columns(5)
    with cg_col1:
        st.metric(
            f"Net Sec 112A LTCG ({current_fy_str})",
            f"₹{tax_summary['post_setoff_ltcg']:,.2f}",
            f"Tax @ 12.5%: ₹{tax_summary['tax_payable_112a']:,.2f}" if tax_summary['post_setoff_ltcg'] > 125000 else "100% Tax-Free (≤ ₹1.25L)"
        )
    with cg_col2:
        st.metric(
            f"Net Sec 111A STCG ({current_fy_str})",
            f"₹{tax_summary['post_setoff_stcg']:,.2f}",
            f"Tax @ 20.0%: ₹{tax_summary['tax_payable_111a']:,.2f}" if tax_summary['post_setoff_stcg'] > 0 else "No STCG Tax Liability"
        )
    with cg_col3:
        st.metric(
            f"Sec 50AA Debt STCG ({current_fy_str})",
            f"₹{tax_summary['net_50aa_stcg']:,.2f}",
            "Taxable @ Slab Rates"
        )
    with cg_col4:
        st.metric(
            "Net Realized Loss (Sec 70/74)",
            f"₹{tax_summary['total_realized_loss_for_setoff']:,.2f}",
            f"STCL: ₹{tax_summary['unabsorbed_stcl']:,.0f} | LTCL: ₹{tax_summary['unabsorbed_ltcl']:,.0f}"
        )
    with cg_col5:
        st.metric(
            "Unrealized Loss Harvestable",
            f"₹{unrealized_tax_analysis['total_harvestable_losses']:,.2f}",
            f"{len(unrealized_tax_analysis['loss_rows'])} Active Position(s)"
        )

    st.markdown("---")

    # Active Tax-Loss Neutralizer
    tax_payable_111a = float(tax_summary['tax_payable_111a'])
    tax_payable_112a = float(tax_summary['tax_payable_112a'])
    total_tax_liability = tax_payable_111a + tax_payable_112a

    if total_tax_liability > 0:
        if not harvest_recommendations_df.empty:
            st.warning(f"💡 **Tax Alpha Opportunity:** You have ₹{tax_payable_111a:,.2f} in STCG tax liabilities. The engine found eligible unrealized loss positions to reduce your tax bill to ₹0.00.")
            st.markdown("#### 🎯 Automated Active Tax-Loss Neutralizing Trades")
            st.dataframe(harvest_recommendations_df.style.format({
                'Live Price (₹)': '₹{:.2f}',
                'Estimated Realized Loss (₹)': '₹{:,.2f}',
                'Tax Savings (₹)': '₹{:,.2f}',
                'Post-Harvest Tax Liability': '₹{:,.2f}'
            }), use_container_width=True)
        else:
            st.warning(f"💡 **Tax Alpha Opportunity:** You have ₹{tax_payable_111a:,.2f} in STCG tax liabilities. (No active unrealized loss positions currently found to harvest).")
    else:
        st.success("✓ **Tax Optimal State:** Zero realized capital gains tax liability for the current financial year.")

    st.markdown("---")

    # Table 1: Qualified LTCG Harvesting
    st.markdown("#### 1. 💎 Qualified Section 112A LTCG Profits (Holding Period ≥ 365 Days)")
    if unrealized_tax_analysis['harvest_rows']:
        st.dataframe(pd.DataFrame(unrealized_tax_analysis['harvest_rows']), use_container_width=True)
        st.success(f"• **Total Available LTCG Profit:** ₹{unrealized_tax_analysis['total_harvestable_profit']:,.2f} / ₹1,25,000 Annual Tax-Free Quota")
    else:
        st.info("✓ Zero unrealized LTCG profits qualifying for tax-free Section 112A harvest today.")

    st.markdown("---")

    # Table 2: Tax-Loss Harvesting
    st.markdown("#### 2. 📉 Tax-Loss Harvesting Inspector (Offsetting STCG / LTCG Liabilities)")
    if unrealized_tax_analysis['loss_rows']:
        st.dataframe(pd.DataFrame(unrealized_tax_analysis['loss_rows']), use_container_width=True)
        st.warning(f"• **Total Harvestable Capital Losses:** ₹{unrealized_tax_analysis['total_harvestable_losses']:,.2f} available to neutralize capital gains tax.")
    else:
        st.info("✓ Zero positions with unrealized losses in your portfolio.")

# --- TAB 5: SQLITE AUDIT & TAX LEDGER ---
with tab_db:
    st.subheader(f"📂 Stored SQLite Tax Lots & Audit Ledger (`{DB_FILE}`)")
    st.caption("Inspect active tax lots, immutable trade ledger, and export Schedule 112A tax reports for ITR filing.")
    st.caption("🔒 Database state protected. Last automatic snapshot created in 'db_backups/'.")

    # Disaster Recovery & 1-Click Database Download Card
    with st.container(border=True):
        st.markdown("#### 🛡️ Database Disaster Recovery & 1-Click Backup")
        st.caption("Download a complete immutable snapshot of your portfolio database (`wealth_ledger.db`) for cold storage on Google Drive, iCloud, or external storage.")
        if os.path.exists(DB_FILE):
            with open(DB_FILE, "rb") as f:
                db_bytes = f.read()
            st.download_button(
                label="💾 Download SQLite Ledger (`wealth_ledger.db`) Backup",
                data=db_bytes,
                file_name=f"wealth_ledger_backup_{TODAY_STR}.db",
                mime="application/x-sqlite3",
                help="Download a complete snapshot of your portfolio database for backup on Google Drive / external drive."
            )
        else:
            st.warning("Database file not yet created on disk.")
        st.caption("*Tip: Download your database snapshot monthly to ensure your multi-year tax lots are permanently safe.*")

    with get_db_connection() as conn:
        all_db_lots = clean_tax_lots_df(pd.read_sql("""
            SELECT lot_id, ticker, asset_name, asset_class, buy_date, quantity, buy_price, last_split_date
            FROM tax_lots 
            WHERE quantity > 0 
              AND ticker IS NOT NULL 
              AND ticker != 'None' 
            ORDER BY buy_date DESC
        """, conn))

        all_trade_ledger = clean_trade_ledger_df(pd.read_sql("""
            SELECT trade_id, execution_date, ticker, action, quantity, price, realized_pnl, gain_type, fees_estimated
            FROM trade_ledger
            ORDER BY trade_id DESC
        """, conn))

    st.markdown("#### 1. Active Portfolio Tax Lots (`tax_lots`)")
    if all_db_lots.empty:
        st.info("No active tax lots currently recorded in database. Upload a CSV tradebook in the sidebar or execute orders to populate.")
    else:
        st.dataframe(all_db_lots, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 2. Immutable Execution Audit Trail (`trade_ledger`)")
    if all_trade_ledger.empty:
        st.info("Trade ledger is currently empty. Executed orders will generate immutable transaction logs.")
    else:
        st.dataframe(all_trade_ledger.style.format({
            'price': '₹{:.2f}',
            'realized_pnl': '₹{:,.2f}',
            'fees_estimated': '₹{:.2f}'
        }), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 3. 📑 Revenue Audit & Schedule 112A / Schedule CG Tax Export Engine")

    fy_options = [
        f"FY {current_fy_year}-{str(current_fy_year+1)[-2:]}",
        f"FY {current_fy_year-1}-{str(current_fy_year)[-2:]}",
        "All Financial Years"
    ]
    
    rep_col1, rep_col2 = st.columns([1, 2])
    with rep_col1:
        selected_fy = st.selectbox("Select Financial Year for Tax Report:", options=fy_options, index=0)

    if selected_fy == "All Financial Years":
        filtered_sells = all_trade_ledger[all_trade_ledger['action'] == 'SELL']
    else:
        sel_yr = int(selected_fy.split()[1].split('-')[0])
        s_date = f"{sel_yr}-04-01"
        e_date = f"{sel_yr+1}-03-31"
        filtered_sells = all_trade_ledger[(all_trade_ledger['action'] == 'SELL') & (all_trade_ledger['execution_date'] >= s_date) & (all_trade_ledger['execution_date'] <= e_date)]

    schedule_df = build_schedule_112a_records(filtered_sells)
    if schedule_df.empty:
        st.info(f"No sell transactions recorded for **{selected_fy}**.")
    else:
        csv_data = schedule_df.to_csv(index=False).encode('utf-8')
        with rep_col2:
            st.write("")
            st.write("")
            st.download_button(
                label=f"📥 Download Realized Tax Report ({selected_fy} CSV)",
                data=csv_data,
                file_name=f"ITR_Schedule_112A_Tax_Report_{selected_fy.replace(' ', '_')}.csv",
                mime="text/csv",
                type="primary"
            )
        sched_format_dict = {
            'Sale Price (₹)': '₹{:.2f}',
            'Cost of Acquisition (₹)': '₹{:.2f}',
            'Total Sale Consideration': '₹{:,.2f}',
            'Total Cost': '₹{:,.2f}',
            'Deductions (Eligible Transfer Expenses)': '₹{:.2f}',
            'Non-Deductible STT (Sec 48)': '₹{:.2f}',
            'Total Fees Paid': '₹{:.2f}',
            'Realized Capital Gain / Loss': '₹{:,.2f}'
        }
        active_sched_format = {k: v for k, v in sched_format_dict.items() if k in schedule_df.columns}
        st.dataframe(schedule_df.style.format(active_sched_format), use_container_width=True)

    st.markdown("---")
    st.markdown("#### 4. 🔄 Corporate Actions & Stock Split Synchronization")
    st.caption("Scan active Demat tax lots against historical corporate actions, bonus issues, and stock splits via yfinance.")
    
    split_col1, split_col2 = st.columns([2, 1])
    with split_col1:
        st.markdown("Historical splits automatically compound holdings and adjust acquisition prices with FIFO integrity.")
    with split_col2:
        if st.button("🔄 Check & Reconcile Stock Splits", key="btn_reconcile_splits", type="secondary"):
            with st.spinner("Checking corporate actions via yfinance..."):
                split_msgs = reconcile_corporate_actions_and_splits()
            if split_msgs:
                for msg in split_msgs:
                    st.warning(msg)
                st.success("✓ Corporate actions reconciliation complete. Quantities and cost bases adjusted in SQLite.")
                st.cache_data.clear()
                st.rerun()
            else:
                st.success("✓ All active Demat tax lots are fully synchronized. Zero unadjusted stock splits detected.")

    st.markdown("---")
    st.markdown("##### 🗑️ Manual Tax Lot Removal")
    del_col1, del_col2 = st.columns([2, 1])
    with del_col1:
        lot_id_to_delete = st.number_input(
            "Enter `lot_id` to delete:",
            min_value=1,
            step=1,
            value=int(all_db_lots['lot_id'].iloc[0]) if not all_db_lots.empty else 1
        )
    with del_col2:
        st.write("")
        st.write("")
        if st.button("🗑️ Delete Tax Lot", type="secondary"):
            if not all_db_lots.empty:
                backup_database()
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM tax_lots WHERE lot_id = ?", [int(lot_id_to_delete)])
                    conn.commit()
                st.success(f"Tax lot #{lot_id_to_delete} deleted successfully!")
                st.cache_data.clear()
                st.rerun()