# ====================================================================================================
# MASTER QUANTITATIVE WEALTH OPERATING SYSTEM - CONFIGURATION MODULE
# Static Parameters, Universal Friction Caps, and Dynamic Time Horizons
# ====================================================================================================

import os
import pandas as pd

# ----------------------------------------------------------------------------------------------------
# 1. DATABASE, STATUTORY FEES & FRICTION CONSTANTS
# ----------------------------------------------------------------------------------------------------
MARKET_TIMEZONE             = 'Asia/Kolkata'

# Database path with environment override support
DB_FILE                     = os.getenv('WEALTH_DB_PATH', 'wealth_ledger.db')

# Risk-Free Rate with environment override support (6.5% Indian Sovereign 91-Day T-Bills)
try:
    RISK_FREE_RATE          = float(os.getenv('RISK_FREE_RATE', '0.065'))
except (ValueError, TypeError):
    RISK_FREE_RATE          = 0.065

MAX_RETAIL_CAP              = 0.20           # 20.0% Single-Asset Conviction Cap
MAX_SECTOR_CAP              = 0.25           # 25.0% Maximum Sector Concentration Cap
MIN_DISPOSAL_VALUE          = 3000.0         # ₹3,000 DP Fee Shield Floor for Trims
MIN_ADTV_INR                = 100_000_000.0  # ₹10.0 Crore (100M INR) 90-Day Median ADTV Liquidity Gate
TOP_N_SELECTED_EQUITIES     = 20             # Top 20 Factor-Ranked Equities Fed to Optimizer

# Granular Indian Exchange Statutory Fees & Friction Parameters
EQUITY_DELIVERY_FRICTION    = 0.0015         # ~0.15% (0.1% STT + Stamp Duty + GST + Exch Txn)
ETF_DEBT_GOLD_FRICTION      = 0.0003         # ~0.03% (0.0% STT for Gold/Debt ETFs + Stamp Duty + GST)
DP_CHARGE_FLAT_INR          = 15.93          # Standard CDSL/Broker DP charge per sell transaction
STATUTORY_FEE_BUFFER        = 1.0025         # 0.25% Friction Buffer (STT 0.1% + GST + Stamp Duty + Brokerage)
HEALTH_AND_EDUCATION_CESS   = 0.04           # 4.0% Statutory Health & Education Cess (Finance Act)

# Universal Sovereign Fixed Income Anchor & Benchmark
SOVEREIGN_BOND_TICKER       = 'GILT5YBEES.NS' # 5-Year Sovereign G-Sec ETF (Sec 50AA)
BENCHMARK_TICKER            = '^NSEI'

# ----------------------------------------------------------------------------------------------------
# 2. DYNAMIC TIME HORIZON RESOLVER
# ----------------------------------------------------------------------------------------------------
def get_market_time_horizons():
    """
    Dynamically resolves real-time date anchors and backtest horizons relative to live IST execution.
    Eliminates module-level frozen timestamps and server timezone drift.
    """
    today = pd.Timestamp.now(tz=MARKET_TIMEZONE).normalize().tz_localize(None)
    oos_split = today - pd.DateOffset(years=1, months=6)
    bt_train_start = oos_split - pd.DateOffset(years=6)
    live_train_start = today - pd.DateOffset(years=6)
    return {
        'TODAY': today,
        'TODAY_STR': today.strftime('%Y-%m-%d'),
        'TODAY_ISO': today.strftime('%Y-%m-%d'),
        'OOS_SPLIT_STR': oos_split.strftime('%Y-%m-%d'),
        'BACKTEST_TRAIN_START_STR': bt_train_start.strftime('%Y-%m-%d'),
        'LIVE_TRAIN_START_STR': live_train_start.strftime('%Y-%m-%d')
    }

def __getattr__(name):
    """
    Dynamically resolves date constants at access time to prevent frozen timestamps.
    Supports backwards compatibility for direct imports: TODAY, TODAY_STR, TODAY_ISO, etc.
    """
    horizons = get_market_time_horizons()
    if name in horizons:
        return horizons[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

def __dir__():
    """
    Enables IDEs, linters, and runtime reflection to discover dynamic date attributes.
    """
    dynamic_attrs = list(get_market_time_horizons().keys())
    return sorted(list(globals().keys()) + dynamic_attrs)
