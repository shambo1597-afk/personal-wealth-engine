# ====================================================================================================
# MASTER QUANTITATIVE WEALTH OPERATING SYSTEM - TAX ENGINE MODULE
# Section 112A LTCG, Section 111A STCG, Section 50AA Debt, and Section 70/74 Set-off Mechanics
# ====================================================================================================

from typing import Any, Dict, Optional, Union

import pandas as pd
import numpy as np
from config import SOVEREIGN_BOND_TICKER, get_market_time_horizons, HEALTH_AND_EDUCATION_CESS

# ----------------------------------------------------------------------------------------------------
# 1. STATUTORY TAX CONSTANTS & THRESHOLDS (BUDGET 2024–2026 / FINANCE ACT)
# ----------------------------------------------------------------------------------------------------
SEC_112A_BASE_RATE        = 0.125      # 12.5% Statutory Base Long-Term Capital Gains Rate (Sec 112A)
SEC_111A_BASE_RATE        = 0.200      # 20.0% Statutory Base Short-Term Capital Gains Rate (Sec 111A)
SEC_50AA_BASE_RATE        = 0.300      # 30.0% Statutory Slab Rate for Specified Debt Funds (Sec 50AA)

# Effective Tax Rates Inclusive of 4% Health & Education Cess
SEC_112A_TAX_RATE         = SEC_112A_BASE_RATE * (1.0 + HEALTH_AND_EDUCATION_CESS)  # 13.0% (0.130)
SEC_111A_TAX_RATE         = SEC_111A_BASE_RATE * (1.0 + HEALTH_AND_EDUCATION_CESS)  # 20.8% (0.208)
SEC_50AA_TAX_RATE         = SEC_50AA_BASE_RATE * (1.0 + HEALTH_AND_EDUCATION_CESS)  # 31.2% (0.312)
SEC_112A_EXEMPTION_LIMIT  = 125000.0   # ₹1,25,000 Annual Tax-Free LTCG Exemption (Sec 112A)

# ----------------------------------------------------------------------------------------------------
# 1.1 SECTION 112A / SECTION 55(2)(ac) PRE-2018 GRANDFATHERING ENGINE
# ----------------------------------------------------------------------------------------------------
def compute_grandfathered_cost_basis(
    buy_date: Any,
    buy_price: float,
    sale_or_live_price: float,
    fmv_31jan2018: Optional[float] = None,
) -> float:
    """
    Computes the deemed cost of acquisition under Section 55(2)(ac) of the Income Tax Act, 1961:
    For equity shares and equity-oriented mutual funds acquired before 1st February 2018 (buy_date < '2018-02-01'):
      - Step 1: Lower of FMV on 31-Jan-2018 and Sale / Live Price = min(FMV_31Jan2018, Sale_Price)
      - Step 2: Higher of Actual Buy Price and Step 1 value = max(Buy_Price, min(FMV_31Jan2018, Sale_Price))
    If FMV on 31-Jan-2018 is not available, missing, or <= 0, falls back to actual buy_price.
    For assets acquired on or after 1st February 2018, returns actual buy_price.
    """
    try:
        if buy_date and pd.notna(buy_date):
            b_str = str(buy_date).strip()
            if len(b_str) >= 10:
                b_date_prefix = b_str[:10]
                if b_date_prefix < '2018-02-01':
                    if fmv_31jan2018 is not None and pd.notna(fmv_31jan2018):
                        fmv = float(fmv_31jan2018)
                        if fmv > 0:
                            p_buy = float(buy_price)
                            p_sale = float(sale_or_live_price)
                            return max(p_buy, min(fmv, p_sale))
    except Exception:
        pass
    return float(buy_price)

# ----------------------------------------------------------------------------------------------------
# 2. REALIZED CAPITAL GAINS & SECTION 70/74 INTRA-HEAD SET-OFF ENGINE
# ----------------------------------------------------------------------------------------------------
def compute_realized_tax_summary(fy_sells_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Computes realized capital gains and enforces statutory Section 70 & 74 intra-head loss set-off rules:
      - Applies Section 55(2)(ac) pre-2018 grandfathering for eligible Section 112A LTCG trades.
      - Combines equity STCL and Section 50AA Debt STCL under Section 70(2).
      - Prioritizes setting off STCL against Section 50AA Debt STCG first (highest slab rate drag).
      - Sets off STCL against Section 111A Equity STCG.
      - Sets off STCL against LTCG ONLY to the extent LTCG exceeds the ₹1,25,000 Section 112A exemption,
        preventing wasteful burning of STCL against already-exempt gains.
      - Long-Term Capital Losses (LTCL) can ONLY set off LTCG (never STCG or Debt).
      - Section 50AA Debt ETF gains are deemed STCG and taxed at slab rates.
      - Section 112A LTCG is exempt up to ₹1,25,000 and taxed at effective 13.0% (12.5% + 4% Cess) on excess.
      - Section 111A STCG is taxed at effective 20.8% (20.0% + 4% Cess).
    """
    if fy_sells_df is None or fy_sells_df.empty:
        return {
            'gross_112a_pnl': 0.0,
            'gross_111a_pnl': 0.0,
            'net_50aa_stcg': 0.0,
            'post_setoff_ltcg': 0.0,
            'post_setoff_stcg': 0.0,
            'post_setoff_50aa': 0.0,
            'stcl_used_against_ltcg': 0.0,
            'stcl_used_against_50aa': 0.0,
            'unabsorbed_stcl': 0.0,
            'unabsorbed_ltcl': 0.0,
            'total_realized_loss_for_setoff': 0.0,
            'taxable_112a_amt': 0.0,
            'tax_payable_112a': 0.0,
            'tax_payable_111a': 0.0
        }

    ltcg_trades = fy_sells_df[fy_sells_df['gain_type'] == 'LTCG']
    debt_50aa_trades = fy_sells_df[
        (fy_sells_df['gain_type'].str.contains('50AA', na=False)) | 
        (fy_sells_df['ticker'] == SOVEREIGN_BOND_TICKER)
    ]

    # Strictly exclude debt ETF trades from standard Section 111A equity STCG
    stcg_trades = fy_sells_df[
        (fy_sells_df['gain_type'] == 'STCG') & 
        (fy_sells_df['ticker'] != SOVEREIGN_BOND_TICKER) & 
        (~fy_sells_df['gain_type'].str.contains('50AA', na=False))
    ]

    ltcg_pnls = []
    for _, row in ltcg_trades.iterrows():
        pnl = float(row['realized_pnl'])
        b_date = row.get('buy_date')
        if b_date and pd.notna(b_date) and str(b_date).strip()[:10] < '2018-02-01':
            q = float(row['quantity'])
            p_sale = float(row['price'])
            fmv = row.get('fmv_31jan2018', row.get('fmv', None))
            p_buy = float(row['buy_price']) if ('buy_price' in row and pd.notna(row['buy_price'])) else ((q * p_sale - pnl) / q if q > 0 else 0.0)
            deemed_cost = compute_grandfathered_cost_basis(b_date, p_buy, p_sale, fmv)
            pnl = round((p_sale - deemed_cost) * q, 2)
        ltcg_pnls.append(pnl)

    gross_112a_pnl = float(sum(ltcg_pnls)) if ltcg_pnls else 0.0
    gross_111a_pnl = float(stcg_trades['realized_pnl'].sum()) if not stcg_trades.empty else 0.0
    net_50aa_stcg  = float(debt_50aa_trades['realized_pnl'].sum()) if not debt_50aa_trades.empty else 0.0

    stcl_used_against_ltcg = 0.0
    stcl_used_against_50aa = 0.0
    unabsorbed_stcl = 0.0
    unabsorbed_ltcl = 0.0

    # 1. Available Short-Term Capital Losses (Section 70(2))
    # Combines equity STCL and Section 50AA Debt STCL
    stcl_equity = abs(gross_111a_pnl) if gross_111a_pnl < 0 else 0.0
    stcl_debt   = abs(net_50aa_stcg) if net_50aa_stcg < 0 else 0.0
    total_stcl_avail = stcl_equity + stcl_debt

    pos_50aa = net_50aa_stcg if net_50aa_stcg > 0 else 0.0
    pos_stcg = gross_111a_pnl if gross_111a_pnl > 0 else 0.0
    pos_ltcg = gross_112a_pnl if gross_112a_pnl > 0 else 0.0

    if total_stcl_avail > 0:
        # Step A: Prioritize setting off STCL against Section 50AA Debt STCG first (highest slab drag)
        if pos_50aa > 0:
            stcl_used_against_50aa = min(total_stcl_avail, pos_50aa)
            post_setoff_50aa = pos_50aa - stcl_used_against_50aa
            total_stcl_avail -= stcl_used_against_50aa
        else:
            post_setoff_50aa = 0.0
            stcl_used_against_50aa = 0.0

        # Step B: Set off against Section 111A Equity STCG
        if total_stcl_avail > 0 and pos_stcg > 0:
            stcl_used_against_stcg = min(total_stcl_avail, pos_stcg)
            post_setoff_stcg = pos_stcg - stcl_used_against_stcg
            total_stcl_avail -= stcl_used_against_stcg
        else:
            post_setoff_stcg = pos_stcg

        # Step C: Set off STCL against LTCG ONLY to the extent LTCG exceeds the ₹1,25,000 exemption limit
        # (Preserves STCL from wasteful absorption against tax-free capital gains)
        taxable_ltcg_before_stcl = max(0.0, pos_ltcg - SEC_112A_EXEMPTION_LIMIT)
        if total_stcl_avail > 0 and taxable_ltcg_before_stcl > 0:
            stcl_used_against_ltcg = min(total_stcl_avail, taxable_ltcg_before_stcl)
            post_setoff_ltcg = pos_ltcg - stcl_used_against_ltcg
            total_stcl_avail -= stcl_used_against_ltcg
        else:
            post_setoff_ltcg = pos_ltcg
            stcl_used_against_ltcg = 0.0

        unabsorbed_stcl = total_stcl_avail
    else:
        post_setoff_50aa = pos_50aa
        post_setoff_stcg = pos_stcg
        post_setoff_ltcg = pos_ltcg
        stcl_used_against_50aa = 0.0
        stcl_used_against_ltcg = 0.0
        unabsorbed_stcl = 0.0

    # 2. Equity LTCL set-off mechanics (Section 70(1) / 74)
    if gross_112a_pnl < 0:
        # LTCL can ONLY set off LTCG (never STCG or Debt)
        post_setoff_ltcg = 0.0
        unabsorbed_ltcl = abs(gross_112a_pnl)
    else:
        unabsorbed_ltcl = 0.0

    total_realized_loss_for_setoff = unabsorbed_stcl + unabsorbed_ltcl

    # 3. Tax Liabilities Post-Exemption Inclusive of 4% Health & Education Cess
    taxable_112a_amt = max(0.0, post_setoff_ltcg - SEC_112A_EXEMPTION_LIMIT)
    tax_payable_112a = taxable_112a_amt * SEC_112A_TAX_RATE
    tax_payable_111a = max(0.0, post_setoff_stcg) * SEC_111A_TAX_RATE

    return {
        'gross_112a_pnl': gross_112a_pnl,
        'gross_111a_pnl': gross_111a_pnl,
        'net_50aa_stcg': net_50aa_stcg,
        'post_setoff_ltcg': post_setoff_ltcg,
        'post_setoff_stcg': post_setoff_stcg,
        'post_setoff_50aa': post_setoff_50aa,
        'stcl_used_against_ltcg': stcl_used_against_ltcg,
        'stcl_used_against_50aa': stcl_used_against_50aa,
        'unabsorbed_stcl': unabsorbed_stcl,
        'unabsorbed_ltcl': unabsorbed_ltcl,
        'total_realized_loss_for_setoff': total_realized_loss_for_setoff,
        'taxable_112a_amt': taxable_112a_amt,
        'tax_payable_112a': tax_payable_112a,
        'tax_payable_111a': tax_payable_111a
    }

# ----------------------------------------------------------------------------------------------------
# 3. UNREALIZED TAX-LOSS & QUALIFIED LTCG PROFIT HARVESTING
# ----------------------------------------------------------------------------------------------------
def compute_unrealized_tax_lots_analysis(
    tax_lots_df: Optional[pd.DataFrame],
    latest_prices_series: Union[pd.Series, Dict[str, float], None],
    current_date: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Analyzes active tax lots to identify:
      1. Qualified Section 112A LTCG profits (Equity/Gold holding period >= 365 days, unrealized gain > 0).
         Strictly excludes Section 50AA Debt ETFs from LTCG exemption.
         Applies Section 55(2)(ac) grandfathering for pre-2018 lots.
      2. Harvestable capital losses categorized into STCG, LTCG, and Section 50AA debt losses.
    Supports fractional and decimal lot quantities.
    """
    today_raw = pd.to_datetime(current_date) if current_date is not None else get_market_time_horizons()['TODAY']
    today = today_raw.tz_localize(None) if getattr(today_raw, 'tzinfo', None) is not None else today_raw
    harvest_rows = []
    total_harvestable_profit = 0.0

    loss_rows = []
    total_harvestable_losses = 0.0

    if tax_lots_df is None or tax_lots_df.empty:
        return {
            'harvest_rows': [],
            'total_harvestable_profit': 0.0,
            'loss_rows': [],
            'total_harvestable_losses': 0.0
        }

    # Normalize latest_prices_series to pd.Series for safe .get() access
    if isinstance(latest_prices_series, dict):
        latest_prices_series = pd.Series(latest_prices_series, dtype='float64')
    elif not isinstance(latest_prices_series, pd.Series):
        latest_prices_series = pd.Series(dtype='float64')

    for _, lot in tax_lots_df.iterrows():
        t = str(lot['ticker'])
        asset_cls = str(lot.get('asset_class', ''))
        is_50aa_debt = (t == SOVEREIGN_BOND_TICKER) or ('50AA' in asset_cls) or ('Debt' in asset_cls)

        try:
            p_live = float(latest_prices_series.get(t, 0.0))
            if pd.isna(p_live) or p_live < 0:
                p_live = 0.0
        except Exception:
            p_live = 0.0
            
        p_buy = float(lot['buy_price'])
        q = float(lot['quantity'])
        q_disp = round(q, 4) if not q.is_integer() else int(q)
        
        buy_raw = pd.to_datetime(lot['buy_date'])
        buy_dt = buy_raw.tz_localize(None) if getattr(buy_raw, 'tzinfo', None) is not None else buy_raw
        days_held = (today - buy_dt).days

        # Apply Section 55(2)(ac) grandfathering for pre-2018 tax lots
        fmv_val = lot.get('fmv_31jan2018', lot.get('fmv', None))
        cost_basis = compute_grandfathered_cost_basis(lot.get('buy_date'), p_buy, p_live, fmv_val)

        # 1. Qualified LTCG Profit Check (Strictly Equity / Non-50AA Assets)
        # Under Section 50AA, Debt ETFs are deemed STCG forever and CANNOT enter harvest_rows
        if not is_50aa_debt and days_held >= 365 and p_live > cost_basis and p_live > 0:
            profit = (p_live - cost_basis) * q
            total_harvestable_profit += profit
            harvest_rows.append({
                'Ticker': t.replace('.NS', ''),
                'Asset Class': lot['asset_class'],
                'Buy Date': lot['buy_date'],
                'Days Held': days_held,
                'Quantity': q_disp,
                'Buy Price (₹)': np.round(cost_basis, 2),
                'Live Price (₹)': np.round(p_live, 2),
                'Unrealized LTCG Profit (₹)': np.round(profit, 2)
            })

        # 2. Tax-Loss Harvesting Check
        if p_live < cost_basis and p_live > 0:
            loss_amount = (cost_basis - p_live) * q
            total_harvestable_losses += loss_amount
            loss_category = (
                '🟡 Debt Loss (Sec 50AA)' if is_50aa_debt 
                else ('🔴 STCG Loss (<365d)' if days_held < 365 else '🔵 LTCG Loss (≥365d)')
            )
            loss_rows.append({
                'Ticker': t.replace('.NS', ''),
                'Asset Class': lot['asset_class'],
                'Buy Date': lot['buy_date'],
                'Days Held': days_held,
                'Loss Category': loss_category,
                'Quantity': q_disp,
                'Buy Price (₹)': np.round(cost_basis, 2),
                'Live Price (₹)': np.round(p_live, 2),
                'Unrealized Loss (₹)': np.round(loss_amount, 2)
            })

    return {
        'harvest_rows': harvest_rows,
        'total_harvestable_profit': total_harvestable_profit,
        'loss_rows': loss_rows,
        'total_harvestable_losses': total_harvestable_losses
    }

# ----------------------------------------------------------------------------------------------------
# 4. SCHEDULE 112A / SCHEDULE CG TAX EXPORT BUILDER
# ----------------------------------------------------------------------------------------------------
def build_schedule_112a_records(filtered_sells: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Formats sell trade ledger records into ITR-2 / ITR-3 Schedule 112A / Schedule CG tax audit structure.
    Enforces Section 48 non-deductibility of Securities Transaction Tax (STT).
    Applies Section 55(2)(ac) pre-2018 grandfathered cost basis when applicable.
    Supports fractional and decimal lot quantities.
    """
    if filtered_sells is None or filtered_sells.empty:
        return pd.DataFrame()

    schedule_rows = []
    for _, row in filtered_sells.iterrows():
        t = str(row['ticker'])
        q = float(row['quantity'])
        q_disp = round(q, 4) if not q.is_integer() else int(q)
        p_sale = float(row['price'])
        pnl = float(row['realized_pnl'])
        fees = float(row.get('fees_estimated', 0.0))
        g_type = str(row.get('gain_type', ''))
        
        tot_sale = q * p_sale
        tot_cost = tot_sale - pnl
        cost_per_share = tot_cost / q if q > 0 else 0.0

        b_date = row.get('buy_date')
        if b_date and pd.notna(b_date) and str(b_date).strip()[:10] < '2018-02-01' and g_type == 'LTCG':
            fmv = row.get('fmv_31jan2018', row.get('fmv', None))
            actual_buy_price = float(row['buy_price']) if ('buy_price' in row and pd.notna(row['buy_price'])) else cost_per_share
            cost_per_share = compute_grandfathered_cost_basis(b_date, actual_buy_price, p_sale, fmv)
            tot_cost = q * cost_per_share
            pnl = tot_sale - tot_cost

        if g_type == 'LTCG':
            sec = 'Section 112A (LTCG)'
        elif '50AA' in g_type or t == SOVEREIGN_BOND_TICKER:
            sec = 'Section 50AA (Debt STCG)'
        else:
            sec = 'Section 111A (STCG)'

        # Under Section 48, STT is non-deductible; eligible transfer expenses include brokerage/GST/stamp duty/exchange fees
        if 'stt' in row and pd.notna(row.get('stt')):
            stt_estimated = float(row.get('stt'))
        else:
            stt_estimated = round(0.001 * tot_sale, 2)
        eligible_deductions = round(max(0.0, fees - stt_estimated), 2)

        schedule_rows.append({
            'ISIN / Ticker': t,
            'Asset Name': t.replace('.NS', ''),
            'Date of Sale': row.get('execution_date', ''),
            'Quantity Sold': q_disp,
            'Sale Price (₹)': np.round(p_sale, 2),
            'Cost of Acquisition (₹)': np.round(cost_per_share, 2),
            'Total Sale Consideration': np.round(tot_sale, 2),
            'Total Cost': np.round(tot_cost, 2),
            'Deductions (Eligible Transfer Expenses)': np.round(eligible_deductions, 2),
            'Non-Deductible STT (Sec 48)': np.round(stt_estimated, 2),
            'Total Fees Paid': np.round(fees, 2),
            'Realized Capital Gain / Loss': np.round(pnl, 2),
            'Tax Section (112A / 111A / 50AA)': sec
        })

    return pd.DataFrame(schedule_rows)

# ----------------------------------------------------------------------------------------------------
# 5. ACTIVE TAX-LOSS NEUTRALIZER (AUTOMATED HARVESTING RECOMMENDER)
# ----------------------------------------------------------------------------------------------------
def recommend_tax_loss_harvesting_trades(
    tax_summary: Optional[Dict[str, float]],
    unrealized_tax_analysis: Optional[Dict[str, Any]],
) -> pd.DataFrame:
    """
    Automated Active Tax-Loss Neutralizer Algorithm:
    Calculates the exact number of shares to harvest from loss positions to neutralize realized
    capital gains tax liabilities (Section 50AA Debt STCG @ 31.2%, Section 111A STCG @ 20.8%, and Section 112A LTCG @ 13.0% inclusive of 4% Cess) to ₹0.00.
    Enforces whole integer share quantities for direct Zerodha Kite execution.
    Includes statutory compliance warning regarding same-day intraday netting under Section 43(5).
    """
    cols = [
        'Ticker', 'Action', 'Shares to Sell', 'Live Price (₹)',
        'Estimated Realized Loss (₹)', 'Tax Savings (₹)', 'Post-Harvest Tax Liability',
        'Compliance Note'
    ]
    if not tax_summary or not unrealized_tax_analysis:
        return pd.DataFrame(columns=cols)

    debt_target = float(tax_summary.get('post_setoff_50aa', 0.0))
    stcg_target = float(tax_summary.get('post_setoff_stcg', 0.0))
    ltcg_target = float(tax_summary.get('taxable_112a_amt', 0.0))

    if (debt_target <= 0 and stcg_target <= 0 and ltcg_target <= 0):
        return pd.DataFrame(columns=cols)

    loss_rows = unrealized_tax_analysis.get('loss_rows', [])
    if not loss_rows:
        return pd.DataFrame(columns=cols)

    # Prioritize STCG / Debt losses (<365 days / Sec 50AA) first since STCL can offset Debt STCG @ 31.2% and Equity STCG @ 20.8%, then LTCG losses
    def _sort_loss_rows(r: Dict[str, Any]):
        cat = str(r.get('Loss Category', ''))
        if 'STCG' in cat or 'Debt' in cat or '50AA' in cat:
            cat_rank = 0
        elif 'LTCG' in cat:
            cat_rank = 1
        else:
            cat_rank = 2
        return (cat_rank, -float(r.get('Unrealized Loss (₹)', 0.0)))

    sorted_loss_rows = sorted(loss_rows, key=_sort_loss_rows)

    rem_debt = debt_target
    rem_stcg = stcg_target
    rem_ltcg = ltcg_target
    recommendations = []

    for lot in sorted_loss_rows:
        if rem_debt <= 0 and rem_stcg <= 0 and rem_ltcg <= 0:
            break

        p_buy = float(lot['Buy Price (₹)'])
        p_live = float(lot['Live Price (₹)'])
        q_lot = float(lot['Quantity'])
        loss_cat = str(lot.get('Loss Category', ''))
        is_stcl = ('STCG' in loss_cat) or ('Debt' in loss_cat) or ('50AA' in loss_cat)

        per_share_loss = p_buy - p_live
        if per_share_loss <= 0 or q_lot <= 0:
            continue

        # If LTCL, it can only offset LTCG under Section 70
        if not is_stcl and 'LTCG' in loss_cat:
            if rem_ltcg <= 0:
                continue
            target_loss_needed = rem_ltcg
        else:
            # STCL can offset Section 50AA Debt STCG, Section 111A STCG, and taxable Section 112A LTCG
            target_loss_needed = rem_debt + rem_stcg + rem_ltcg
            if target_loss_needed <= 0:
                continue

        # Enforce strict integer quantities for Zerodha execution
        shares_needed = int(np.ceil(target_loss_needed / per_share_loss))
        shares_to_sell = int(min(np.floor(q_lot), shares_needed))

        if shares_to_sell <= 0:
            continue

        realized_loss = float(shares_to_sell * per_share_loss)
        tax_savings = 0.0

        if is_stcl:
            leftover_loss = realized_loss
            # Step 1: Offset Section 50AA Debt STCG @ 31.2% (highest slab tax savings)
            if rem_debt > 0:
                debt_absorbed = min(rem_debt, leftover_loss)
                rem_debt -= debt_absorbed
                tax_savings += debt_absorbed * SEC_50AA_TAX_RATE
                leftover_loss -= debt_absorbed

            # Step 2: Offset Section 111A Equity STCG @ 20.8%
            if leftover_loss > 0 and rem_stcg > 0:
                stcg_absorbed = min(rem_stcg, leftover_loss)
                rem_stcg -= stcg_absorbed
                tax_savings += stcg_absorbed * SEC_111A_TAX_RATE
                leftover_loss -= stcg_absorbed

            # Step 3: Offset taxable Section 112A LTCG @ 13.0%
            if leftover_loss > 0 and rem_ltcg > 0:
                ltcg_absorbed = min(rem_ltcg, leftover_loss)
                rem_ltcg -= ltcg_absorbed
                tax_savings += ltcg_absorbed * SEC_112A_TAX_RATE
        else:
            # LTCL offsets taxable Section 112A LTCG @ 13.0%
            ltcg_absorbed = min(rem_ltcg, realized_loss)
            rem_ltcg -= ltcg_absorbed
            tax_savings += ltcg_absorbed * SEC_112A_TAX_RATE

        post_harvest_liability = (rem_debt * SEC_50AA_TAX_RATE) + (rem_stcg * SEC_111A_TAX_RATE) + (rem_ltcg * SEC_112A_TAX_RATE)

        recommendations.append({
            'Ticker': lot['Ticker'],
            'Action': f"🔴 TRIM {shares_to_sell} Share(s)",
            'Shares to Sell': shares_to_sell,
            'Live Price (₹)': p_live,
            'Estimated Realized Loss (₹)': np.round(realized_loss, 2),
            'Tax Savings (₹)': np.round(tax_savings, 2),
            'Post-Harvest Tax Liability': np.round(post_harvest_liability, 2),
            'Compliance Note': "⚠️ Note: To avoid same-day intraday netting (Sec 43(5)), repurchase on T+1 day or switch into a correlated sector proxy."
        })

    return pd.DataFrame(recommendations)
