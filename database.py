# ====================================================================================================
# MASTER QUANTITATIVE WEALTH OPERATING SYSTEM - DATABASE & AUDIT LEDGER MODULE
# Relational SQLite Architecture, FIFO Tax Lots, Corporate Splits, and Broker CSV Importers
# ====================================================================================================

import os
import io
import sqlite3
import datetime
import collections
import pandas as pd
import numpy as np
import yfinance as yf

from config import (
    DB_FILE, STATUTORY_FEE_BUFFER, SOVEREIGN_BOND_TICKER,
    EQUITY_DELIVERY_FRICTION, ETF_DEBT_GOLD_FRICTION, DP_CHARGE_FLAT_INR,
    get_market_time_horizons, TODAY, TODAY_STR, TODAY_ISO
)

# ----------------------------------------------------------------------------------------------------
# 1. DATABASE INITIALIZATION & CONNECTION POOL (WAL & THREAD-SAFE CONFIGURATION)
# ----------------------------------------------------------------------------------------------------
def get_db_connection():
    """
    Returns a thread-safe SQLite connection handle with Write-Ahead Logging (WAL),
    foreign key enforcement, busy timeout, and optimized synchronous pragmas.
    """
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    """
    Initializes the relational tax_lots and trade_ledger tables and performance indexes if not present.
    Configures SQLite WAL mode, foreign key enforcement, and REAL decimal quantities.
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        # Table 1: Current holdings and FIFO tax lots (quantity as REAL for fractional/mutual funds)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tax_lots (
                lot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                asset_name TEXT,
                asset_class TEXT,
                buy_date TEXT,
                quantity REAL,
                buy_price REAL,
                last_split_date TEXT DEFAULT ''
            )
        """)
        # Table 2: Immutable Trade Audit Ledger (quantity as REAL)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_ledger (
                trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
                execution_date TEXT,
                ticker TEXT,
                action TEXT,
                quantity REAL,
                price REAL,
                realized_pnl REAL,
                gain_type TEXT,
                fees_estimated REAL
            )
        """)
        # Database Performance Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tax_lots_ticker ON tax_lots(ticker);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tax_lots_qty ON tax_lots(quantity);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_ledger_date ON trade_ledger(execution_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_ledger_ticker ON trade_ledger(ticker);")
        conn.commit()

def create_database_backup(db_file=DB_FILE, backup_dir='db_backups'):
    """
    Creates an atomic online snapshot of the SQLite database using SQLite's native backup API.
    Maintains a rolling window of the 10 most recent backups in backup_dir and auto-prunes older files.
    """
    if not os.path.exists(db_file):
        return None

    os.makedirs(backup_dir, exist_ok=True)
    timestamp_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = os.path.join(backup_dir, f"wealth_ledger_{timestamp_str}.db.bak")
    if os.path.exists(backup_path):
        backup_path = os.path.join(backup_dir, f"wealth_ledger_{timestamp_str}_{datetime.datetime.now().strftime('%f')}.db.bak")

    src = None
    dst = None
    try:
        src = sqlite3.connect(db_file)
        dst = sqlite3.connect(backup_path)
        src.backup(dst)
    except Exception as e:
        print(f"Warning: Failed to create database backup: {e}")
        return None
    finally:
        if dst is not None:
            dst.close()
        if src is not None:
            src.close()

    # Prune older backups: retain only the latest 10 .db.bak files
    try:
        backup_files = [
            os.path.join(backup_dir, f)
            for f in os.listdir(backup_dir)
            if f.endswith('.db.bak')
        ]
        backup_files.sort()
        if len(backup_files) > 10:
            for old_file in backup_files[:-10]:
                try:
                    os.remove(old_file)
                except Exception:
                    pass
    except Exception:
        pass

    return backup_path

def backup_database(db_file=DB_FILE, backup_file='db_backups/wealth_ledger_backup.db'):
    """
    Copies wealth_ledger.db into db_backups/wealth_ledger_backup.db whenever trades are committed.
    Also calls create_database_backup() to ensure rolling versioned snapshots are maintained.
    """
    if not os.path.exists(db_file):
        return None

    os.makedirs(os.path.dirname(backup_file) if os.path.dirname(backup_file) else '.', exist_ok=True)
    src = None
    dst = None
    try:
        src = sqlite3.connect(db_file)
        dst = sqlite3.connect(backup_file)
        src.backup(dst)
    except Exception as e:
        print(f"Warning: Failed to backup database to {backup_file}: {e}")
        return None
    finally:
        if dst is not None:
            dst.close()
        if src is not None:
            src.close()

    create_database_backup(db_file=db_file)
    return backup_file

# ----------------------------------------------------------------------------------------------------
# 2. STATUTORY EXCHANGE FRICTION & DATAFRAME SANITIZERS
# ----------------------------------------------------------------------------------------------------
def get_trade_friction(ticker, quantity, price, action='BUY', asset_class=''):
    """
    Calculates statutory Indian exchange friction in INR:
      - Equity Delivery: 0.15% (STT 0.1% + Brokerage/GST/Exchange turnover)
      - Sovereign Debt & Precious Metals ETFs: 0.03% (0% STT + GST/Stamp duty)
      - Sell DP charges: flat DP charge (₹15.93) added on sell transactions
    """
    t_upper = str(ticker).upper()
    cls_upper = str(asset_class).upper()
    is_debt_or_gold = (
        t_upper == SOVEREIGN_BOND_TICKER.upper() or
        'GOLDBEES' in t_upper or
        'SILVERBEES' in t_upper or
        'DEBT' in cls_upper or
        'GOLD' in cls_upper or
        '50AA' in cls_upper or
        'SOVEREIGN' in cls_upper
    )
    friction_rate = ETF_DEBT_GOLD_FRICTION if is_debt_or_gold else EQUITY_DELIVERY_FRICTION
    trade_val = float(quantity) * float(price)
    fees = trade_val * friction_rate
    if str(action).upper() == 'SELL':
        fees += DP_CHARGE_FLAT_INR
    return round(float(fees), 2)

def clean_tax_lots_df(df):
    """
    Sanitizes raw SQLite records into strongly typed Pandas DataFrames with exact precision
    supporting fractional shares, mutual fund decimal units, and exact rupee precision.
    """
    if df.empty:
        return pd.DataFrame(columns=['lot_id', 'ticker', 'asset_name', 'asset_class', 'buy_date', 'quantity', 'buy_price', 'last_split_date'])
    
    def parse_qty(x):
        if isinstance(x, bytes):
            try:
                import struct
                return round(float(struct.unpack('<d', x)[0]), 4)
            except Exception:
                try: return float(int.from_bytes(x, byteorder='little'))
                except Exception: return 0.0
        try: return round(float(x), 4)
        except Exception: return 0.0

    def parse_price(x):
        try: return round(float(x), 2)
        except Exception: return 0.0

    df['quantity'] = df['quantity'].apply(parse_qty)
    df['buy_price'] = df['buy_price'].apply(parse_price)
    if 'lot_id' in df.columns:
        df['lot_id'] = pd.to_numeric(df['lot_id'], errors='coerce').fillna(0).astype(int)
    if 'last_split_date' not in df.columns:
        df['last_split_date'] = ''
    else:
        df['last_split_date'] = df['last_split_date'].fillna('').astype(str)
    if 'fmv_31jan2018' in df.columns:
        df['fmv_31jan2018'] = pd.to_numeric(df['fmv_31jan2018'], errors='coerce')
    return df

def clean_trade_ledger_df(df):
    """
    Sanitizes raw trade ledger records with explicit numeric casting supporting decimal quantities
    and exact rupee precision.
    """
    if df.empty:
        return pd.DataFrame(columns=['trade_id', 'execution_date', 'ticker', 'action', 'quantity', 'price', 'realized_pnl', 'gain_type', 'fees_estimated'])
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0.0).apply(lambda v: round(float(v), 4))
    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0.0).apply(lambda v: round(float(v), 2))
    df['realized_pnl'] = pd.to_numeric(df['realized_pnl'], errors='coerce').fillna(0.0).apply(lambda v: round(float(v), 2))
    df['fees_estimated'] = pd.to_numeric(df['fees_estimated'], errors='coerce').fillna(0.0).apply(lambda v: round(float(v), 2))
    if 'fmv_31jan2018' in df.columns:
        df['fmv_31jan2018'] = pd.to_numeric(df['fmv_31jan2018'], errors='coerce')
    if 'buy_date' in df.columns:
        df['buy_date'] = df['buy_date'].fillna('').astype(str)
    return df

# ----------------------------------------------------------------------------------------------------
# 3. TIMEZONE-NEUTRAL STOCK SPLIT & BONUS RECONCILER (ATOMIC TRANSACTION)
# ----------------------------------------------------------------------------------------------------
def reconcile_corporate_actions_and_splits(conn=None):
    """
    Scans active tax lots for historical stock splits / bonuses via yfinance
    and adjusts quantities and cost bases in-place with ACID safety inside an atomic transaction.
    Can be called manually from Tab 5 or automatically upon trade execution.
    """
    reconciled_messages = []
    should_close = False
    if conn is None:
        conn = get_db_connection()
        should_close = True

    try:
        lots_df = clean_tax_lots_df(pd.read_sql("SELECT lot_id, ticker, buy_date, quantity, buy_price, last_split_date FROM tax_lots WHERE quantity > 0", conn))
        if not lots_df.empty:
            cursor = conn.cursor()
            for t in lots_df['ticker'].unique():
                try:
                    try:
                        from quant_engine import GLOBAL_RESILIENT_SESSION
                        ticker_obj = yf.Ticker(t, session=GLOBAL_RESILIENT_SESSION)
                    except Exception:
                        ticker_obj = yf.Ticker(t)
                    splits = ticker_obj.splits
                    if splits is not None and not splits.empty:
                        t_lots = lots_df[lots_df['ticker'] == t]
                        for _, lot in t_lots.iterrows():
                            lot_buy_dt = pd.to_datetime(lot['buy_date']).tz_localize(None)
                            last_split_str = str(lot['last_split_date']).strip()
                            latest_split_dt = pd.to_datetime(last_split_str).tz_localize(None) if (last_split_str and last_split_str != 'None') else lot_buy_dt
                            
                            relevant_splits = []
                            for split_dt, split_ratio in splits.items():
                                naive_dt = pd.to_datetime(split_dt).tz_localize(None)
                                if naive_dt > lot_buy_dt and naive_dt > latest_split_dt and split_ratio > 0:
                                    relevant_splits.append((naive_dt, split_ratio))
                            
                            if relevant_splits:
                                relevant_splits.sort(key=lambda x: x[0])
                                running_qty = float(lot['quantity'])
                                running_price = float(lot['buy_price'])
                                for s_date, split_ratio in relevant_splits:
                                    if split_ratio != 1.0 and split_ratio > 0:
                                        running_qty = round(float(running_qty * split_ratio), 4)
                                        running_price = round(float(running_price / split_ratio), 2)
                                        if s_date > latest_split_dt:
                                            latest_split_dt = s_date
                                if running_qty != lot['quantity']:
                                    cursor.execute("""
                                        UPDATE tax_lots 
                                        SET quantity = ?, buy_price = ?, last_split_date = ? 
                                        WHERE lot_id = ?
                                    """, [float(running_qty), float(running_price), latest_split_dt.strftime('%Y-%m-%d'), int(lot['lot_id'])])
                                    reconciled_messages.append(f"⚠️ Corporate Action Reconciled: {t} compounded to {running_qty:g} shares (Adj. Price: ₹{running_price:.2f}).")
                except Exception as e:
                    reconciled_messages.append(f"⚠️ Split check failed for {t}: {str(e)}")
            conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        if should_close:
            conn.close()

    return reconciled_messages

# ----------------------------------------------------------------------------------------------------
# 4. BROKER TRADEBOOK CSV PARSER (ZERODHA CONSOLE & GROWW) - STRICT FIFO MATCHING & ATOMIC TRANSACTIONS
# ----------------------------------------------------------------------------------------------------
def parse_and_import_broker_csv(file_bytes, conn, mode='Merge New Trades (Incremental)'):
    """
    Parses Zerodha Console Tradebook, Groww Order Report, or generic broker CSV.
    Supports both Incremental Merge (with robust tranche / SIP support) and Full Ledger Reset & Rebuild.
    Properly records FIFO-matched realized SELL trades into trade_ledger with statutory gain_type classification
    inside an atomic database transaction.
    """
    try:
        text = file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        text = file_bytes.decode('latin1', errors='ignore')

    lines = text.splitlines()
    if not lines:
        return {'success': False, 'error': 'Uploaded CSV file is empty.'}

    # Detect header row index
    header_idx = 0
    for i, line in enumerate(lines[:20]):
        lower_line = line.lower()
        if any(k in lower_line for k in ['symbol', 'trade_date', 'stock name', 'order date', 'isin', 'trade_type', 'transaction type', 'trade price', 'avg. price']):
            header_idx = i
            break

    try:
        df = pd.read_csv(io.StringIO("\n".join(lines[header_idx:])))
    except Exception as e:
        return {'success': False, 'error': f"Failed to parse CSV: {str(e)}"}

    if df.empty:
        return {'success': False, 'error': 'No data found in uploaded CSV.'}

    col_map = {str(c).strip().lower(): c for c in df.columns}
    
    # Identify broker
    broker_name = "Generic CSV"
    if any('trade_type' in c or 'trade_date' in c or 'series' in c for c in col_map):
        broker_name = "Zerodha (Console Tradebook)"
    elif any('stock name' in c or 'order date' in c or 'avg. price' in c for c in col_map):
        broker_name = "Groww (Order Report)"

    def find_col(possible_names):
        for name in possible_names:
            for clean_col, orig_col in col_map.items():
                if name == clean_col:
                    return orig_col
        for name in possible_names:
            for clean_col, orig_col in col_map.items():
                if name in clean_col:
                    return orig_col
        return None

    symbol_col = find_col(['symbol', 'stock symbol', 'scrip', 'instrument', 'stock name', 'company name', 'ticker'])
    type_col = find_col(['trade_type', 'trade type', 'type', 'transaction type', 'action', 'side', 'buy/sell'])
    date_col = find_col(['trade_date', 'trade date', 'order_execution_time', 'execution date', 'order date', 'date', 'order_timestamp'])
    qty_col = find_col(['quantity', 'qty', 'shares', 'filled_qty', 'executed_qty', 'traded qty', 'traded quantity'])
    price_col = find_col(['price', 'trade_price', 'trade price', 'avg_price', 'avg. price', 'average price', 'rate', 'execution price', 'buy price'])
    status_col = find_col(['status', 'order status', 'trade status'])
    product_col = find_col(['segment', 'product', 'order_type'])
    order_id_col = find_col(['order_id', 'order id', 'trade_id', 'trade id', 'trade_no', 'order no'])

    if not symbol_col or not qty_col or not price_col:
        return {
            'success': False,
            'error': f"Missing required columns. Found: {list(df.columns)}. Need Symbol/Stock, Quantity, and Price columns."
        }

    # Filter by Status if present
    if status_col:
        df = df[df[status_col].astype(str).str.upper().str.contains('EXEC|COMPLET|FILL|DONE|SUCCESS', na=True)]

    # Filter by Product/Segment if present (exclude Intraday/MIS)
    if product_col:
        df = df[~df[product_col].astype(str).str.upper().str.contains('MIS|INTRADAY', na=False)]

    records = []
    for _, row in df.iterrows():
        raw_sym = str(row[symbol_col]).strip().upper()
        if not raw_sym or raw_sym == 'NAN' or raw_sym == 'NONE':
            continue
        
        if ':' in raw_sym:
            raw_sym = raw_sym.split(':')[-1].strip()

        if not raw_sym.endswith('.NS') and not raw_sym.endswith('.BO'):
            ticker = f"{raw_sym}.NS"
        else:
            ticker = raw_sym.replace('.BO', '.NS')

        action = 'BUY'
        if type_col and pd.notna(row[type_col]):
            t_str = str(row[type_col]).strip().upper()
            if 'SELL' in t_str or t_str.startswith('S'):
                action = 'SELL'
            else:
                action = 'BUY'

        try:
            qty = round(float(row[qty_col]), 4)
        except Exception:
            qty = 0.0
        if qty <= 0:
            continue

        try:
            price = round(float(row[price_col]), 2)
        except Exception:
            price = 0.0
        if price <= 0:
            continue

        date_val = datetime.date.today().strftime('%Y-%m-%d')
        order_time_str = ''
        if date_col and pd.notna(row[date_col]):
            try:
                raw_dt_str = str(row[date_col]).strip()
                parsed_dt = pd.to_datetime(raw_dt_str, format='mixed', errors='coerce')
                if pd.isna(parsed_dt):
                    parsed_dt = pd.to_datetime(raw_dt_str, errors='coerce')
                if pd.notna(parsed_dt):
                    date_val = parsed_dt.strftime('%Y-%m-%d')
                    order_time_str = parsed_dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                pass

        order_id_val = str(row[order_id_col]).strip() if (order_id_col and pd.notna(row[order_id_col])) else ''

        if ticker == 'GILT5YBEES.NS':
            asset_name = 'GILT5YBEES (5-Yr G-Sec)'
            asset_class = 'Sovereign Debt ETF (Sec 50AA)'
        elif ticker == 'EMBASSY.NS':
            asset_name = 'Embassy Office Parks REIT'
            asset_class = 'Real Estate (REIT)'
        elif ticker == 'GOLDBEES.NS':
            asset_name = 'Nippon India ETF Gold BeES'
            asset_class = 'Gold ETF'
        else:
            asset_name = str(row.get('stock name', str(row.get('company name', ticker.replace('.NS', ''))))).strip()
            if not asset_name or asset_name == 'nan' or asset_name == ticker:
                asset_name = ticker.replace('.NS', '')
            asset_class = 'Large-Cap Equity'

        records.append({
            'ticker': ticker,
            'asset_name': asset_name,
            'asset_class': asset_class,
            'action': action,
            'trade_date': date_val,
            'order_time': order_time_str,
            'order_id': order_id_val,
            'quantity': qty,
            'price': price
        })

    if not records:
        return {'success': False, 'error': 'No valid trade records could be extracted from CSV.'}

    records_df = pd.DataFrame(records)

    # Consolidate same-day BUYs of the same ticker into a single weighted-average cost lot before checking duplicates:
    # Consolidated Qty = sum(q_i), Weighted Avg Price = sum(q_i * p_i) / sum(q_i)
    consolidated_records = []
    buy_df = records_df[records_df['action'] == 'BUY']
    sell_df = records_df[records_df['action'] == 'SELL']

    if not buy_df.empty:
        for (trade_dt, t), group in buy_df.groupby(['trade_date', 'ticker'], sort=False):
            tot_qty = float(group['quantity'].sum())
            if tot_qty > 0:
                tot_outlay = float((group['quantity'] * group['price']).sum())
                w_avg_p = round(tot_outlay / tot_qty, 2)
                first_row = group.iloc[0]
                consolidated_records.append({
                    'ticker': t,
                    'asset_name': first_row['asset_name'],
                    'asset_class': first_row['asset_class'],
                    'action': 'BUY',
                    'trade_date': trade_dt,
                    'order_time': str(first_row.get('order_time', '')),
                    'order_id': str(first_row.get('order_id', '')),
                    'quantity': round(tot_qty, 4),
                    'price': w_avg_p
                })

    if not sell_df.empty:
        for _, row in sell_df.iterrows():
            consolidated_records.append(row.to_dict())

    records_df = pd.DataFrame(consolidated_records)
    records_df['action_priority'] = records_df['action'].apply(lambda a: 0 if str(a).upper() == 'BUY' else 1)
    records_df = records_df.sort_values(by=['trade_date', 'order_time', 'action_priority'], ascending=[True, True, True])

    create_database_backup()
    cursor = conn.cursor()
    imported_count = 0
    skipped_count = 0
    total_invested = 0.0
    active_lots_count = 0
    realized_sells_count = 0
    unmatched_sells_count = 0

    try:
        if 'Full Ledger Reset' in mode:
            cursor.execute("DELETE FROM tax_lots")
            cursor.execute("DELETE FROM trade_ledger")
            
            for t, t_df in records_df.groupby('ticker'):
                buy_lots = []
                for _, trade in t_df.iterrows():
                    if trade['action'] == 'BUY':
                        b_lot = {
                            'ticker': trade['ticker'],
                            'asset_name': trade['asset_name'],
                            'asset_class': trade['asset_class'],
                            'buy_date': trade['trade_date'],
                            'quantity': float(trade['quantity']),
                            'buy_price': float(trade['price'])
                        }
                        buy_lots.append(b_lot)
                        fees_buy = get_trade_friction(trade['ticker'], trade['quantity'], trade['price'], action='BUY', asset_class=trade['asset_class'])
                        cursor.execute("""
                            INSERT INTO trade_ledger (execution_date, ticker, action, quantity, price, realized_pnl, gain_type, fees_estimated)
                            VALUES (?, ?, 'BUY', ?, ?, 0.0, 'N/A', ?)
                        """, [trade['trade_date'], trade['ticker'], float(trade['quantity']), float(trade['price']), float(fees_buy)])
                    elif trade['action'] == 'SELL':
                        sell_rem = float(trade['quantity'])
                        sell_price = float(trade['price'])
                        sell_date = trade['trade_date']
                        
                        for lot in buy_lots:
                            if sell_rem <= 1e-6:
                                break
                            if lot['quantity'] > 1e-6:
                                deduct = min(lot['quantity'], sell_rem)
                                lot['quantity'] -= deduct
                                sell_rem -= deduct
                                
                                b_price = float(lot['buy_price'])
                                b_date = lot['buy_date']
                                matched_pnl = round((sell_price - b_price) * deduct, 2)
                                
                                days_held = (pd.to_datetime(sell_date) - pd.to_datetime(b_date)).days
                                is_50aa = (t == SOVEREIGN_BOND_TICKER) or ('50AA' in str(lot.get('asset_class', ''))) or ('Debt' in str(lot.get('asset_class', '')))
                                
                                if is_50aa:
                                    gain_type = 'Section 50AA (Debt STCG)'
                                elif days_held >= 365:
                                    gain_type = 'LTCG'
                                else:
                                    gain_type = 'STCG'
                                    
                                est_fees = get_trade_friction(t, deduct, sell_price, action='SELL', asset_class=lot.get('asset_class', ''))
                                cursor.execute("""
                                    INSERT INTO trade_ledger (execution_date, ticker, action, quantity, price, realized_pnl, gain_type, fees_estimated)
                                    VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?)
                                """, [sell_date, t, float(deduct), float(sell_price), float(matched_pnl), gain_type, float(est_fees)])
                                realized_sells_count += 1

                        if sell_rem > 1e-6:
                            unmatched_sells_count += 1

                for lot in buy_lots:
                    if lot['quantity'] > 1e-6:
                        cursor.execute("""
                            INSERT INTO tax_lots (ticker, asset_name, asset_class, buy_date, quantity, buy_price, last_split_date)
                            VALUES (?, ?, ?, ?, ?, ?, '')
                        """, [lot['ticker'], lot['asset_name'], lot['asset_class'], lot['buy_date'], float(lot['quantity']), float(lot['buy_price'])])
                        imported_count += 1
                        active_lots_count += 1
                        total_invested += round(lot['quantity'] * lot['buy_price'], 2)

        else: # Incremental Merge
            # Load existing active tax lots from database
            cursor.execute("SELECT lot_id, ticker, asset_name, asset_class, buy_date, quantity, buy_price, last_split_date FROM tax_lots WHERE quantity > 0 ORDER BY buy_date ASC, lot_id ASC")
            existing_lot_rows = cursor.fetchall()
            
            # Group existing lots by ticker
            existing_lots_by_ticker = {}
            for r in existing_lot_rows:
                tk = r[1]
                if tk not in existing_lots_by_ticker:
                    existing_lots_by_ticker[tk] = []
                existing_lots_by_ticker[tk].append({
                    'lot_id': r[0],
                    'ticker': r[1],
                    'asset_name': r[2],
                    'asset_class': r[3],
                    'buy_date': r[4],
                    'quantity': float(r[5]),
                    'buy_price': float(r[6]),
                    'last_split_date': r[7],
                    'is_existing_db': True
                })

            # Load existing trades into a multiset Counter to support multiple identical fills on the same date
            cursor.execute("SELECT execution_date, ticker, action, quantity, price FROM trade_ledger")
            existing_trades_counter = collections.Counter(
                (r[0], r[1], r[2], round(float(r[3]), 4), round(float(r[4]), 2)) for r in cursor.fetchall()
            )

            for t, t_df in records_df.groupby('ticker'):
                ticker_active_lots = existing_lots_by_ticker.get(t, [])
                
                for _, trade in t_df.iterrows():
                    trade_dt = trade['trade_date']
                    trade_act = trade['action']
                    trade_q = round(float(trade['quantity']), 4)
                    trade_p = round(float(trade['price']), 2)
                    trade_key = (trade_dt, t, trade_act, trade_q, trade_p)

                    if existing_trades_counter[trade_key] > 0:
                        existing_trades_counter[trade_key] -= 1
                        skipped_count += 1
                        continue

                    if trade_act == 'BUY':
                        fees_buy = get_trade_friction(t, trade_q, trade_p, action='BUY', asset_class=trade['asset_class'])
                        cursor.execute("""
                            INSERT INTO trade_ledger (execution_date, ticker, action, quantity, price, realized_pnl, gain_type, fees_estimated)
                            VALUES (?, ?, 'BUY', ?, ?, 0.0, 'N/A', ?)
                        """, [trade_dt, t, float(trade_q), float(trade_p), float(fees_buy)])
                        
                        cursor.execute("""
                            INSERT INTO tax_lots (ticker, asset_name, asset_class, buy_date, quantity, buy_price, last_split_date)
                            VALUES (?, ?, ?, ?, ?, ?, '')
                        """, [t, trade['asset_name'], trade['asset_class'], trade_dt, float(trade_q), float(trade_p)])
                        new_lot_id = cursor.lastrowid

                        ticker_active_lots.append({
                            'lot_id': new_lot_id,
                            'ticker': t,
                            'asset_name': trade['asset_name'],
                            'asset_class': trade['asset_class'],
                            'buy_date': trade_dt,
                            'quantity': float(trade_q),
                            'buy_price': float(trade_p),
                            'last_split_date': '',
                            'is_existing_db': True
                        })
                        imported_count += 1
                        total_invested += round(trade_q * trade_p, 2)

                    elif trade_act == 'SELL':
                        sell_rem = float(trade_q)
                        sell_price = float(trade_p)
                        sell_date = trade_dt

                        for lot in ticker_active_lots:
                            if sell_rem <= 1e-6:
                                break
                            if lot['quantity'] > 1e-6:
                                deduct = min(lot['quantity'], sell_rem)
                                lot['quantity'] -= deduct
                                sell_rem -= deduct

                                b_price = float(lot['buy_price'])
                                b_date = lot['buy_date']
                                matched_pnl = round((sell_price - b_price) * deduct, 2)

                                days_held = (pd.to_datetime(sell_date) - pd.to_datetime(b_date)).days
                                is_50aa = (t == SOVEREIGN_BOND_TICKER) or ('50AA' in str(lot.get('asset_class', ''))) or ('Debt' in str(lot.get('asset_class', '')))

                                if is_50aa:
                                    gain_type = 'Section 50AA (Debt STCG)'
                                elif days_held >= 365:
                                    gain_type = 'LTCG'
                                else:
                                    gain_type = 'STCG'

                                est_fees = get_trade_friction(t, deduct, sell_price, action='SELL', asset_class=lot.get('asset_class', ''))
                                cursor.execute("""
                                    INSERT INTO trade_ledger (execution_date, ticker, action, quantity, price, realized_pnl, gain_type, fees_estimated)
                                    VALUES (?, ?, 'SELL', ?, ?, ?, ?, ?)
                                """, [sell_date, t, float(deduct), float(sell_price), float(matched_pnl), gain_type, float(est_fees)])
                                realized_sells_count += 1

                                # Update or prune the active lot in SQLite
                                if lot.get('lot_id'):
                                    if lot['quantity'] <= 1e-6:
                                        cursor.execute("DELETE FROM tax_lots WHERE lot_id = ?", [lot['lot_id']])
                                    else:
                                        cursor.execute("UPDATE tax_lots SET quantity = ? WHERE lot_id = ?", [float(lot['quantity']), lot['lot_id']])

                        if sell_rem > 1e-6:
                            unmatched_sells_count += 1

            active_lots_count = sum(1 for lots in existing_lots_by_ticker.values() for l in lots if l['quantity'] > 1e-6)

        conn.commit()
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': f"Transaction failed and was rolled back: {str(e)}"}

    return {
        'success': True,
        'broker': broker_name,
        'mode': mode,
        'imported_count': imported_count,
        'skipped_count': skipped_count,
        'total_invested': round(total_invested, 2),
        'active_lots_count': active_lots_count,
        'realized_sells_count': realized_sells_count,
        'unmatched_sells_count': unmatched_sells_count
    }

# ----------------------------------------------------------------------------------------------------
# 5. XIRR CASH FLOW COMPILER
# ----------------------------------------------------------------------------------------------------
def compile_xirr_cash_flows(conn, terminal_valuation, xirr_solver_func=None):
    """
    Compiles chronological cash flow ledger from SQLite:
      - BUY trades: Outflow = -(quantity * price + fees_estimated)
      - SELL trades: Inflow = +(quantity * price - fees_estimated)
      - Direct initial tax lots (if trade_ledger is empty)
      - Terminal Valuation today: Inflow = +terminal_valuation on dynamically resolved TODAY_ISO
    """
    _horizons = get_market_time_horizons()
    _today_iso = _horizons['TODAY_ISO']
    _today_str = _horizons['TODAY_STR']

    ledger_df = clean_trade_ledger_df(pd.read_sql("SELECT * FROM trade_ledger ORDER BY execution_date ASC, trade_id ASC", conn))
    tax_lots_df = clean_tax_lots_df(pd.read_sql("SELECT * FROM tax_lots WHERE quantity > 0 ORDER BY buy_date ASC", conn))
    
    entries = []
    
    if not ledger_df.empty:
        for _, row in ledger_df.iterrows():
            dt = pd.to_datetime(row['execution_date']).strftime('%Y-%m-%d')
            act = str(row['action']).upper()
            q = float(row['quantity'])
            p = float(row['price'])
            fees = float(row.get('fees_estimated', 0.0))
            if 'BUY' in act:
                cf = -(q * p + fees)
                desc = f"BUY {q:g} {row['ticker']}"
            else:
                cf = +(q * p - fees)
                desc = f"SELL {q:g} {row['ticker']}"
            entries.append({
                'Date': dt,
                'Type': act,
                'Description': desc,
                'Cash Flow (₹)': round(cf, 2)
            })
    elif not tax_lots_df.empty:
        for _, lot in tax_lots_df.iterrows():
            dt = pd.to_datetime(lot['buy_date']).strftime('%Y-%m-%d')
            q = float(lot['quantity'])
            p = float(lot['buy_price'])
            fees = get_trade_friction(lot['ticker'], q, p, action='BUY', asset_class=lot.get('asset_class', ''))
            cf = -(q * p + fees)
            entries.append({
                'Date': dt,
                'Type': 'BUY',
                'Description': f"PORTFOLIO LOT: BUY {q:g} {lot['ticker']}",
                'Cash Flow (₹)': round(cf, 2)
            })
            
    if terminal_valuation > 0 and entries:
        entries.append({
            'Date': _today_iso,
            'Type': 'VALUATION',
            'Description': f"TERMINAL VALUATION (Current Holding Value on {_today_str})",
            'Cash Flow (₹)': round(float(terminal_valuation), 2)
        })
        
    cf_df = pd.DataFrame(entries)
    if not cf_df.empty:
        cf_df['Cumulative Net Invested (₹)'] = np.where(cf_df['Type'] == 'VALUATION', np.nan, -cf_df['Cash Flow (₹)'].cumsum())
        
    dates = [e['Date'] for e in entries]
    cfs = [e['Cash Flow (₹)'] for e in entries]
    
    if xirr_solver_func is not None:
        xirr_float, xirr_text = xirr_solver_func(cfs, dates)
    else:
        xirr_float, xirr_text = np.nan, "N/A"
    
    return {
        'cf_df': cf_df,
        'xirr_float': xirr_float,
        'xirr_text': xirr_text,
        'dates': dates,
        'cfs': cfs
    }
