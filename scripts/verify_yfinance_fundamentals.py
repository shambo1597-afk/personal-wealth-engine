#!/usr/bin/env python3
"""
Diagnostic tool: checks what yfinance actually returns for NSE fundamentals
(balance sheet, income statement, cash flow) so a real DuPont ROE / Piotroski
F-Score provider can be built against real field names instead of guesses.

This exists because Claude auditing this repo has no network access to
Yahoo Finance's endpoints from its sandbox (confirmed blocked, same as
eodhd.com, screener.in, and niftyindices.com) -- every field name in a
would-be yfinance-based fundamentals provider would otherwise be an
unverified guess, exactly the mistake that had to be fixed for the EODHD
path. Run this locally (yfinance needs no API key) and share the output.

Usage:
    python scripts/verify_yfinance_fundamentals.py RELIANCE.NS TCS.NS HDFCBANK.NS
    python scripts/verify_yfinance_fundamentals.py   # uses a small default sample
"""
import sys

import yfinance as yf

DEFAULT_SAMPLE = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "TATASTEEL.NS"]

# What a DuPont ROE / Piotroski F-Score computation needs, and the row-label
# keywords to search for in yfinance's statement DataFrames (case-insensitive
# substring match) -- reported as candidates, not asserted matches, since the
# exact label varies by yfinance version and hasn't been verified live here.
WANTED = {
    "Total Assets":               ["total assets"],
    "Total Equity":               ["stockholders equity", "total equity"],
    "Total Debt":                 ["total debt"],
    "Current Assets":             ["current assets"],
    "Current Liabilities":        ["current liabilities"],
    "Total Revenue":              ["total revenue", "operating revenue"],
    "Net Income":                 ["net income"],
    "Gross Profit":               ["gross profit"],
    "Operating Cash Flow":        ["operating cash flow", "cash flow from continuing operating"],
    "Shares Outstanding (basic)": ["basic average shares", "ordinary shares number"],
}


def find_candidates(index_labels, keywords):
    lower_map = {str(lbl).lower(): lbl for lbl in index_labels}
    hits = []
    for lower_lbl, orig_lbl in lower_map.items():
        if any(kw in lower_lbl for kw in keywords):
            hits.append(orig_lbl)
    return hits


def report_statement(name, df):
    print(f"  {name}:")
    if df is None or df.empty:
        print("    EMPTY or unavailable.")
        return None
    print(f"    Periods available: {list(df.columns)}")
    print(f"    Row labels ({len(df.index)}): {list(df.index)}")
    return df


def main():
    tickers = sys.argv[1:] or DEFAULT_SAMPLE

    for tk in tickers:
        print(f"=== {tk} ===")
        t = yf.Ticker(tk)

        try:
            info = t.info
        except Exception as e:
            info = {}
            print(f"  .info fetch failed: {e}")
        pe = info.get("trailingPE") or info.get("forwardPE")
        print(f"  P/E from .info: {pe!r} (trailingPE={info.get('trailingPE')!r}, forwardPE={info.get('forwardPE')!r})")

        try:
            bs = t.balance_sheet
        except Exception as e:
            bs = None
            print(f"  .balance_sheet fetch failed: {e}")
        try:
            inc = t.income_stmt if hasattr(t, "income_stmt") else t.financials
        except Exception as e:
            inc = None
            print(f"  income statement fetch failed: {e}")
        try:
            cf = t.cashflow if hasattr(t, "cashflow") else t.cash_flow
        except Exception as e:
            cf = None
            print(f"  cash flow fetch failed: {e}")

        bs = report_statement("Balance Sheet", bs)
        inc = report_statement("Income Statement", inc)
        cf = report_statement("Cash Flow", cf)
        print()

        print("  --- Field candidates found (substring match, not exact) ---")
        for wanted_name, keywords in WANTED.items():
            source = bs if "Assets" in wanted_name or "Equity" in wanted_name or "Debt" in wanted_name or "Liabilities" in wanted_name else (
                cf if "Cash Flow" in wanted_name else inc
            )
            if source is None:
                print(f"    {wanted_name:28} -- statement unavailable")
                continue
            hits = find_candidates(source.index, keywords)
            status = ", ".join(hits) if hits else "NOT FOUND"
            print(f"    {wanted_name:28} -> {status}")
        print("\n" + "=" * 70 + "\n")

    print("Send this full output back so a real yfinance-based fundamentals provider")
    print("can be built against the actual row labels shown above, per ticker.")


if __name__ == "__main__":
    main()
