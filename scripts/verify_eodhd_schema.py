#!/usr/bin/env python3
"""
One-time (and re-runnable) verification tool: checks that the field names
fetch_structured_company_fundamentals() in quant_engine.py assumes actually
exist in a real EODHD fundamentals API response.

This exists because the code that parses EODHD payloads was written and
tested against synthetic mocks -- Claude auditing this repo never had
network access to eodhd.com to verify the assumed schema against a live
response. Run this once against a real ticker before trusting the
"Sync Audited Structured Financials" button in the app.

Usage:
    EODHD_API_KEY=your_key_here python scripts/verify_eodhd_schema.py RELIANCE
    python scripts/verify_eodhd_schema.py RELIANCE --key your_key_here

Never hardcode your key into this file or any file you commit -- pass it via
the environment variable or the --key flag, same as the app's own sidebar
input does.
"""
import argparse
import os
import sys

import requests

# (json_path, description) -- every field fetch_structured_company_fundamentals()
# reads from the EODHD payload, so we can report exactly which ones are missing.
EXPECTED_FIELDS = [
    ("Financials.Balance_Sheet.yearly", "balance sheet, keyed by period-end date"),
    ("Financials.Balance_Sheet.yearly.<latest>.totalAssets", None),
    ("Financials.Balance_Sheet.yearly.<latest>.totalStockholderEquity", None),
    ("Financials.Balance_Sheet.yearly.<latest>.shortLongTermDebtTotal", None),
    ("Financials.Income_Statement.yearly", "income statement, keyed by period-end date"),
    ("Financials.Income_Statement.yearly.<latest>.totalRevenue", None),
    ("Financials.Income_Statement.yearly.<latest>.netIncome", None),
    ("Financials.Cash_Flow.yearly", "cash flow statement, keyed by period-end date"),
    ("Financials.Cash_Flow.yearly.<latest>.totalCashFromOperatingActivities", None),
    ("Highlights.PERatio", "valuation multiple, used for the P/E Ratio column"),
]


def get_path(data, path_parts, latest_key=None):
    cur = data
    for part in path_parts:
        if part == "<latest>":
            if not isinstance(cur, dict) or not cur:
                return None, False
            part = latest_key or sorted(cur.keys())[-1]
        if not isinstance(cur, dict) or part not in cur:
            return None, False
        cur = cur[part]
    return cur, True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbol", help="NSE symbol without the .NS/.NSE suffix, e.g. RELIANCE")
    parser.add_argument("--key", default=None, help="EODHD API key (defaults to $EODHD_API_KEY)")
    args = parser.parse_args()

    api_key = args.key or os.environ.get("EODHD_API_KEY")
    if not api_key:
        print("No API key given -- pass --key or set EODHD_API_KEY.", file=sys.stderr)
        sys.exit(1)

    url = f"https://eodhd.com/api/fundamentals/{args.symbol}.NSE?api_token={api_key}&fmt=json"
    print(f"Requesting: https://eodhd.com/api/fundamentals/{args.symbol}.NSE?api_token=***&fmt=json")
    resp = requests.get(url, timeout=15.0)
    print(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        print("Non-200 response body (first 500 chars):", resp.text[:500])
        sys.exit(1)

    data = resp.json()

    inc = ((data.get("Financials") or {}).get("Income_Statement") or {}).get("yearly") or {}
    latest_key = sorted(inc.keys())[-1] if inc else None
    print(f"Latest annual period found: {latest_key!r}")
    print(f"Total annual periods available: {len(inc)} {'(>= 2, YoY Piotroski criteria can be computed)' if len(inc) >= 2 else '(< 2, YoY Piotroski criteria will fail closed)'}")
    print()

    all_ok = True
    for field_path, _desc in EXPECTED_FIELDS:
        parts = field_path.split(".")
        value, found = get_path(data, parts, latest_key=latest_key)
        status = "OK" if found else "MISSING"
        if not found:
            all_ok = False
        display_val = value if not isinstance(value, dict) else f"<dict with {len(value)} keys>"
        print(f"  [{status:7}] {field_path:70} {display_val if found else ''}")

    print()
    if all_ok:
        print("All assumed fields were found. The code's field mapping matches this response.")
    else:
        print("Some fields were MISSING. quant_engine.py's fetch_structured_company_fundamentals()")
        print("needs its field names corrected to match the real schema shown above -- report the")
        print("MISSING fields (and, ideally, a redacted dump of the real JSON structure around them)")
        print("so the parsing code can be fixed.")

    print()
    print("Raw top-level keys in the response:", list(data.keys()))


if __name__ == "__main__":
    main()
