#!/usr/bin/env python3
"""
Diagnostic tool: figures out why fetch_structured_company_fundamentals() in
quant_engine.py can't reach real EODHD data for an NSE ticker, and (once it
can) checks that the field names the parsing code assumes actually exist in
the live response.

This exists because the code that parses EODHD payloads was written and
tested against synthetic mocks -- Claude auditing this repo has no network
access to eodhd.com to verify any of this against a live account. Run this
once before trusting the "Sync Audited Structured Financials" button.

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

# Every field fetch_structured_company_fundamentals() reads from the EODHD
# fundamentals payload, so we can report exactly which ones are present.
EXPECTED_FIELDS = [
    "Financials.Balance_Sheet.yearly",
    "Financials.Balance_Sheet.yearly.<latest>.totalAssets",
    "Financials.Balance_Sheet.yearly.<latest>.totalStockholderEquity",
    "Financials.Balance_Sheet.yearly.<latest>.shortLongTermDebtTotal",
    "Financials.Income_Statement.yearly",
    "Financials.Income_Statement.yearly.<latest>.totalRevenue",
    "Financials.Income_Statement.yearly.<latest>.netIncome",
    "Financials.Cash_Flow.yearly",
    "Financials.Cash_Flow.yearly.<latest>.totalCashFromOperatingActivities",
    "Highlights.PERatio",
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


def probe(label, url_no_key, url):
    """GETs a URL, printing status and returning (status_code, parsed_json_or_None)."""
    print(f"--- {label} ---")
    print(f"GET {url_no_key}")
    try:
        resp = requests.get(url, timeout=15.0)
    except requests.RequestException as e:
        print(f"Request failed: {e}\n")
        return None, None
    print(f"HTTP {resp.status_code}")
    if resp.status_code != 200:
        print(f"Body (first 300 chars): {resp.text[:300]}\n")
        return resp.status_code, None
    try:
        data = resp.json()
    except ValueError:
        print(f"Non-JSON body (first 300 chars): {resp.text[:300]}\n")
        return resp.status_code, None
    print("OK\n")
    return resp.status_code, data


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("symbol", help="NSE symbol without the .NS/.NSE suffix, e.g. RELIANCE")
    parser.add_argument("--key", default=None, help="EODHD API key (defaults to $EODHD_API_KEY)")
    parser.add_argument("--exchange", default="NSE", help="Exchange suffix to try (default: NSE)")
    args = parser.parse_args()

    api_key = args.key or os.environ.get("EODHD_API_KEY")
    if not api_key:
        print("No API key given -- pass --key or set EODHD_API_KEY.", file=sys.stderr)
        sys.exit(1)

    ticker = f"{args.symbol}.{args.exchange}"

    # Probe 1: the fundamentals endpoint this app actually uses.
    fund_url = f"https://eodhd.com/api/fundamentals/{ticker}?api_token={api_key}&fmt=json"
    fund_url_masked = f"https://eodhd.com/api/fundamentals/{ticker}?api_token=***&fmt=json"
    fund_status, fund_data = probe("1. Fundamentals endpoint (what the app calls)", fund_url_masked, fund_url)

    if fund_status == 200 and fund_data:
        inc = ((fund_data.get("Financials") or {}).get("Income_Statement") or {}).get("yearly") or {}
        latest_key = sorted(inc.keys())[-1] if inc else None
        print(f"Latest annual period found: {latest_key!r}")
        print(f"Annual periods available: {len(inc)} "
              f"{'(>= 2, YoY criteria can be computed)' if len(inc) >= 2 else '(< 2, YoY criteria will fail closed)'}\n")

        all_ok = True
        for field_path in EXPECTED_FIELDS:
            parts = field_path.split(".")
            value, found = get_path(fund_data, parts, latest_key=latest_key)
            status = "OK" if found else "MISSING"
            if not found:
                all_ok = False
            display_val = value if not isinstance(value, dict) else f"<dict with {len(value)} keys>"
            print(f"  [{status:7}] {field_path:70} {display_val if found else ''}")
        print()
        if all_ok:
            print("RESULT: All assumed fields were found -- the code's field mapping matches this response.")
        else:
            print("RESULT: Some fields were MISSING -- quant_engine.py's field names need correcting to")
            print("match the schema shown above. Send this output so the parsing code can be fixed.")
        print("\nRaw top-level keys in the response:", list(fund_data.keys()))
        return

    # Fundamentals failed -- run diagnostic probes to tell a coverage/plan
    # problem apart from a symbol/format problem, rather than just failing.
    print("Fundamentals lookup failed. Running diagnostics to narrow down why...\n")

    # Probe 2: does the EOD *price* endpoint (a different EODHD product)
    # recognize this same ticker? If yes, the symbol/exchange format is
    # right and this is specifically a Fundamentals-coverage/plan issue.
    eod_url = f"https://eodhd.com/api/eod/{ticker}?api_token={api_key}&fmt=json&period=d&order=d"
    eod_url_masked = f"https://eodhd.com/api/eod/{ticker}?api_token=***&fmt=json&period=d&order=d"
    eod_status, eod_data = probe("2. EOD price endpoint (different product, same ticker)", eod_url_masked, eod_url)

    # Probe 3: does Fundamentals work at all on this key/plan, for a
    # well-covered US ticker? If yes, the key/plan supports Fundamentals in
    # general and this is specifically about NSE/India coverage.
    us_url = f"https://eodhd.com/api/fundamentals/AAPL.US?api_token={api_key}&fmt=json"
    us_url_masked = "https://eodhd.com/api/fundamentals/AAPL.US?api_token=***&fmt=json"
    us_status, _ = probe("3. Fundamentals endpoint for AAPL.US (sanity check on the key/plan)", us_url_masked, us_url)

    # Probe 4: what does EODHD's own NSE symbol directory say this ticker's
    # code actually is? Authoritative rather than guessing at suffixes.
    list_url = f"https://eodhd.com/api/exchange-symbol-list/{args.exchange}?api_token={api_key}&fmt=json"
    list_url_masked = f"https://eodhd.com/api/exchange-symbol-list/{args.exchange}?api_token=***&fmt=json"
    list_status, list_data = probe(f"4. {args.exchange} exchange symbol directory (authoritative ticker codes)", list_url_masked, list_url)
    if list_status == 200 and isinstance(list_data, list):
        matches = [row for row in list_data if str(row.get("Code", "")).upper() == args.symbol.upper()]
        if matches:
            print(f"Found {args.symbol!r} in the {args.exchange} directory: {matches[0]}\n")
        else:
            print(f"{args.symbol!r} was NOT found in the {args.exchange} directory "
                  f"({len(list_data)} symbols listed). Check spelling, or this exchange code may be wrong.\n")

    print("=== DIAGNOSIS ===")
    if eod_status is None or us_status is None or list_status is None:
        print("One or more probes never got an HTTP response at all (network/proxy/DNS failure, not a")
        print("4xx from EODHD) -- the reasoning below only applies to probes that got a real HTTP status;")
        print("re-run this on a connection that can actually reach eodhd.com.")
    if eod_status == 200 and us_status == 200:
        print("EOD prices work for this exact ticker, and Fundamentals works for a US ticker on this key.")
        print("=> Your plan's Fundamentals product likely does NOT include NSE/India coverage.")
        print("   Check your EODHD account's Fundamentals coverage/add-ons at eodhd.com -- this is a")
        print("   billing/plan limitation, not a bug in this app's code.")
    elif eod_status is not None and eod_status != 200:
        print(f"Even the EOD price endpoint returned HTTP {eod_status} for {ticker!r}.")
        print("=> The ticker/exchange format itself is likely wrong for this account, or the key is invalid")
        print("   for this ticker's exchange. Check probe 4's directory results above for the real code.")
    elif us_status is not None and us_status != 200:
        print(f"Fundamentals failed even for AAPL.US (HTTP {us_status}).")
        print("=> Your plan/key may not include Fundamentals data at all -- check your EODHD subscription.")
    else:
        print("Inconclusive from these probes alone -- inspect the raw responses above.")


if __name__ == "__main__":
    main()
