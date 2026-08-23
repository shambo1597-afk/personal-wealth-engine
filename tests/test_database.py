"""
Unit tests for database.py — FIFO tax-lot deduction, fractional-quantity precision,
and same-day broker CSV trade consolidation.
"""
from typing import Any, Dict

import pytest

ZERODHA_CSV_HEADER = "trade_date,symbol,trade_type,quantity,price"


def _import(isolated_db, csv_text: str, mode: str = "Full Ledger Reset & Rebuild") -> Dict[str, Any]:
    conn = isolated_db.get_db_connection()
    try:
        result = isolated_db.parse_and_import_broker_csv(csv_text.encode("utf-8"), conn, mode=mode)
    finally:
        conn.close()
    return result


class TestFIFOLotDeduction:
    def test_sell_consumes_oldest_lot_first(self, isolated_db):
        csv_text = f"""{ZERODHA_CSV_HEADER}
2023-01-01,TEST,buy,10.0,100.0
2023-06-01,TEST,buy,10.0,110.0
2024-07-01,TEST,sell,5.0,150.0
"""
        result = _import(isolated_db, csv_text)
        assert result["success"] is True
        assert result["realized_sells_count"] == 1
        assert result["unmatched_sells_count"] == 0

        conn = isolated_db.get_db_connection()
        rows = conn.execute(
            "SELECT ticker, quantity, buy_price FROM tax_lots ORDER BY buy_date ASC"
        ).fetchall()
        conn.close()

        # The 5.0 sale must be deducted from the oldest (2023-01-01 @ 100) lot first,
        # leaving 5.0 of that lot and the full 10.0 of the newer lot untouched.
        assert len(rows) == 2
        assert rows[0][1] == pytest.approx(5.0)
        assert rows[0][2] == pytest.approx(100.0)
        assert rows[1][1] == pytest.approx(10.0)
        assert rows[1][2] == pytest.approx(110.0)

    def test_sell_spanning_two_lots_splits_correctly(self, isolated_db):
        csv_text = f"""{ZERODHA_CSV_HEADER}
2023-01-01,TEST,buy,10.0,100.0
2023-06-01,TEST,buy,10.0,110.0
2024-07-01,TEST,sell,15.0,150.0
"""
        result = _import(isolated_db, csv_text)
        assert result["realized_sells_count"] == 2  # matched across both lots

        conn = isolated_db.get_db_connection()
        remaining = conn.execute("SELECT quantity, buy_price FROM tax_lots").fetchall()
        sells = conn.execute(
            "SELECT quantity, price, realized_pnl FROM trade_ledger WHERE action = 'SELL' ORDER BY trade_id"
        ).fetchall()
        conn.close()

        assert len(remaining) == 1
        assert remaining[0][0] == pytest.approx(5.0)  # 10.0 - 5.0 left of the second lot
        assert remaining[0][1] == pytest.approx(110.0)

        assert len(sells) == 2
        assert sells[0][0] == pytest.approx(10.0)  # first lot fully consumed
        assert sells[1][0] == pytest.approx(5.0)   # remainder from second lot

    def test_fractional_quantity_deduction_has_no_floating_point_drift(self, isolated_db):
        # 10.3333 + 5.6667 bought, 12.0 sold -> exactly 4.0 must remain, not 3.999999999999999
        csv_text = f"""{ZERODHA_CSV_HEADER}
2023-01-01,TEST,buy,10.3333,100.0
2023-06-01,TEST,buy,5.6667,110.0
2024-07-01,TEST,sell,12.0,150.0
"""
        _import(isolated_db, csv_text)

        conn = isolated_db.get_db_connection()
        qty = conn.execute("SELECT quantity FROM tax_lots").fetchone()[0]
        conn.close()

        assert qty == 4.0  # exact equality, not just approx -- guards against float drift
        assert round(qty, 10) == round(qty, 4)

    def test_ltcg_vs_stcg_classification_by_holding_period(self, isolated_db):
        csv_text = f"""{ZERODHA_CSV_HEADER}
2023-01-01,TEST,buy,10.0,100.0
2024-07-01,TEST,sell,10.0,150.0
"""
        _import(isolated_db, csv_text)
        conn = isolated_db.get_db_connection()
        gain_type = conn.execute(
            "SELECT gain_type FROM trade_ledger WHERE action = 'SELL'"
        ).fetchone()[0]
        conn.close()
        assert gain_type == "LTCG"  # 2023-01-01 -> 2024-07-01 is well over 365 days

    def test_incremental_merge_preserves_fifo_precision_across_partial_sells(self, isolated_db):
        _import(isolated_db, f"""{ZERODHA_CSV_HEADER}
2023-01-01,TEST,buy,10.3333,100.0
2023-06-01,TEST,buy,5.6667,110.0
2024-07-01,TEST,sell,12.0,150.0
""", mode="Full Ledger Reset & Rebuild")

        result = _import(isolated_db, f"""{ZERODHA_CSV_HEADER}
2024-09-01,TEST,sell,1.5,160.0
""", mode="Merge New Trades (Incremental)")
        assert result["success"] is True

        conn = isolated_db.get_db_connection()
        qty = conn.execute("SELECT quantity FROM tax_lots").fetchone()[0]
        conn.close()
        assert qty == 2.5  # 4.0 - 1.5, exact


class TestSameDayTradeConsolidation:
    def test_two_same_day_buys_consolidate_into_one_weighted_average_lot(self, isolated_db):
        csv_text = f"""{ZERODHA_CSV_HEADER}
2024-01-15,TEST,buy,10.0,100.0
2024-01-15,TEST,buy,20.0,106.0
"""
        result = _import(isolated_db, csv_text)
        assert result["imported_count"] == 1  # consolidated into a single lot

        conn = isolated_db.get_db_connection()
        buy_rows = conn.execute(
            "SELECT quantity, price FROM trade_ledger WHERE action = 'BUY'"
        ).fetchall()
        lot_rows = conn.execute("SELECT quantity, buy_price FROM tax_lots").fetchall()
        conn.close()

        # weighted avg price = (10*100 + 20*106) / 30 = 104.0
        assert len(buy_rows) == 1
        assert buy_rows[0][0] == pytest.approx(30.0)
        assert buy_rows[0][1] == pytest.approx(104.0)

        assert len(lot_rows) == 1
        assert lot_rows[0][0] == pytest.approx(30.0)
        assert lot_rows[0][1] == pytest.approx(104.0)

    def test_buys_on_different_days_are_not_consolidated(self, isolated_db):
        csv_text = f"""{ZERODHA_CSV_HEADER}
2024-01-15,TEST,buy,10.0,100.0
2024-01-16,TEST,buy,20.0,106.0
"""
        result = _import(isolated_db, csv_text)
        assert result["imported_count"] == 2

        conn = isolated_db.get_db_connection()
        lot_rows = conn.execute("SELECT quantity FROM tax_lots ORDER BY buy_date").fetchall()
        conn.close()
        assert len(lot_rows) == 2

    def test_same_day_sells_are_not_consolidated(self, isolated_db):
        csv_text = f"""{ZERODHA_CSV_HEADER}
2023-01-01,TEST,buy,30.0,100.0
2024-07-01,TEST,sell,5.0,150.0
2024-07-01,TEST,sell,7.0,155.0
"""
        result = _import(isolated_db, csv_text)
        assert result["realized_sells_count"] == 2

        conn = isolated_db.get_db_connection()
        sell_rows = conn.execute(
            "SELECT quantity, price FROM trade_ledger WHERE action = 'SELL' ORDER BY trade_id"
        ).fetchall()
        conn.close()
        assert len(sell_rows) == 2
        assert {round(r[0], 4) for r in sell_rows} == {5.0, 7.0}


class TestKiteHoldingsSync:
    def test_sync_kite_holdings_inserts_tax_lots(self, isolated_db):
        conn = isolated_db.get_db_connection()
        try:
            holdings = [
                {'ticker': 'TCS.NS', 'quantity': 15, 'buy_price': 3450.0},
                {'ticker': 'INFY.NS', 'quantity': 40, 'buy_price': 1480.0},
            ]
            res = isolated_db.sync_kite_holdings_to_tax_lots(holdings, conn, mode='Full Ledger Reset & Rebuild')
            assert res['success'] is True
            assert res['imported_count'] == 2

            lots = conn.execute("SELECT ticker, quantity, buy_price FROM tax_lots ORDER BY ticker").fetchall()
            assert len(lots) == 2
            assert lots[0][0] == 'INFY.NS'
            assert lots[0][1] == 40.0
            assert lots[0][2] == 1480.0
            assert lots[1][0] == 'TCS.NS'
            assert lots[1][1] == 15.0
            assert lots[1][2] == 3450.0
        finally:
            conn.close()

    def test_sync_kite_holdings_empty_returns_cleanly(self, isolated_db):
        conn = isolated_db.get_db_connection()
        try:
            res = isolated_db.sync_kite_holdings_to_tax_lots([], conn)
            assert res['success'] is True
            assert res['imported_count'] == 0
        finally:
            conn.close()
