"""
Unit tests for tax_engine.py -- Section 112A LTCG, Section 111A STCG, Section 50AA Debt
ETF slab classification, Section 55(2)(ac) pre-2018 grandfathering, and the Section 70/74
intra-head loss set-off ordering.
"""
from typing import Any, Dict, List

import pandas as pd
import pytest

from config import SOVEREIGN_BOND_TICKER
from tax_engine import (
    SEC_111A_TAX_RATE,
    SEC_112A_EXEMPTION_LIMIT,
    SEC_112A_TAX_RATE,
    SEC_50AA_TAX_RATE,
    compute_grandfathered_cost_basis,
    compute_realized_tax_summary,
)


def _sells_df(records: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(records)


class TestSection112ALTCGExemptionAndRate:
    def test_statutory_rate_constants(self):
        # 12.5% base + 4% Health & Education Cess = 13.0% effective
        assert SEC_112A_EXEMPTION_LIMIT == pytest.approx(125000.0)
        assert SEC_112A_TAX_RATE == pytest.approx(0.130)

    def test_no_tax_at_or_below_1_25l_exemption(self):
        df = _sells_df([{"ticker": "TCS.NS", "gain_type": "LTCG", "realized_pnl": 125000.0}])
        res = compute_realized_tax_summary(df)
        assert res["taxable_112a_amt"] == pytest.approx(0.0)
        assert res["tax_payable_112a"] == pytest.approx(0.0)

    def test_tax_charged_only_on_excess_above_exemption(self):
        df = _sells_df([{"ticker": "TCS.NS", "gain_type": "LTCG", "realized_pnl": 225000.0}])
        res = compute_realized_tax_summary(df)
        assert res["taxable_112a_amt"] == pytest.approx(100000.0)
        assert res["tax_payable_112a"] == pytest.approx(100000.0 * 0.130)

    def test_pure_ltcg_loss_produces_no_tax(self):
        df = _sells_df([{"ticker": "TCS.NS", "gain_type": "LTCG", "realized_pnl": -50000.0}])
        res = compute_realized_tax_summary(df)
        assert res["tax_payable_112a"] == pytest.approx(0.0)
        assert res["unabsorbed_ltcl"] == pytest.approx(50000.0)


class TestSection111ASTCG:
    def test_statutory_rate_constant(self):
        assert SEC_111A_TAX_RATE == pytest.approx(0.208)  # 20.0% base + 4% cess

    def test_flat_rate_applied_with_no_exemption(self):
        df = _sells_df([{"ticker": "INFY.NS", "gain_type": "STCG", "realized_pnl": 100000.0}])
        res = compute_realized_tax_summary(df)
        assert res["tax_payable_111a"] == pytest.approx(100000.0 * SEC_111A_TAX_RATE)


class TestSection50AADebtETFSlabClassification:
    def test_statutory_rate_constant_is_highest_bracket(self):
        assert SEC_50AA_TAX_RATE == pytest.approx(0.312)  # 30.0% slab + 4% cess
        assert SEC_50AA_TAX_RATE > SEC_111A_TAX_RATE > SEC_112A_TAX_RATE

    def test_debt_etf_gain_is_isolated_from_equity_111a_bucket(self):
        df = _sells_df([
            {"ticker": SOVEREIGN_BOND_TICKER, "gain_type": "Section 50AA (Debt STCG)", "realized_pnl": 40000.0},
        ])
        res = compute_realized_tax_summary(df)
        assert res["net_50aa_stcg"] == pytest.approx(40000.0)
        assert res["gross_111a_pnl"] == pytest.approx(0.0)
        assert res["tax_payable_111a"] == pytest.approx(0.0)

    def test_debt_etf_classified_by_ticker_even_without_50aa_in_gain_type(self):
        # Belt-and-braces: the sovereign bond ticker alone routes into the 50AA bucket.
        df = _sells_df([{"ticker": SOVEREIGN_BOND_TICKER, "gain_type": "STCG", "realized_pnl": 10000.0}])
        res = compute_realized_tax_summary(df)
        assert res["net_50aa_stcg"] == pytest.approx(10000.0)
        assert res["gross_111a_pnl"] == pytest.approx(0.0)


class TestSection70And74LossSetOffOrdering:
    def test_stcl_from_debt_bucket_offsets_positive_equity_stcg(self):
        """A net loss in the Section 50AA debt bucket is available Section 70(2) STCL and
        must reduce a positive net equity Section 111A STCG bucket."""
        df = _sells_df([
            {"ticker": SOVEREIGN_BOND_TICKER, "gain_type": "Section 50AA (Debt STCG)", "realized_pnl": -15000.0},
            {"ticker": "TCS.NS", "gain_type": "STCG", "realized_pnl": 50000.0},
        ])
        res = compute_realized_tax_summary(df)
        assert res["post_setoff_stcg"] == pytest.approx(35000.0)
        assert res["unabsorbed_stcl"] == pytest.approx(0.0)

    def test_stcl_from_equity_bucket_offsets_positive_debt_50aa_stcg_first(self):
        """A net loss in the equity STCG bucket is Section 70(2) STCL and is applied to the
        Section 50AA debt STCG bucket (the higher effective tax rate) before anything else."""
        df = _sells_df([
            {"ticker": "INFY.NS", "gain_type": "STCG", "realized_pnl": -40000.0},
            {"ticker": SOVEREIGN_BOND_TICKER, "gain_type": "Section 50AA (Debt STCG)", "realized_pnl": 25000.0},
        ])
        res = compute_realized_tax_summary(df)
        assert res["post_setoff_50aa"] == pytest.approx(0.0)
        assert res["stcl_used_against_50aa"] == pytest.approx(25000.0)
        assert res["unabsorbed_stcl"] == pytest.approx(15000.0)  # 40000 - 25000 left over

    def test_stcl_offsets_ltcg_only_to_the_extent_ltcg_exceeds_exemption(self):
        df = _sells_df([
            {"ticker": "INFY.NS", "gain_type": "STCG", "realized_pnl": -50000.0},
            {"ticker": "TCS.NS", "gain_type": "LTCG", "realized_pnl": 200000.0},
        ])
        res = compute_realized_tax_summary(df)
        # taxable LTCG before STCL = 200000 - 125000 = 75000; the 50000 STCL is fully absorbed there
        assert res["stcl_used_against_ltcg"] == pytest.approx(50000.0)
        assert res["post_setoff_ltcg"] == pytest.approx(150000.0)
        assert res["unabsorbed_stcl"] == pytest.approx(0.0)

    def test_stcl_is_not_wasted_against_already_exempt_ltcg(self):
        df = _sells_df([
            {"ticker": "INFY.NS", "gain_type": "STCG", "realized_pnl": -10000.0},
            {"ticker": "TCS.NS", "gain_type": "LTCG", "realized_pnl": 100000.0},  # fully within exemption
        ])
        res = compute_realized_tax_summary(df)
        assert res["stcl_used_against_ltcg"] == pytest.approx(0.0)
        assert res["unabsorbed_stcl"] == pytest.approx(10000.0)

    def test_ltcl_can_only_offset_ltcg_never_stcg_or_debt(self):
        df = _sells_df([
            {"ticker": "TCS.NS", "gain_type": "LTCG", "realized_pnl": -40000.0},
            {"ticker": "INFY.NS", "gain_type": "STCG", "realized_pnl": 60000.0},
            {"ticker": SOVEREIGN_BOND_TICKER, "gain_type": "Section 50AA (Debt STCG)", "realized_pnl": 15000.0},
        ])
        res = compute_realized_tax_summary(df)
        assert res["unabsorbed_ltcl"] == pytest.approx(40000.0)
        assert res["post_setoff_stcg"] == pytest.approx(60000.0)
        assert res["post_setoff_50aa"] == pytest.approx(15000.0)
        assert res["tax_payable_111a"] == pytest.approx(60000.0 * SEC_111A_TAX_RATE)

    def test_empty_ledger_returns_all_zero_summary(self):
        res = compute_realized_tax_summary(pd.DataFrame())
        assert res["tax_payable_112a"] == pytest.approx(0.0)
        assert res["tax_payable_111a"] == pytest.approx(0.0)
        assert res["unabsorbed_stcl"] == pytest.approx(0.0)
        assert res["unabsorbed_ltcl"] == pytest.approx(0.0)


class TestSection55PreGrandfathering:
    def test_deemed_cost_is_higher_of_buy_price_and_fmv_capped_at_sale_price(self):
        # Step 1: min(FMV, sale) = min(300, 500) = 300; Step 2: max(buy, 300) = max(100, 300) = 300
        deemed = compute_grandfathered_cost_basis("2015-01-01", buy_price=100.0, sale_or_live_price=500.0, fmv_31jan2018=300.0)
        assert deemed == pytest.approx(300.0)

    def test_fmv_floor_is_capped_at_sale_price_when_sale_is_lower(self):
        deemed = compute_grandfathered_cost_basis("2015-01-01", buy_price=100.0, sale_or_live_price=250.0, fmv_31jan2018=300.0)
        assert deemed == pytest.approx(250.0)

    def test_actual_buy_price_wins_when_above_the_fmv_floor(self):
        deemed = compute_grandfathered_cost_basis("2015-01-01", buy_price=350.0, sale_or_live_price=500.0, fmv_31jan2018=300.0)
        assert deemed == pytest.approx(350.0)

    def test_purchases_on_or_after_1_feb_2018_ignore_fmv_entirely(self):
        deemed = compute_grandfathered_cost_basis("2019-06-01", buy_price=100.0, sale_or_live_price=500.0, fmv_31jan2018=300.0)
        assert deemed == pytest.approx(100.0)

    def test_missing_fmv_falls_back_to_actual_buy_price(self):
        deemed = compute_grandfathered_cost_basis("2015-01-01", buy_price=100.0, sale_or_live_price=500.0, fmv_31jan2018=None)
        assert deemed == pytest.approx(100.0)

    def test_grandfathering_reduces_taxable_gain_in_realized_summary(self):
        # Bought pre-2018 at 50, FMV on 31-Jan-2018 was 200, sold now at 500 for 10 units.
        # Deemed cost = max(50, min(200, 500)) = 200 -> taxable gain = (500-200)*10 = 3000,
        # NOT the naive (500-50)*10 = 4500 the raw ledger P&L would suggest.
        df = _sells_df([{
            "ticker": "TCS.NS", "gain_type": "LTCG", "realized_pnl": 4500.0,
            "buy_date": "2016-01-01", "quantity": 10.0, "price": 500.0,
            "buy_price": 50.0, "fmv_31jan2018": 200.0,
        }])
        res = compute_realized_tax_summary(df)
        assert res["gross_112a_pnl"] == pytest.approx(3000.0)
