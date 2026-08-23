"""
Unit tests for quant_engine.py — HRP tree clustering, the sector/asset-capped simplex
projection, Newton-Raphson XIRR convergence, and the final-mile Zerodha order basket
conversion (continuous weights -> discrete whole-share orders).
"""
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from config import MAX_RETAIL_CAP, MAX_SECTOR_CAP
from quant_engine import (
    compute_hrp_weights,
    project_weights_sector_capped,
    compute_portfolio_xirr,
    compute_zerodha_order_basket,
    compute_multifactor_rankings,
    fetch_universe_fundamentals,
    sync_screener_fundamentals_for_universe,
    get_fundamentals_last_synced,
    extract_screener_financials,
    compute_dupont_and_piotroski_from_financials,
    _build_fundamentals_row,
    _clean_screener_number,
    _parse_screener_statement_table,
    _parse_screener_top_ratios,
    _find_statement_row,
    _screener_symbol,
    solve_portfolio_in_memory,
)


def _synthetic_cov(n: int, seed: int = 0) -> np.ndarray:
    """A deterministic, well-conditioned positive-definite covariance matrix of size n."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.02, size=(n, n))
    cov = a @ a.T + np.eye(n) * 0.001
    return cov


class TestHRPTreeClustering:
    def test_weights_sum_to_one_and_are_non_negative(self):
        cov = _synthetic_cov(12)
        returns = pd.DataFrame(np.random.default_rng(1).normal(0.0005, 0.01, (300, 12)))
        w = compute_hrp_weights(returns, cov)
        assert w.shape[0] == 12
        assert np.all(w >= -1e-12)
        assert w.sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("n", [2, 5, 13, 25, 40])
    def test_no_leaf_index_out_of_bounds_across_sizes(self, n):
        """Every recursively-bisected leaf index must stay within [0, n) -- a regression
        guard for tree-traversal bugs that would otherwise raise IndexError or silently
        drop/duplicate assets."""
        cov = _synthetic_cov(n, seed=n)
        returns = pd.DataFrame(np.random.default_rng(n).normal(0.0005, 0.01, (250, n)))
        w = compute_hrp_weights(returns, cov)
        assert w.shape[0] == n
        assert np.isfinite(w).all()
        assert w.sum() == pytest.approx(1.0)

    def test_degenerate_single_and_empty_inputs(self):
        single = compute_hrp_weights(pd.DataFrame(np.zeros((10, 1))), np.array([[0.02]]))
        assert list(single) == pytest.approx([1.0])
        empty = compute_hrp_weights(pd.DataFrame(np.zeros((10, 0))), np.zeros((0, 0)))
        assert len(empty) == 0

    def test_zero_variance_asset_does_not_raise_or_produce_nan(self):
        """A constant-return (zero-variance) column must not trigger a divide-by-zero
        in the naive inverse-variance cluster weighting."""
        rng = np.random.default_rng(3)
        returns = pd.DataFrame(rng.normal(0.0005, 0.01, (200, 6)))
        returns[3] = 0.0  # zero-variance column
        cov = returns.cov().values * 252
        w = compute_hrp_weights(returns, cov)
        assert np.isfinite(w).all()
        assert w.sum() == pytest.approx(1.0)

    def test_perfectly_correlated_duplicate_assets_do_not_raise(self):
        rng = np.random.default_rng(4)
        returns = pd.DataFrame(rng.normal(0.0006, 0.012, (200, 4)))
        returns[3] = returns[0]  # exact duplicate -> zero distance
        cov = returns.cov().values * 252
        w = compute_hrp_weights(returns, cov)
        assert np.isfinite(w).all()
        assert w.sum() == pytest.approx(1.0)


class TestSectorCappedSimplexProjection:
    def _run(self, weights: List[float], tickers: List[str], sector_map: Dict[str, str]) -> np.ndarray:
        return project_weights_sector_capped(
            np.array(weights), tickers,
            max_asset_cap=MAX_RETAIL_CAP, max_sector_cap=MAX_SECTOR_CAP,
            sector_map=sector_map,
        )

    def test_single_asset_cap_strictly_enforced(self):
        tickers = [f"A{i}" for i in range(10)]
        sector_map = {t: ("SEC1" if i < 5 else "SEC2") for i, t in enumerate(tickers)}
        weights = [0.5, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        w = self._run(weights, tickers, sector_map)
        assert w.max() <= MAX_RETAIL_CAP + 1e-6

    def test_budget_conservation_and_long_only(self):
        tickers = [f"A{i}" for i in range(10)]
        sector_map = {t: ("SEC1" if i < 5 else "SEC2") for i, t in enumerate(tickers)}
        weights = [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0]
        w = self._run(weights, tickers, sector_map)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= -1e-9)

    def test_sector_cap_enforced_with_sufficient_sector_diversity(self):
        # 4 sectors of 5 assets each: 25% cap per sector is feasible (4 * 25% = 100%).
        tickers = [f"A{i}" for i in range(20)]
        sectors = ["FIN", "IT", "PHARMA", "AUTO"]
        sector_map = {t: sectors[i // 5] for i, t in enumerate(tickers)}
        # Concentrate almost everything into the FIN sector to force the projector to act.
        weights = [0.2] * 5 + [0.0] * 15
        w = self._run(weights, tickers, sector_map)
        sector_sums: Dict[str, float] = defaultdict(float)
        for t, wi in zip(tickers, w):
            sector_sums[sector_map[t]] += wi
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert w.max() <= MAX_RETAIL_CAP + 1e-6
        for s, total in sector_sums.items():
            assert total <= MAX_SECTOR_CAP + 1e-4, f"Sector {s} breached the 25% cap: {total}"

    def test_gracefully_relaxes_only_when_strictly_infeasible(self):
        # Only 2 sectors present: 25% + 25% = 50% < 100% budget, so a strict 25% cap on
        # both is mathematically infeasible. The projector must still return a valid
        # simplex (sum=1, long-only, asset cap respected) rather than raising or failing.
        tickers = [f"A{i}" for i in range(10)]
        sector_map = {t: ("SEC1" if i < 4 else "SEC2") for i, t in enumerate(tickers)}
        weights = [0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0, 0, 0]
        w = self._run(weights, tickers, sector_map)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert w.max() <= MAX_RETAIL_CAP + 1e-6
        assert np.all(w >= -1e-9)


class TestXIRRNewtonRaphsonConvergence:
    def test_single_year_round_trip_matches_closed_form_rate(self):
        dates = ["2023-01-01", "2024-01-01"]
        cash_flows = [-100000.0, 110000.0]
        rate, label = compute_portfolio_xirr(cash_flows, dates)
        assert rate == pytest.approx(0.10, abs=1e-4)
        assert "%" in label

    def test_two_year_compounding_matches_closed_form_rate(self):
        # -100000 grows to 100000 * 1.21 = 121000 over exactly 2 years at 10% p.a.
        dates = ["2022-01-01", "2024-01-01"]
        cash_flows = [-100000.0, 121000.0]
        rate, _ = compute_portfolio_xirr(cash_flows, dates)
        assert rate == pytest.approx(0.10, abs=1e-3)

    def test_multi_leg_sip_like_cash_flow_converges_to_positive_return(self):
        dates = ["2021-01-01", "2021-07-01", "2022-01-01", "2022-07-01", "2023-01-01"]
        cash_flows = [-50000.0, -50000.0, -50000.0, -50000.0, 240000.0]
        rate, label = compute_portfolio_xirr(cash_flows, dates)
        assert np.isfinite(rate)
        assert -0.999 < rate < 5.0
        assert rate > 0  # total redemption (240000) exceeds total invested (200000)
        assert "N/A" not in label

    def test_insufficient_trades_returns_not_applicable(self):
        rate, label = compute_portfolio_xirr([-100000.0], ["2023-01-01"])
        assert np.isnan(rate)
        assert "N/A" in label

    def test_all_cash_flows_same_day_uses_simple_return_branch(self):
        dates = ["2024-05-01", "2024-05-01"]
        cash_flows = [-100000.0, 105000.0]
        rate, label = compute_portfolio_xirr(cash_flows, dates)
        assert rate == pytest.approx(0.05, abs=1e-6)
        assert "Same-day" in label

    def test_all_negative_cash_flows_have_no_solvable_rate(self):
        rate, label = compute_portfolio_xirr([-1000.0, -2000.0], ["2023-01-01", "2023-06-01"])
        assert np.isnan(rate)
        assert "N/A" in label


class TestZerodhaOrderBasket:
    """
    This is the last function in the pipeline before a human clicks "buy" -- it turns
    continuous optimizer weights into whole-share order quantities. Every rupee here is
    real money, so it gets the same scrutiny as the optimizer and tax math.
    """

    def test_whole_share_rounding_never_overspends_allocated_budget(self):
        # 1.5% buffer -> deployable = 98500; weight 1.0 all into one stock @ price 301
        # -> floor(98500/301) = 327 shares, allocated = 327*301 = 98427
        basket_df, leftover = compute_zerodha_order_basket(
            target_weights={'TCS.NS': 1.0},
            latest_prices={'TCS.NS': 301.0},
            total_capital_inr=100000.0,
            cash_buffer_pct=0.015,
        )
        row = basket_df.iloc[0]
        assert row['Quantity'] == 327
        assert row['Allocated_INR'] == pytest.approx(327 * 301.0)
        # never buy more shares than the deployable (post-buffer) budget allows
        assert row['Allocated_INR'] <= 100000.0 * (1 - 0.015) + 1e-6
        assert leftover == pytest.approx(100000.0 - row['Allocated_INR'])

    def test_cash_buffer_is_reserved_and_shows_up_in_leftover(self):
        # A stock priced just above the deployable budget: zero shares bought, and the
        # entire capital (including the 1.5% buffer) must come back as leftover cash.
        basket_df, leftover = compute_zerodha_order_basket(
            target_weights={'EXPENSIVE.NS': 1.0},
            latest_prices={'EXPENSIVE.NS': 1_000_000.0},
            total_capital_inr=100000.0,
            cash_buffer_pct=0.015,
        )
        assert basket_df.iloc[0]['Quantity'] == 0
        assert leftover == pytest.approx(100000.0)

    def test_multi_ticker_allocation_is_proportional_to_weights(self):
        basket_df, leftover = compute_zerodha_order_basket(
            target_weights={'A': 0.6, 'B': 0.4},
            latest_prices={'A': 100.0, 'B': 50.0},
            total_capital_inr=100000.0,
            cash_buffer_pct=0.0,
        )
        row_a = basket_df[basket_df['Ticker'] == 'A'].iloc[0]
        row_b = basket_df[basket_df['Ticker'] == 'B'].iloc[0]
        # floor(60000/100)=600, floor(40000/50)=800
        assert row_a['Quantity'] == 600
        assert row_b['Quantity'] == 800
        assert leftover == pytest.approx(100000.0 - (600 * 100.0 + 800 * 50.0))

    def test_zero_or_negative_weight_gets_zero_shares(self):
        basket_df, _ = compute_zerodha_order_basket(
            target_weights={'A': 0.0, 'B': -0.1, 'C': 1.0},
            latest_prices={'A': 100.0, 'B': 100.0, 'C': 100.0},
            total_capital_inr=100000.0,
        )
        assert basket_df.set_index('Ticker').loc['A', 'Quantity'] == 0
        assert basket_df.set_index('Ticker').loc['B', 'Quantity'] == 0
        assert basket_df.set_index('Ticker').loc['C', 'Quantity'] > 0

    def test_missing_or_zero_price_is_never_divided_by(self):
        # A ticker with no price entry and one with an explicit zero price must both
        # resolve to zero shares, never a ZeroDivisionError or a garbage quantity.
        basket_df, _ = compute_zerodha_order_basket(
            target_weights={'NOPRICE.NS': 0.5, 'ZEROPRICE.NS': 0.5},
            latest_prices={'ZEROPRICE.NS': 0.0},
            total_capital_inr=100000.0,
        )
        assert (basket_df['Quantity'] == 0).all()
        assert (basket_df['Allocated_INR'] == 0.0).all()

    def test_ticker_suffix_mismatch_falls_back_to_clean_symbol(self):
        # Weights keyed with the .NS suffix, prices keyed without it -- the function must
        # still resolve the price via the clean-symbol fallback rather than treating it as missing.
        basket_df, _ = compute_zerodha_order_basket(
            target_weights={'INFY.NS': 1.0},
            latest_prices={'INFY': 1500.0},
            total_capital_inr=100000.0,
            cash_buffer_pct=0.0,
        )
        row = basket_df.iloc[0]
        assert row['Quantity'] == int(100000.0 // 1500.0)
        assert row['Quantity'] > 0

    def test_pandas_series_input_matches_dict_input(self):
        weights_dict = {'A': 0.5, 'B': 0.5}
        prices_dict = {'A': 200.0, 'B': 400.0}
        basket_from_dict, leftover_dict = compute_zerodha_order_basket(
            weights_dict, prices_dict, 100000.0, cash_buffer_pct=0.01
        )
        basket_from_series, leftover_series = compute_zerodha_order_basket(
            pd.Series(weights_dict), pd.Series(prices_dict), 100000.0, cash_buffer_pct=0.01
        )
        pd.testing.assert_series_equal(
            basket_from_dict.set_index('Ticker')['Quantity'].sort_index(),
            basket_from_series.set_index('Ticker')['Quantity'].sort_index(),
        )
        assert leftover_dict == pytest.approx(leftover_series)

    def test_cash_buffer_percent_is_clamped_to_valid_range(self):
        # A nonsensical >100% buffer must not go negative or explode the math -- it clamps to 1.0
        # (i.e. deploy nothing), leaving effectively all capital as leftover.
        basket_df, leftover = compute_zerodha_order_basket(
            target_weights={'A': 1.0},
            latest_prices={'A': 100.0},
            total_capital_inr=100000.0,
            cash_buffer_pct=5.0,
        )
        assert basket_df.iloc[0]['Quantity'] == 0
        assert leftover == pytest.approx(100000.0)

    def test_zero_or_negative_capital_produces_an_empty_all_zero_basket(self):
        basket_df, leftover = compute_zerodha_order_basket(
            target_weights={'A': 1.0},
            latest_prices={'A': 100.0},
            total_capital_inr=-500.0,
        )
        assert basket_df.iloc[0]['Quantity'] == 0
        assert leftover == pytest.approx(0.0)

    def test_output_columns_are_stable_for_downstream_zerodha_upload(self):
        basket_df, _ = compute_zerodha_order_basket({'A': 1.0}, {'A': 100.0}, 100000.0)
        expected_cols = {
            'Ticker', 'TradingSymbol', 'Target_Weight', 'Price', 'Quantity',
            'Allocated_INR', 'Transaction_Type', 'Order_Type', 'Product', 'Exchange',
        }
        assert expected_cols.issubset(set(basket_df.columns))
        row = basket_df.iloc[0]
        assert row['Transaction_Type'] == 'BUY'
        assert row['Order_Type'] == 'MARKET'
        assert row['Product'] == 'CNC'
        assert row['Exchange'] == 'NSE'
        assert row['TradingSymbol'] == 'A'


class TestScreenerHTMLParsing:
    """
    Verifies the HTML-parsing logic against a synthetic Screener.in-style fixture with known
    correct answers. NOTE: this tests internal parsing correctness, not that the fixture
    actually matches Screener.in's real current markup -- this scraper has never been run
    against a live page (see the module docstring in quant_engine.py). Treat these as
    regression tests for the parser's own logic, not as proof the scraper works in production.
    """

    FIXTURE_HTML = """
    <html><body>
    <ul id="top-ratios">
      <li><span class="name">Market Cap</span><span class="value"><span class="number">50000</span></span></li>
      <li><span class="name">Stock P/E</span><span class="value"><span class="number">22.5</span></span></li>
      <li><span class="name">ROE</span><span class="value"><span class="number">23.8</span></span></li>
    </ul>
    <section id="profit-loss">
    <table class="data-table">
    <thead><tr><th class="text"></th><th>2022</th><th>2023</th><th>2024</th></tr></thead>
    <tbody>
    <tr><td class="text">Sales</td><td>1,000</td><td>1,100</td><td>1,300</td></tr>
    <tr><td class="text">OPM %</td><td>15</td><td>16</td><td>18</td></tr>
    <tr><td class="text">Net Profit</td><td>100</td><td>120</td><td>150</td></tr>
    </tbody>
    </table>
    </section>
    <section id="balance-sheet">
    <table class="data-table">
    <thead><tr><th class="text"></th><th>2022</th><th>2023</th><th>2024</th></tr></thead>
    <tbody>
    <tr><td class="text">Equity Share Capital</td><td>50</td><td>50</td><td>50</td></tr>
    <tr><td class="text">Reserves</td><td>400</td><td>480</td><td>580</td></tr>
    <tr><td class="text">Borrowings</td><td>200</td><td>180</td><td>150</td></tr>
    <tr><td class="text">Total Assets</td><td>800</td><td>850</td><td>900</td></tr>
    </tbody>
    </table>
    </section>
    <section id="cash-flow">
    <table class="data-table">
    <thead><tr><th class="text"></th><th>2022</th><th>2023</th><th>2024</th></tr></thead>
    <tbody>
    <tr><td class="text">Cash from Operating Activity</td><td>90</td><td>130</td><td>160</td></tr>
    </tbody>
    </table>
    </section>
    </body></html>
    """

    def _soup(self):
        from bs4 import BeautifulSoup
        return BeautifulSoup(self.FIXTURE_HTML, 'html.parser')

    def test_clean_screener_number_handles_commas_percent_and_blanks(self):
        assert _clean_screener_number("1,234") == pytest.approx(1234.0)
        assert _clean_screener_number("-56.7") == pytest.approx(-56.7)
        assert _clean_screener_number("12.3%") == pytest.approx(12.3)
        assert _clean_screener_number("-") is None
        assert _clean_screener_number("") is None
        assert _clean_screener_number(None) is None
        assert _clean_screener_number("not a number") is None

    def test_statement_table_parses_rows_oldest_to_newest(self):
        pnl = _parse_screener_statement_table(self._soup(), 'profit-loss')
        assert pnl['Sales'] == [1000.0, 1100.0, 1300.0]
        assert pnl['Net Profit'] == [100.0, 120.0, 150.0]

    def test_statement_table_missing_section_returns_empty(self):
        assert _parse_screener_statement_table(self._soup(), 'does-not-exist') == {}

    def test_top_ratios_parsed_correctly(self):
        ratios = _parse_screener_top_ratios(self._soup())
        assert ratios['Stock P/E'] == pytest.approx(22.5)
        assert ratios['Market Cap'] == pytest.approx(50000.0)

    def test_find_statement_row_is_case_and_alias_insensitive(self):
        pnl = _parse_screener_statement_table(self._soup(), 'profit-loss')
        assert _find_statement_row(pnl, 'sales') == [1000.0, 1100.0, 1300.0]
        assert _find_statement_row(pnl, 'Revenue', 'Sales') == [1000.0, 1100.0, 1300.0]
        assert _find_statement_row(pnl, 'Nonexistent Row') is None

    def test_screener_symbol_strips_exchange_suffix(self):
        assert _screener_symbol('TCS.NS') == 'TCS'
        assert _screener_symbol('RELIANCE.BO') == 'RELIANCE'

    def test_extract_screener_financials_end_to_end_against_fixture(self, monkeypatch):
        import quant_engine as qe
        monkeypatch.setattr(qe, 'fetch_screener_company_html', lambda ticker, session=None: (self.FIXTURE_HTML, None))
        result = extract_screener_financials('TESTCO.NS')
        assert result['success'] is True
        assert result['sales'] == [1000.0, 1100.0, 1300.0]
        assert result['net_profit'] == [100.0, 120.0, 150.0]
        assert result['total_equity'] == [450.0, 530.0, 630.0]  # reserves + share capital
        assert result['top_ratios']['Stock P/E'] == pytest.approx(22.5)

    def test_extract_screener_financials_fetch_failure_is_reported_not_swallowed(self, monkeypatch):
        import quant_engine as qe
        monkeypatch.setattr(qe, 'fetch_screener_company_html', lambda ticker, session=None: (None, 'HTTP 404 from all URLs'))
        result = extract_screener_financials('DOESNOTEXIST.NS')
        assert result['success'] is False
        assert 'error' in result and result['error']

    def test_extract_screener_financials_unrecognized_layout_fails_loudly(self, monkeypatch):
        import quant_engine as qe
        monkeypatch.setattr(qe, 'fetch_screener_company_html', lambda ticker, session=None: ('<html><body>not a real page</body></html>', None))
        result = extract_screener_financials('WEIRDPAGE.NS')
        assert result['success'] is False
        assert 'error' in result and result['error']


class TestDuPontAndPiotroskiFromRealFinancials:
    """
    Hand-computed ground truth for the DuPont/Piotroski math, matching the FIXTURE_HTML company
    above: Sales 1000->1300, Net Profit 100->150, Total Assets 800->900, Total Equity 450->630,
    Borrowings 200->150, CFO 90->160, OPM% 15->18, shares constant at 50.
    """

    FINANCIALS = {
        'sales': [1000.0, 1100.0, 1300.0],
        'net_profit': [100.0, 120.0, 150.0],
        'opm_pct': [15.0, 16.0, 18.0],
        'equity_share_capital': [50.0, 50.0, 50.0],
        'total_equity': [450.0, 530.0, 630.0],
        'borrowings': [200.0, 180.0, 150.0],
        'total_assets': [800.0, 850.0, 900.0],
        'cfo': [90.0, 130.0, 160.0],
        'top_ratios': {'Stock P/E': 22.5},
    }

    def test_dupont_roe_matches_hand_computed_value(self):
        result = compute_dupont_and_piotroski_from_financials(self.FINANCIALS)
        # Net Margin 150/1300, Asset Turnover 1300/900, Financial Leverage 900/630
        expected_roe = (150 / 1300) * (1300 / 900) * (900 / 630) * 100
        assert result['roe'] == pytest.approx(expected_roe, rel=1e-9)

    def test_debt_to_equity_matches_hand_computed_value(self):
        result = compute_dupont_and_piotroski_from_financials(self.FINANCIALS)
        assert result['debt_to_equity'] == pytest.approx(150 / 630, rel=1e-9)

    def test_all_nine_piotroski_criteria_true_except_current_ratio(self):
        result = compute_dupont_and_piotroski_from_financials(self.FINANCIALS)
        breakdown = result['piotroski_breakdown']
        assert breakdown['current_ratio_improving'] is None  # genuinely unavailable, not guessed
        assert all(v is True for k, v in breakdown.items() if k != 'current_ratio_improving')
        assert result['piotroski_score'] == 8
        assert result['piotroski_criteria_available'] == 8

    def test_cfo_less_than_net_income_fails_the_accrual_quality_check(self):
        bad = dict(self.FINANCIALS, cfo=[90.0, 130.0, 100.0])  # CFO 100 < Net Profit 150
        result = compute_dupont_and_piotroski_from_financials(bad)
        assert result['piotroski_breakdown']['cfo_gt_net_income'] is False

    def test_rising_leverage_fails_the_de_reduction_check(self):
        bad = dict(self.FINANCIALS, borrowings=[200.0, 180.0, 400.0])  # D/E rises YoY
        result = compute_dupont_and_piotroski_from_financials(bad)
        assert result['piotroski_breakdown']['de_reduction'] is False

    def test_share_dilution_fails_the_no_dilution_check(self):
        bad = dict(self.FINANCIALS, equity_share_capital=[50.0, 50.0, 60.0])  # >1% dilution
        result = compute_dupont_and_piotroski_from_financials(bad)
        assert result['piotroski_breakdown']['no_dilution'] is False

    def test_single_year_of_data_still_yields_roe_but_nulls_all_yoy_criteria(self):
        one_year = {
            'sales': [1300.0], 'net_profit': [150.0], 'opm_pct': [18.0],
            'equity_share_capital': [50.0], 'total_equity': [630.0], 'borrowings': [150.0],
            'total_assets': [900.0], 'cfo': [160.0], 'top_ratios': {},
        }
        result = compute_dupont_and_piotroski_from_financials(one_year)
        assert result['roe'] is not None  # only needs the latest year
        assert result['piotroski_breakdown']['pos_net_income'] is True  # doesn't need prior year
        for key in ('roa_improving', 'de_reduction', 'no_dilution', 'margin_improving', 'turnover_improving'):
            assert result['piotroski_breakdown'][key] is None

    def test_missing_total_equity_nulls_roe_and_leverage_rather_than_guessing(self):
        no_equity = dict(self.FINANCIALS, total_equity=None)
        result = compute_dupont_and_piotroski_from_financials(no_equity)
        assert result['roe'] is None
        assert result['financial_leverage'] is None
        assert result['debt_to_equity'] is None
        # Net margin and asset turnover don't depend on equity and should still compute
        assert result['net_margin'] is not None
        assert result['asset_turnover'] is not None

    def test_financial_institution_gets_no_piotroski_score_but_keeps_real_roe(self):
        result = compute_dupont_and_piotroski_from_financials(self.FINANCIALS, is_financial_institution=True)
        assert result['piotroski_score'] is None
        assert result['piotroski_breakdown'] == {}
        assert result['piotroski_note'] is not None
        assert result['roe'] is not None
        assert result['debt_to_equity'] is not None


class TestFundamentalsRowBuilderAndSyncOrchestrator:
    """
    Covers the row-assembly and full sync pipeline, including the three states a ticker's
    fundamentals can end up in: successfully audited, exempt (Financial Institution), or
    failed/unsynced -- and that the failed/unsynced states never carry a fabricated number.
    """

    GOOD_FINANCIALS = {
        'success': True, 'error': None,
        'sales': [1000.0, 1300.0], 'net_profit': [100.0, 150.0], 'opm_pct': [15.0, 18.0],
        'equity_share_capital': [50.0, 50.0], 'total_equity': [450.0, 630.0], 'borrowings': [200.0, 150.0],
        'total_assets': [800.0, 900.0], 'cfo': [90.0, 160.0], 'top_ratios': {'Stock P/E': 22.5},
    }

    def test_successful_sync_produces_audited_row_with_real_numbers(self):
        row = _build_fundamentals_row('TESTCO.NS', 'Information Technology', dict(self.GOOD_FINANCIALS))
        assert row['Data Source'] == 'Screener.in (Audited)'
        assert row['piotroski_f_score'] == 8
        assert row['roe_num'] > 0

    def test_financial_institution_row_is_exempt_but_keeps_roe(self):
        row = _build_fundamentals_row('HDFCBANK.NS', 'Financials', dict(self.GOOD_FINANCIALS))
        assert row['Data Source'] == 'Not Applicable (Financial Institution)'
        assert row['piotroski_f_score'] is None
        assert row['roe_num'] > 0

    def test_failed_extraction_never_fabricates_a_number(self):
        row = _build_fundamentals_row('BROKEN.NS', 'Automotive', {'success': False, 'error': 'HTTP 500'})
        assert row['Data Source'] == 'Sync Failed'
        assert row['sync_error'] == 'HTTP 500'
        assert pd.isna(row['roe_num'])
        assert row['piotroski_f_score'] is None

    def test_sync_orchestrator_persists_and_reports_progress(self, tmp_path, monkeypatch):
        import quant_engine as qe
        test_path = str(tmp_path / 'fundamentals.parquet')

        def fake_extract(ticker, session=None):
            if ticker == 'FAILME.NS':
                return {'success': False, 'error': 'HTTP 500'}
            return dict(self.GOOD_FINANCIALS, ticker=ticker)

        monkeypatch.setattr(qe, 'extract_screener_financials', fake_extract)
        monkeypatch.setattr(qe, 'SCREENER_REQUEST_DELAY_SEC', 0.0)

        progress_log = []
        tickers = ['TCS.NS', 'HDFCBANK.NS', 'FAILME.NS']
        sector_map = {'TCS.NS': 'Information Technology', 'HDFCBANK.NS': 'Financials', 'FAILME.NS': 'Automotive'}

        df = sync_screener_fundamentals_for_universe(
            tickers, sector_map=sector_map,
            progress_callback=lambda done, total, t: progress_log.append((done, total, t)),
            fundamentals_path=test_path,
        )

        assert len(df) == 3
        assert df.set_index('Ticker').loc['TCS.NS', 'Data Source'] == 'Screener.in (Audited)'
        assert df.set_index('Ticker').loc['HDFCBANK.NS', 'Data Source'] == 'Not Applicable (Financial Institution)'
        assert df.set_index('Ticker').loc['FAILME.NS', 'Data Source'] == 'Sync Failed'
        assert progress_log == [(1, 3, 'TCS.NS'), (2, 3, 'HDFCBANK.NS'), (3, 3, 'FAILME.NS')]
        assert (tmp_path / 'fundamentals.parquet').exists()

        last_synced = get_fundamentals_last_synced(fundamentals_path=test_path)
        assert last_synced is not None

    def test_fetch_universe_fundamentals_reads_synced_and_flags_unsynced(self, tmp_path, monkeypatch):
        import quant_engine as qe
        test_path = str(tmp_path / 'fundamentals.parquet')
        monkeypatch.setattr(qe, 'extract_screener_financials', lambda t, session=None: dict(self.GOOD_FINANCIALS, ticker=t))
        monkeypatch.setattr(qe, 'SCREENER_REQUEST_DELAY_SEC', 0.0)

        sync_screener_fundamentals_for_universe(['TCS.NS'], fundamentals_path=test_path)
        result = fetch_universe_fundamentals(['TCS.NS', 'NEVERSYNCED.NS'], fundamentals_path=test_path)

        by_ticker = result.set_index('Ticker')
        assert by_ticker.loc['TCS.NS', 'Data Source'] == 'Screener.in (Audited)'
        assert by_ticker.loc['NEVERSYNCED.NS', 'Data Source'] == 'Not Yet Synced'
        assert pd.isna(by_ticker.loc['NEVERSYNCED.NS', 'roe_num'])

    def test_fetch_universe_fundamentals_with_no_store_at_all_blocks_everything(self, tmp_path):
        empty_path = str(tmp_path / 'does_not_exist.parquet')
        result = fetch_universe_fundamentals(['TCS.NS', 'INFY.NS'], fundamentals_path=empty_path)
        assert (result['Data Source'] == 'Not Yet Synced').all()


class TestMultifactorRankingsFundamentalsGating:
    """
    Verifies compute_multifactor_rankings' buy-eligibility gate treats the three real data
    states correctly: a synced ticker with a healthy score is eligible; a Financial Institution
    is exempted from the F-Score gate (not blocked for lacking one); an unsynced ticker is
    always blocked and never given a fabricated 'average' placeholder score.
    """

    GOOD_FINANCIALS = {
        'success': True, 'error': None,
        'sales': [1000.0, 1300.0], 'net_profit': [100.0, 150.0], 'opm_pct': [15.0, 18.0],
        'equity_share_capital': [50.0, 50.0], 'total_equity': [450.0, 630.0], 'borrowings': [200.0, 150.0],
        'total_assets': [800.0, 900.0], 'cfo': [90.0, 160.0], 'top_ratios': {'Stock P/E': 22.5},
    }

    def _scorecard(self):
        tickers = ['SYNCED.NS', 'BANK.NS', 'UNSYNCED.NS']
        rng = np.random.default_rng(11)
        n = 300
        price_hist = pd.DataFrame(
            np.cumprod(1 + rng.normal(0.0006, 0.012, (n, len(tickers))), axis=0) * 100,
            columns=tickers,
        )
        vol_hist = pd.DataFrame(rng.integers(100000, 500000, (n, len(tickers))), columns=tickers)
        adtv_series = pd.Series({t: 2e8 for t in tickers})
        fundamentals_df = pd.DataFrame([
            _build_fundamentals_row('SYNCED.NS', 'Information Technology', dict(self.GOOD_FINANCIALS)),
            _build_fundamentals_row('BANK.NS', 'Financials', dict(self.GOOD_FINANCIALS)),
        ])
        return compute_multifactor_rankings(
            price_hist, tickers, fundamentals_df,
            volume_history_df=vol_hist, adtv_series=adtv_series,
        ).set_index('Ticker')

    def test_synced_ticker_with_healthy_score_is_eligible(self):
        by_ticker = self._scorecard()
        assert by_ticker.loc['SYNCED.NS', 'Selection_Status'] == '🟢 Selected (Top 20)'
        assert by_ticker.loc['SYNCED.NS', 'Data_Available'] == True

    def test_financial_institution_is_exempted_not_blocked(self):
        by_ticker = self._scorecard()
        assert by_ticker.loc['BANK.NS', 'Fundamentals Source'] == 'Not Applicable (Financial Institution)'
        assert by_ticker.loc['BANK.NS', 'F_Score_Safe'] == True
        assert by_ticker.loc['BANK.NS', 'Selection_Status'] not in (
            '⛔ Fundamentals Not Synced (Blocked)', '⛔ F-Score Deteriorating (Blocked)',
        )

    def test_unsynced_ticker_is_blocked_with_no_fabricated_data(self):
        by_ticker = self._scorecard()
        assert by_ticker.loc['UNSYNCED.NS', 'Selection_Status'] == '⛔ Fundamentals Not Synced (Blocked)'
        assert by_ticker.loc['UNSYNCED.NS', 'Data_Available'] == False
        assert pd.isna(by_ticker.loc['UNSYNCED.NS', 'ROE_Clean'])
        assert pd.isna(by_ticker.loc['UNSYNCED.NS', 'DE_Clean'])


class TestSolvePortfolioInMemoryHandlesEmptyCandidatePool:
    """
    Regression tests for a real crash a user hit: with the fundamentals dict eliminated, a
    fresh install (before the first "Sync Audited Filings" run) has zero eligible candidates,
    which used to reach LedoitWolf().fit() on a zero-column DataFrame and raise
    'ValueError: at least one array or dtype is required'. This must degrade to a well-defined
    empty portfolio instead of crashing the app on load.
    """

    def test_empty_dataframe_returns_empty_portfolio_not_a_crash(self):
        result = solve_portfolio_in_memory(pd.DataFrame(), mode='Max-Sharpe (Ledoit-Wolf)')
        assert result['optimal_k'] == 0
        assert result['tickers'] == []
        assert result['clean_tickers'] == []
        assert len(result['w_optimal']) == 0
        assert result['active_cov_matrix'].shape == (0, 0)
        assert len(result['active_mean_vector']) == 0
        assert result['active_returns_df'].empty

    def test_none_returns_df_also_handled(self):
        result = solve_portfolio_in_memory(None, mode='Max-Sharpe (Ledoit-Wolf)')
        assert result['optimal_k'] == 0

    @pytest.mark.parametrize("mode", [
        'Max-Sharpe (Ledoit-Wolf)', 'Minimum Variance Portfolio (MVP)', 'Hierarchical Risk Parity (HRP)',
    ])
    def test_every_optimizer_mode_handles_zero_columns(self, mode):
        zero_col_df = pd.DataFrame(index=pd.date_range('2023-01-01', periods=50))
        result = solve_portfolio_in_memory(zero_col_df, mode=mode)
        assert result['optimal_k'] == 0
        assert len(result['w_optimal']) == 0

    def test_non_empty_input_still_works_after_the_guard(self):
        # The empty-input guard must not accidentally short-circuit real input.
        rng = np.random.default_rng(3)
        returns_df = pd.DataFrame(rng.normal(0.0006, 0.012, (300, 10)), columns=[f'T{i}.NS' for i in range(10)])
        result = solve_portfolio_in_memory(returns_df, mode='Minimum Variance Portfolio (MVP)')
        assert result['optimal_k'] > 0
        assert result['w_optimal'].sum() == pytest.approx(1.0, abs=1e-6)
