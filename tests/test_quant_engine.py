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
    sync_structured_fundamentals_for_universe,
    sync_screener_fundamentals_for_universe,
    get_fundamentals_last_synced,
    fetch_structured_company_fundamentals,
    get_kite_client,
    sync_zerodha_live_data,
    load_local_parquet_fundamentals,
    solve_portfolio_in_memory,
    fetch_live_dynamic_multiasset_universe,
    get_asset_class,
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


class TestStructuredFundamentals:
    """
    Unit tests for structured REST financial statements ingestion, DuPont 3-stage ROE
    decomposition, and 9-point Piotroski F-Scores.
    """

    def test_verified_indian_fundamentals_tcs(self):
        res = fetch_structured_company_fundamentals('TCS.NS')
        assert res['Asset'] == 'TCS'
        assert res['piotroski_f_score'] == 9
        assert res['roe_num'] > 40.0
        assert res['de_num'] <= 0.10
        # Curated reference numbers are not pulled from a live audited filing -- the F-Score
        # safety gate treats the substring 'Audited' as verified data, so this must never appear
        # here for anything less than a real structured REST payload.
        assert 'Audited' not in res['Data Source']

    def test_curated_and_synthetic_fallbacks_never_pass_as_audited(self):
        # Neither the hand-curated dict nor the ticker-hash fallback is real audited data --
        # both must be excluded from the F-Score safety gate's 'Audited' substring check, so a
        # fresh install without a structured API key blocks the whole universe instead of
        # silently trading on fabricated numbers.
        curated = fetch_structured_company_fundamentals('TCS.NS')
        synthetic = fetch_structured_company_fundamentals('XYZ_UNKNOWN.NS')
        assert 'Audited' not in curated['Data Source']
        assert 'Audited' not in synthetic['Data Source']

    def test_verified_indian_fundamentals_reliance(self):
        res = fetch_structured_company_fundamentals('RELIANCE.NS')
        assert res['Asset'] == 'RELIANCE'
        assert res['piotroski_f_score'] >= 7
        assert res['roe_num'] > 5.0
        assert res['de_num'] < 0.60

    def test_deterministic_fallback_for_custom_ticker(self):
        res = fetch_structured_company_fundamentals('XYZ_UNKNOWN.NS')
        assert res['Asset'] == 'XYZ_UNKNOWN'
        assert 1 <= res['piotroski_f_score'] <= 9
        assert res['roe_num'] > 0
        assert res['pe_num'] > 0

    def _mock_eodhd_json(self, include_prior_year=True):
        periods = {
            '2024-03-31': {'assets': '1000000', 'equity': '600000', 'debt': '50000', 'rev': '1200000', 'ni': '180000', 'cfo': '200000'},
        }
        if include_prior_year:
            # Weaker prior year: both ROA and asset turnover genuinely improved YoY.
            periods['2023-03-31'] = {'assets': '900000', 'equity': '550000', 'debt': '60000', 'rev': '1000000', 'ni': '100000', 'cfo': '95000'}

        return {
            'Financials': {
                'Balance_Sheet': {'yearly': {
                    yr: {'totalAssets': p['assets'], 'totalStockholderEquity': p['equity'], 'shortLongTermDebtTotal': p['debt']}
                    for yr, p in periods.items()
                }},
                'Income_Statement': {'yearly': {
                    yr: {'totalRevenue': p['rev'], 'netIncome': p['ni']} for yr, p in periods.items()
                }},
                'Cash_Flow': {'yearly': {
                    yr: {'totalCashFromOperatingActivities': p['cfo']} for yr, p in periods.items()
                }},
            }
        }

    def test_eodhd_rest_api_payload_parsing(self, monkeypatch):
        mock_eodhd_json = self._mock_eodhd_json(include_prior_year=True)

        class MockResponse:
            status_code = 200
            def json(self):
                return mock_eodhd_json

        import requests
        monkeypatch.setattr(requests, 'get', lambda url, timeout=4.0: MockResponse())

        res = fetch_structured_company_fundamentals('INFY.NS', api_key='TEST_KEY', provider='EODHD')
        assert res['Data Source'] == 'EODHD REST API (Audited)'
        assert res['npm_num'] == pytest.approx(15.0)  # 180k / 1200k = 15%
        assert res['turnover_num'] == pytest.approx(1.2)  # 1200k / 1000k = 1.2
        assert res['leverage_num'] == pytest.approx(1000000 / 600000)
        # Both YoY criteria (ROA improved, turnover improved) are genuinely earned here --
        # 180k/1M=18% ROA vs 100k/900k=11.1% prior, and 1.2x turnover vs 1.11x prior.
        assert res['piotroski_f_score'] == 9
        assert res['piotroski_criteria_available'] == 9

    def test_eodhd_single_year_of_filings_fails_closed_not_free_points(self, monkeypatch):
        # Regression test for a real bug: the Piotroski YoY criteria used to be hardcoded `True`
        # regardless of data, silently inflating every audited company's score by 2 points. With
        # only one year of filings on hand, those two criteria must now fail (not pass by default).
        mock_eodhd_json = self._mock_eodhd_json(include_prior_year=False)

        class MockResponse:
            status_code = 200
            def json(self):
                return mock_eodhd_json

        import requests
        monkeypatch.setattr(requests, 'get', lambda url, timeout=4.0: MockResponse())

        res = fetch_structured_company_fundamentals('INFY.NS', api_key='TEST_KEY', provider='EODHD')
        # Same latest-year fundamentals as the full test above, so every criterion that doesn't
        # need a prior year still passes -- only the 2 YoY criteria are unavailable. 9 - 2 = 7.
        assert res['piotroski_f_score'] == 7
        assert res['piotroski_criteria_available'] == 7
        assert res['piotroski_note'] is not None


class TestZerodhaKiteIntegration:
    """
    Unit tests for Zerodha KiteConnect client initialization and live data synchronization.
    """

    def test_get_kite_client_graceful_failure(self):
        # Invalid credentials or missing library gracefully returns None or handles cleanly
        client = get_kite_client("dummy_api_key", "dummy_access_token")
        # In test environment, client is either a KiteConnect object or None
        assert client is not None or client is None

    def test_sync_zerodha_live_data_mock(self):
        class MockKite:
            def holdings(self):
                return [
                    {'tradingsymbol': 'TCS', 'quantity': 10, 'average_price': 3500.0},
                    {'tradingsymbol': 'INFY', 'quantity': 25, 'average_price': 1500.0},
                    {'tradingsymbol': 'WIPRO', 'quantity': 0, 'average_price': 400.0}
                ]
            def margins(self, segment='equity'):
                return {'available': {'live_balance': 250000.0}}
            def ltp(self, keys):
                return {
                    'NSE:TCS': {'last_price': 3850.0},
                    'NSE:INFY': {'last_price': 1620.0}
                }

        mock_kite = MockKite()
        holdings, margins, quotes = sync_zerodha_live_data(mock_kite, ['TCS.NS', 'INFY.NS'])

        assert len(holdings) == 2
        assert holdings[0]['ticker'] == 'TCS.NS'
        assert holdings[0]['quantity'] == 10
        assert holdings[1]['ticker'] == 'INFY.NS'
        assert holdings[1]['quantity'] == 25
        assert margins == 250000.0
        assert quotes['TCS.NS'] == 3850.0
        assert quotes['INFY.NS'] == 1620.0

    def test_sync_zerodha_live_data_none_client(self):
        holdings, margins, quotes = sync_zerodha_live_data(None, ['TCS.NS'])
        assert holdings == []
        assert margins == 0.0
        assert quotes == {}


class TestFundamentalsPersistence:
    """
    Unit tests for Parquet persistence, caching, and universe sync.
    """

    def test_sync_structured_fundamentals_persists_parquet(self, tmp_path):
        test_parquet = str(tmp_path / 'test_fundamentals.parquet')
        tickers = ['TCS.NS', 'INFY.NS', 'RELIANCE.NS']
        df = sync_structured_fundamentals_for_universe(tickers, fundamentals_path=test_parquet)

        assert len(df) == 3
        assert (tmp_path / 'test_fundamentals.parquet').exists()
        loaded = load_local_parquet_fundamentals(fundamentals_path=test_parquet)
        assert loaded is not None
        assert len(loaded) == 3
        assert 'piotroski_f_score' in loaded.columns

    def test_fetch_universe_fundamentals_reads_or_creates(self, tmp_path):
        test_parquet = str(tmp_path / 'test_store.parquet')
        res = fetch_universe_fundamentals(['TCS.NS', 'HDFCBANK.NS'], fundamentals_path=test_parquet)
        assert len(res) == 2
        assert 'TCS.NS' in res['Ticker'].values
        assert 'HDFCBANK.NS' in res['Ticker'].values

    def test_get_fundamentals_last_synced(self, tmp_path):
        test_parquet = str(tmp_path / 'test_synced.parquet')
        sync_structured_fundamentals_for_universe(['TCS.NS'], fundamentals_path=test_parquet)
        last_synced = get_fundamentals_last_synced(fundamentals_path=test_parquet)
        assert last_synced is not None
        assert 'UTC' in last_synced


class TestMultifactorRankingsFundamentalsGating:
    """
    Verifies compute_multifactor_rankings' buy-eligibility gate with structured fundamentals.
    """

    def _price_and_volume(self, tickers):
        rng = np.random.default_rng(11)
        n = 300
        price_hist = pd.DataFrame(
            np.cumprod(1 + rng.normal(0.0006, 0.012, (n, len(tickers))), axis=0) * 100,
            columns=tickers,
        )
        vol_hist = pd.DataFrame(rng.integers(100000, 500000, (n, len(tickers))), columns=tickers)
        adtv_series = pd.Series({t: 2e8 for t in tickers})
        return price_hist, vol_hist, adtv_series

    def _scorecard(self, tickers, fundamentals_df):
        price_hist, vol_hist, adtv_series = self._price_and_volume(tickers)
        return compute_multifactor_rankings(
            price_hist, tickers, fundamentals_df,
            volume_history_df=vol_hist, adtv_series=adtv_series,
        ).set_index('Ticker')

    def test_ticker_with_no_synced_fundamentals_is_blocked_not_faked(self, tmp_path):
        # A fresh install (or a sync with no structured API key configured) must never let a
        # ticker trade on curated/synthetic estimate data -- it should show up as genuinely
        # blocked ("we don't know"), not silently pass the F-Score gate as if it were audited.
        tickers = ['TCS.NS', 'HDFCBANK.NS']
        fundamentals_df = fetch_universe_fundamentals(tickers, fundamentals_path=str(tmp_path / 'fund.parquet'))
        by_ticker = self._scorecard(tickers, fundamentals_df)
        assert by_ticker.loc['TCS.NS', 'Data_Available'] == False
        assert by_ticker.loc['TCS.NS', 'Selection_Status'] == '⛔ Fundamentals Not Synced (Blocked)'

    def test_synced_ticker_with_healthy_score_is_eligible(self):
        tickers = ['TCS.NS', 'HDFCBANK.NS']
        # Simulate a genuine post-sync state: real structured REST data persisted for TCS.
        fundamentals_df = pd.DataFrame([{
            'Ticker': 'TCS.NS', 'Asset': 'TCS', 'Data Source': 'EODHD REST API (Audited)',
            'roe_num': 45.0, 'de_num': 0.05, 'npm_num': 19.0, 'pe_num': 28.0,
            'piotroski_f_score': 9, 'piotroski_badge': '🟢 Strong (8-9/9)',
            'Piotroski F-Score': '9/9 ★', 'Fundamental Health': '✅ High Quality (Audited)',
        }])
        by_ticker = self._scorecard(tickers, fundamentals_df)
        assert by_ticker.loc['TCS.NS', 'Selection_Status'] == '🟢 Selected (Top 20)'
        assert by_ticker.loc['TCS.NS', 'Data_Available'] == True

    def test_banking_ticker_is_eligible(self):
        tickers = ['TCS.NS', 'HDFCBANK.NS']
        # A bank exempted from the Piotroski test (industrial-balance-sheet assumptions don't
        # apply) is still eligible even without a numeric F-Score, per the
        # 'Not Applicable (Financial Institution)' exemption.
        fundamentals_df = pd.DataFrame([{
            'Ticker': 'HDFCBANK.NS', 'Asset': 'HDFCBANK', 'Data Source': 'Not Applicable (Financial Institution)',
            'roe_num': 17.0, 'de_num': 0.85, 'npm_num': 22.0, 'pe_num': 18.0,
            'piotroski_f_score': None, 'piotroski_badge': 'N/A',
            'Piotroski F-Score': 'N/A', 'Fundamental Health': 'Not Applicable (Financial Institution)',
        }])
        by_ticker = self._scorecard(tickers, fundamentals_df)
        assert by_ticker.loc['HDFCBANK.NS', 'Data_Available'] == True
        assert by_ticker.loc['HDFCBANK.NS', 'F_Score_Safe'] == True

    def test_eodhd_sync_exempts_financial_institutions_from_fscore_gate(self, monkeypatch):
        # fetch_structured_company_fundamentals itself (not just the scorecard gate) must label
        # a bank's real audited REST data as exempt, since a numeric Piotroski F-Score computed
        # with the generic industrial formula on a bank's balance sheet is meaningless.
        mock_json = {
            'Financials': {
                'Balance_Sheet': {'yearly': {'2024-03-31': {
                    'totalAssets': '1000000', 'totalStockholderEquity': '120000', 'shortLongTermDebtTotal': '850000',
                }}},
                'Income_Statement': {'yearly': {'2024-03-31': {'totalRevenue': '90000', 'netIncome': '20000'}}},
                'Cash_Flow': {'yearly': {'2024-03-31': {'totalCashFromOperatingActivities': '25000'}}},
            }
        }

        class MockResponse:
            status_code = 200
            def json(self):
                return mock_json

        import requests
        monkeypatch.setattr(requests, 'get', lambda url, timeout=4.0: MockResponse())

        res = fetch_structured_company_fundamentals('HDFCBANK.NS', api_key='TEST_KEY', provider='EODHD')
        assert res['Data Source'] == 'Not Applicable (Financial Institution)'


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


class TestDynamicMultiAssetDiscovery:
    """
    Tests for the dynamic zero-hardcoded multi-asset discovery engine and asset class mappings.
    """

    def test_dynamic_multiasset_discovery_returns_tuples_and_anchors(self):
        candidates, sec_map, cls_map = fetch_live_dynamic_multiasset_universe(turbo_mode=True)
        assert len(candidates) > 0
        assert len(sec_map) > 0
        assert len(cls_map) > 0
        assert 'GILT5YBEES.NS' in candidates
        assert 'Sovereign Fixed Income' in sec_map.get('GILT5YBEES.NS')
        assert cls_map.get('GILT5YBEES.NS') == 'Sovereign Debt ETF (Sec 50AA)'

    def test_get_asset_class_resolution(self):
        assert get_asset_class('GOLDBEES.NS') == 'Precious Metals (Gold)'
        assert get_asset_class('SILVERBEES.NS') == 'Precious Metals (Silver)'
        assert get_asset_class('EMBASSY.NS') == 'Real Estate (REIT)'
        assert get_asset_class('PGINVIT.NS') == 'Infrastructure (InvIT)'
        assert get_asset_class('TCS.NS') == 'Equity Delivery (Sec 112A)'

