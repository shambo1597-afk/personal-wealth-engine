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
    _fetch_single_fundamental,
    CALIBRATED_CONSTITUENT_FUNDAMENTALS,
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


class TestFundamentalsDataProvenance:
    """
    The Piotroski/DuPont quality gate runs on a static, undated snapshot -- never a live
    filing. These tests lock in the 'Data Source' / 'Fundamentals Source' labelling so a
    future change can't silently make a hardcoded number look indistinguishable from a
    live one in the UI.
    """

    def test_known_constituent_is_labelled_calibrated_snapshot(self):
        known_ticker = next(iter(CALIBRATED_CONSTITUENT_FUNDAMENTALS))
        result = _fetch_single_fundamental(known_ticker)
        assert result['Data Source'] == 'Calibrated Snapshot'
        # and its numbers must match the hardcoded profile exactly, not a derived estimate
        prof = CALIBRATED_CONSTITUENT_FUNDAMENTALS[known_ticker]
        assert result['roe_num'] == pytest.approx(float(prof['roe']))

    def test_unknown_ticker_is_labelled_sector_average_estimate(self):
        result = _fetch_single_fundamental('NOT_A_REAL_TICKER.NS', sector='Information Technology')
        assert result['Data Source'] == 'Sector-Average Estimate'
        assert 'NOT_A_REAL_TICKER.NS' not in CALIBRATED_CONSTITUENT_FUNDAMENTALS

    def test_fetch_universe_fundamentals_preserves_data_source_per_ticker(self):
        known_ticker = next(iter(CALIBRATED_CONSTITUENT_FUNDAMENTALS))
        tickers = [known_ticker, 'NOT_A_REAL_TICKER.NS']
        df = fetch_universe_fundamentals(tickers, sector_map={known_ticker: 'Financials'})
        by_ticker = df.set_index('Ticker')['Data Source']
        assert by_ticker[known_ticker] == 'Calibrated Snapshot'
        assert by_ticker['NOT_A_REAL_TICKER.NS'] == 'Sector-Average Estimate'

    def test_multifactor_rankings_scorecard_carries_fundamentals_source_column(self):
        known_ticker = next(iter(CALIBRATED_CONSTITUENT_FUNDAMENTALS))
        tickers = [known_ticker, 'NOT_A_REAL_TICKER.NS']
        n = 300
        rng = np.random.default_rng(9)
        price_hist = pd.DataFrame(
            np.cumprod(1 + rng.normal(0.0005, 0.01, (n, len(tickers))), axis=0) * 100,
            columns=tickers,
        )
        vol_hist = pd.DataFrame(rng.integers(100000, 500000, (n, len(tickers))), columns=tickers)
        adtv_series = pd.Series({t: 2e8 for t in tickers})
        fundamentals_df = fetch_universe_fundamentals(tickers)

        scorecard = compute_multifactor_rankings(
            price_hist, tickers, fundamentals_df,
            volume_history_df=vol_hist, adtv_series=adtv_series,
        )
        assert 'Fundamentals Source' in scorecard.columns
        by_ticker = scorecard.set_index('Ticker')['Fundamentals Source']
        assert by_ticker[known_ticker] == 'Calibrated Snapshot'
        assert by_ticker['NOT_A_REAL_TICKER.NS'] == 'Sector-Average Estimate'
