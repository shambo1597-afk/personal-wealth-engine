"""
Unit tests for quant_engine.py — HRP tree clustering, the sector/asset-capped simplex
projection, and Newton-Raphson XIRR convergence.
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
