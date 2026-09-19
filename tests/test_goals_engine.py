"""
Unit tests for goals_engine.py -- shared inflation/future-value and required-SIP machinery
across Retirement, Child Education, and Home Purchase goals, plus the Emergency Fund's
separate months-of-expenses sizing.
"""
import pytest

from goals_engine import (
    DEFAULT_EMERGENCY_FUND_MONTHS,
    DEFAULT_ON_TRACK_TOLERANCE,
    GOAL_TYPES,
    GOAL_TYPE_CHILD_EDUCATION,
    GOAL_TYPE_EMERGENCY_FUND,
    GOAL_TYPE_HOME_PURCHASE,
    GOAL_TYPE_RETIREMENT,
    compute_emergency_fund_target,
    compute_goal_future_value,
    compute_goal_progress,
    compute_required_monthly_sip,
)


class TestGoalTypes:
    def test_all_four_named_goal_types_present(self):
        assert set(GOAL_TYPES) == {
            GOAL_TYPE_RETIREMENT, GOAL_TYPE_CHILD_EDUCATION, GOAL_TYPE_HOME_PURCHASE, GOAL_TYPE_EMERGENCY_FUND,
        }


class TestGoalFutureValue:
    def test_matches_hand_computed_compound_inflation(self):
        # 10L today, 10 years, 6% inflation -> 10,00,000 * 1.06^10
        fv = compute_goal_future_value(1_000_000.0, 10, 0.06)
        assert fv == pytest.approx(1_000_000.0 * (1.06 ** 10))

    def test_zero_years_returns_present_value_unchanged(self):
        assert compute_goal_future_value(500_000.0, 0, 0.06) == pytest.approx(500_000.0)

    def test_zero_inflation_returns_present_value_unchanged(self):
        assert compute_goal_future_value(500_000.0, 15, 0.0) == pytest.approx(500_000.0)

    def test_negative_target_value_is_floored_at_zero(self):
        assert compute_goal_future_value(-100.0, 10, 0.06) == pytest.approx(0.0)

    def test_negative_years_is_floored_at_zero(self):
        assert compute_goal_future_value(100_000.0, -5, 0.06) == pytest.approx(100_000.0)


class TestRequiredMonthlySIP:
    def test_sip_compounds_to_exactly_the_future_value_needed(self):
        # The core correctness check: accumulating the returned SIP monthly at the same
        # assumed return must reconstruct future_value_needed, not some scaled multiple of it.
        fv_needed = 1_790_847.70
        years = 10
        annual_return = 0.12
        sip = compute_required_monthly_sip(fv_needed, years, annual_return)

        i = annual_return / 12.0
        n = years * 12
        reconstructed_fv = sip * (((1.0 + i) ** n - 1.0) / i)
        assert reconstructed_fv == pytest.approx(fv_needed, rel=1e-9)

    def test_zero_return_falls_back_to_straight_line_division(self):
        sip = compute_required_monthly_sip(120_000.0, 10, 0.0)
        assert sip == pytest.approx(120_000.0 / 120.0)

    def test_near_zero_return_matches_the_straight_line_limit(self):
        # As i -> 0, the annuity factor's limit is exactly n -- verify the two branches agree
        # at the boundary rather than diverging.
        sip_zero = compute_required_monthly_sip(1_200_000.0, 20, 0.0)
        sip_tiny = compute_required_monthly_sip(1_200_000.0, 20, 1e-7)
        assert sip_tiny == pytest.approx(sip_zero, rel=1e-4)

    def test_higher_expected_return_lowers_required_sip(self):
        sip_low_return = compute_required_monthly_sip(2_000_000.0, 15, 0.06)
        sip_high_return = compute_required_monthly_sip(2_000_000.0, 15, 0.14)
        assert sip_high_return < sip_low_return

    def test_zero_future_value_needed_requires_zero_sip(self):
        assert compute_required_monthly_sip(0.0, 10, 0.10) == pytest.approx(0.0)

    def test_zero_years_requires_zero_sip_rather_than_dividing_by_zero(self):
        assert compute_required_monthly_sip(500_000.0, 0, 0.10) == pytest.approx(0.0)

    def test_negative_future_value_is_floored_to_zero_sip(self):
        assert compute_required_monthly_sip(-500_000.0, 10, 0.10) == pytest.approx(0.0)


class TestGoalProgress:
    def test_partially_funded_goal_reports_correct_shortfall(self):
        result = compute_goal_progress(current_corpus=600_000.0, required_corpus=1_000_000.0)
        assert result['pct_funded'] == pytest.approx(0.6)
        assert result['shortfall'] == pytest.approx(400_000.0)
        assert result['surplus'] == pytest.approx(0.0)

    def test_overfunded_goal_reports_surplus_and_uncapped_pct(self):
        result = compute_goal_progress(current_corpus=1_200_000.0, required_corpus=1_000_000.0)
        assert result['pct_funded'] == pytest.approx(1.2)
        assert result['surplus'] == pytest.approx(200_000.0)
        assert result['shortfall'] == pytest.approx(0.0)
        assert result['on_track'] is True

    def test_within_tolerance_shortfall_still_counts_as_on_track(self):
        # 97% funded, default 5% tolerance -> on_track should be True.
        result = compute_goal_progress(current_corpus=970_000.0, required_corpus=1_000_000.0)
        assert result['on_track'] is True
        assert result['pct_funded'] == pytest.approx(0.97)

    def test_beyond_tolerance_shortfall_is_off_track(self):
        # 80% funded, well beyond the default 5% tolerance -> on_track should be False.
        result = compute_goal_progress(current_corpus=800_000.0, required_corpus=1_000_000.0)
        assert result['on_track'] is False

    def test_custom_tolerance_is_respected(self):
        result = compute_goal_progress(current_corpus=800_000.0, required_corpus=1_000_000.0, tolerance=0.25)
        assert result['on_track'] is True

    def test_zero_or_negative_required_corpus_is_treated_as_fully_funded(self):
        result = compute_goal_progress(current_corpus=50_000.0, required_corpus=0.0)
        assert result['pct_funded'] == pytest.approx(1.0)
        assert result['on_track'] is True
        assert result['surplus'] == pytest.approx(50_000.0)

    def test_negative_current_corpus_is_floored_at_zero(self):
        result = compute_goal_progress(current_corpus=-100.0, required_corpus=1_000_000.0)
        assert result['current_corpus'] == pytest.approx(0.0)
        assert result['pct_funded'] == pytest.approx(0.0)


class TestEmergencyFundTarget:
    def test_matches_expenses_times_months_coverage(self):
        target = compute_emergency_fund_target(monthly_expenses=80_000.0, months_coverage=6.0)
        assert target == pytest.approx(480_000.0)

    def test_default_months_coverage_is_six(self):
        assert compute_emergency_fund_target(monthly_expenses=50_000.0) == pytest.approx(
            50_000.0 * DEFAULT_EMERGENCY_FUND_MONTHS
        )

    def test_custom_months_coverage_for_variable_income(self):
        target = compute_emergency_fund_target(monthly_expenses=60_000.0, months_coverage=12.0)
        assert target == pytest.approx(720_000.0)

    def test_negative_expenses_is_floored_at_zero(self):
        assert compute_emergency_fund_target(monthly_expenses=-1000.0) == pytest.approx(0.0)

    def test_does_not_use_any_return_or_inflation_assumption(self):
        # Regression guard: emergency fund sizing must stay independent of the SIP/FV machinery
        # -- doubling months_coverage must exactly double the target with no compounding curve.
        base = compute_emergency_fund_target(monthly_expenses=40_000.0, months_coverage=6.0)
        doubled = compute_emergency_fund_target(monthly_expenses=40_000.0, months_coverage=12.0)
        assert doubled == pytest.approx(base * 2.0)
