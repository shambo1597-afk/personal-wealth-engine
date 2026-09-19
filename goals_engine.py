# ====================================================================================================
# MASTER QUANTITATIVE WEALTH OPERATING SYSTEM - GOALS ENGINE MODULE
# Named Goal-Based Financial Planning: Retirement, Child Education, Home Purchase, Emergency Fund
# ====================================================================================================

from typing import Any, Dict

# ----------------------------------------------------------------------------------------------------
# 1. NAMED GOAL TYPES
#
# All goal types except Emergency Fund share the same three-step planning pipeline:
#   1. Inflate today's target cost to its future nominal value (compute_goal_future_value)
#   2. Solve for the monthly SIP required to reach that future value (compute_required_monthly_sip)
#   3. Compare an already-accumulated corpus against the requirement (compute_goal_progress)
# Emergency Fund skips all three -- see section 4.
# ----------------------------------------------------------------------------------------------------
GOAL_TYPE_RETIREMENT       = 'Retirement'
GOAL_TYPE_CHILD_EDUCATION  = 'Child Education'
GOAL_TYPE_EMERGENCY_FUND   = 'Emergency Fund'
GOAL_TYPE_HOME_PURCHASE    = 'Home Purchase'

GOAL_TYPES = (GOAL_TYPE_RETIREMENT, GOAL_TYPE_CHILD_EDUCATION, GOAL_TYPE_HOME_PURCHASE, GOAL_TYPE_EMERGENCY_FUND)

DEFAULT_ON_TRACK_TOLERANCE = 0.05      # 5% shortfall still counts as "on track" (rounding/timing slack)
DEFAULT_EMERGENCY_FUND_MONTHS = 6.0    # Standard 6-month-expenses emergency corpus baseline

# ----------------------------------------------------------------------------------------------------
# 2. INFLATION-ADJUSTED FUTURE VALUE (SHARED ACROSS RETIREMENT / EDUCATION / HOME PURCHASE)
# ----------------------------------------------------------------------------------------------------
def compute_goal_future_value(target_today_value: float, years_to_goal: float, inflation_rate: float) -> float:
    """
    Inflates today's target cost to its future nominal value at the goal date via standard
    compound inflation:
        FV = PV * (1 + inflation_rate) ** years_to_goal
    Shared by every non-Emergency-Fund goal type -- none should duplicate this compounding logic
    independently. `target_today_value` and `years_to_goal` are floored at 0.0 (a goal can't
    have a negative cost or be in the past for planning purposes).
    """
    pv = max(0.0, float(target_today_value))
    years = max(0.0, float(years_to_goal))
    rate = float(inflation_rate)
    return pv * ((1.0 + rate) ** years)

# ----------------------------------------------------------------------------------------------------
# 3. REQUIRED MONTHLY SIP (FUTURE VALUE OF AN ORDINARY ANNUITY, SOLVED FOR THE PAYMENT)
# ----------------------------------------------------------------------------------------------------
def compute_required_monthly_sip(future_value_needed: float, years_to_goal: float, expected_annual_return: float) -> float:
    """
    Solves the standard future-value-of-an-ordinary-annuity formula for the monthly payment
    (month-end contributions, the conventional SIP timing assumption):
        FV = SIP * [ ((1 + i)^n - 1) / i ]
    where i = expected_annual_return / 12 (monthly compounding rate) and
    n = years_to_goal * 12 (number of monthly contributions). Solved for SIP:
        SIP = FV * i / ((1 + i)^n - 1)
    Falls back to a straight-line FV / n split when expected_annual_return is ~0 -- as i -> 0 the
    annuity factor's limit is exactly n (L'Hopital / first-order Taylor expansion of (1+i)^n),
    so SIP = FV / n is the correct zero-return limit, not an approximation.
    Returns 0.0 if future_value_needed <= 0 or years_to_goal <= 0 (nothing to save for, or no
    time left to save in).
    """
    fv = max(0.0, float(future_value_needed))
    years = max(0.0, float(years_to_goal))
    if fv <= 0.0 or years <= 0.0:
        return 0.0

    n = years * 12.0
    i = float(expected_annual_return) / 12.0
    if abs(i) < 1e-9:
        return fv / n

    growth_term = (1.0 + i) ** n - 1.0
    if growth_term <= 0.0:
        return fv / n
    return fv * i / growth_term

# ----------------------------------------------------------------------------------------------------
# 4. GOAL PROGRESS TRACKING (CURRENT CORPUS VS. REQUIRED CORPUS)
# ----------------------------------------------------------------------------------------------------
def compute_goal_progress(
    current_corpus: float,
    required_corpus: float,
    tolerance: float = DEFAULT_ON_TRACK_TOLERANCE,
) -> Dict[str, Any]:
    """
    Compares an already-accumulated corpus against the required corpus for a goal and reports
    funding status. `on_track` is True whenever the corpus is within `tolerance` of the
    requirement or exceeds it, i.e. pct_funded >= (1 - tolerance) -- a small, expected shortfall
    close to the goal date isn't flagged as "off track" the same way a large one is.
    Returns:
        {
            'current_corpus': float,
            'required_corpus': float,
            'pct_funded': float,   # uncapped, so overfunding shows e.g. 1.2 (120% funded)
            'shortfall': float,    # >= 0; 0.0 if fully funded or in surplus
            'surplus': float,      # >= 0; 0.0 if underfunded
            'on_track': bool,
        }
    If required_corpus <= 0, the goal is treated as already fully funded (there is nothing left
    to save for) rather than dividing by zero.
    """
    current = max(0.0, float(current_corpus))
    required = float(required_corpus)

    if required <= 0.0:
        return {
            'current_corpus': current,
            'required_corpus': 0.0,
            'pct_funded': 1.0,
            'shortfall': 0.0,
            'surplus': current,
            'on_track': True,
        }

    pct_funded = current / required
    shortfall = max(0.0, required - current)
    surplus = max(0.0, current - required)
    on_track = pct_funded >= (1.0 - tolerance)

    return {
        'current_corpus': current,
        'required_corpus': required,
        'pct_funded': pct_funded,
        'shortfall': shortfall,
        'surplus': surplus,
        'on_track': on_track,
    }

# ----------------------------------------------------------------------------------------------------
# 5. EMERGENCY FUND (MONTHS-OF-EXPENSES LIQUIDITY BUFFER -- NO RETURN ASSUMPTION)
# ----------------------------------------------------------------------------------------------------
def compute_emergency_fund_target(monthly_expenses: float, months_coverage: float = DEFAULT_EMERGENCY_FUND_MONTHS) -> float:
    """
    Emergency Fund target corpus = monthly_expenses * months_coverage.
    Deliberately bypasses the future-value / SIP machinery used by the other three goal types:
    an emergency fund is a liquidity buffer meant to sit in cash / liquid instruments for
    immediate access, not a growth target with an inflation or return assumption attached to it.
    `months_coverage` defaults to the standard 6-month baseline but is a parameter so a user with
    less stable income (e.g. freelance/business) can plan for a longer runway.
    """
    expenses = max(0.0, float(monthly_expenses))
    months = max(0.0, float(months_coverage))
    return expenses * months
