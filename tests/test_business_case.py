"""Tests for module 6: business case (prioritized initiatives)."""

from __future__ import annotations

import math

import pytest


def test_build_business_case_returns_exactly_three_top_initiatives(business_case_result):
    top3, _all = business_case_result
    assert len(top3) == 3


def test_each_initiative_has_positive_revenue_impact_and_cost(business_case_result):
    _top3, all_ranked = business_case_result
    for init in all_ranked:
        assert init.revenue_impact_annual >= 0
        assert init.implementation_cost > 0


def test_payback_months_matches_cost_over_monthly_gross_profit(business_case_result):
    _top3, all_ranked = business_case_result
    for init in all_ranked:
        if init.gross_profit_impact_annual > 0:
            expected = init.implementation_cost / (init.gross_profit_impact_annual / 12.0)
            assert init.payback_months == pytest.approx(expected, rel=1e-9)
        else:
            assert math.isinf(init.payback_months)
