"""Tests for module 4: pricing elasticity / willingness-to-pay scenarios."""

from __future__ import annotations

import pytest

from gsdt.pricing_elasticity import simulate_pricing_scenario, ELASTICITY_BY_BAND


def test_baseline_scenario_price_change_is_zero(pricing_results):
    baseline = pricing_results[pricing_results["scenario"] == "Baseline (no change)"].iloc[0]
    assert baseline["price_change_pct"] == 0.0
    assert baseline["revenue_change_vs_baseline_pct"] == pytest.approx(0.0, abs=1e-9)


def test_revenue_equals_price_times_retained_quantity(pricing_base):
    detail = simulate_pricing_scenario(pricing_base, price_change_pct=0.08)
    expected_revenue = (detail["retained_fraction"] * detail["new_price"]).sum()
    assert detail["expected_revenue"].sum() == pytest.approx(expected_revenue, rel=1e-9)


def test_higher_price_change_reduces_retention_rate(pricing_results):
    ordered = pricing_results.sort_values("price_change_pct")
    retention = ordered["retention_rate_pct"].values
    assert all(retention[i] >= retention[i + 1] for i in range(len(retention) - 1))


def test_gross_margin_pct_matches_profit_over_revenue(pricing_results):
    for _, row in pricing_results.iterrows():
        expected_margin_pct = row["projected_gross_profit"] / row["projected_revenue"] * 100.0
        assert row["gross_margin_pct"] == pytest.approx(expected_margin_pct, rel=1e-9)


def test_more_price_sensitive_band_retains_less_at_same_price_increase(pricing_base):
    smb_detail = simulate_pricing_scenario(
        pricing_base[pricing_base["company_size_band"] == "SMB"], price_change_pct=0.10
    )
    enterprise_detail = simulate_pricing_scenario(
        pricing_base[pricing_base["company_size_band"] == "Enterprise"], price_change_pct=0.10
    )
    assert ELASTICITY_BY_BAND["SMB"] > ELASTICITY_BY_BAND["Enterprise"]
    if len(smb_detail) and len(enterprise_detail):
        assert smb_detail["retained_fraction"].mean() <= enterprise_detail["retained_fraction"].mean()
