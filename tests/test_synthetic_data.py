"""Sanity checks on the shared synthetic data generator."""

from __future__ import annotations

import pandas as pd

from gsdt.synthetic_data import generate_customers, generate_revenue_panel, generate_rfm_snapshot, generate_full_dataset


def test_customer_count_in_requested_range():
    customers = generate_customers(n=1200, seed=1)
    assert len(customers) == 1200
    assert 800 <= len(customers) <= 1500


def test_generation_is_deterministic_given_a_seed():
    a = generate_full_dataset(n=300, seed=7)
    b = generate_full_dataset(n=300, seed=7)
    pd.testing.assert_frame_equal(a.customers, b.customers)
    pd.testing.assert_frame_equal(a.panel, b.panel)


def test_revenue_panel_has_no_negative_revenue():
    customers = generate_customers(n=300, seed=2)
    panel = generate_revenue_panel(customers, seed=3)
    assert (panel["revenue"] >= 0).all()


def test_rfm_snapshot_monetary_matches_final_year_panel_revenue():
    customers = generate_customers(n=300, seed=8)
    panel = generate_revenue_panel(customers, seed=9)
    rfm = generate_rfm_snapshot(customers, panel, seed=10)
    final_year = max(panel["year"])
    final_year_revenue = panel[panel["year"] == final_year].set_index("customer_id")["revenue"]
    merged = rfm.set_index("customer_id")["monetary_12m"]
    pd.testing.assert_series_equal(
        merged.sort_index(), final_year_revenue.reindex(merged.index).sort_index(), check_names=False
    )
