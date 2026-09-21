"""Tests for module 2: growth decomposition (growth bridge arithmetic)."""

from __future__ import annotations

import pytest

from gsdt.growth_decomposition import validate_bridge, RECONCILIATION_TOLERANCE


def test_total_bridge_reconciles_for_every_transition(bridges):
    for bridge in bridges.values():
        validate_bridge(bridge)
        assert (bridge["reconciliation_diff"].abs() < RECONCILIATION_TOLERANCE).all()


def test_segment_level_bridge_reconciles(latest_bridge):
    assert "TOTAL" in latest_bridge.index
    for seg_id in latest_bridge.index:
        row = latest_bridge.loc[seg_id]
        reconciled = row["start_revenue"] + row["new"] + row["expansion"] + row["contraction"] + row["churn"] + row["price"]
        assert reconciled == pytest.approx(row["end_revenue"], abs=1e-6)


def test_total_row_equals_sum_of_segment_rows(latest_bridge):
    non_total = latest_bridge.drop(index="TOTAL")
    for col in ["start_revenue", "new", "expansion", "contraction", "churn", "price", "end_revenue"]:
        assert non_total[col].sum() == pytest.approx(latest_bridge.loc["TOTAL", col], abs=1e-6)


def test_new_driver_is_never_negative(latest_bridge):
    assert (latest_bridge["new"] >= -1e-9).all()


def test_churn_driver_is_never_positive(latest_bridge):
    assert (latest_bridge["churn"] <= 1e-9).all()


def test_cumulative_bridge_telescopes_to_first_start_and_last_end(bridges, cum_bridge):
    pairs = sorted(bridges.keys())
    first_start = bridges[pairs[0]].loc["TOTAL", "start_revenue"]
    last_end = bridges[pairs[-1]].loc["TOTAL", "end_revenue"]
    assert cum_bridge.loc["TOTAL", "start_revenue"] == pytest.approx(first_start, abs=1e-6)
    assert cum_bridge.loc["TOTAL", "end_revenue"] == pytest.approx(last_end, abs=1e-6)
    validate_bridge(cum_bridge)


def test_validate_bridge_raises_on_tampered_data(latest_bridge):
    tampered = latest_bridge.copy()
    tampered.loc["TOTAL", "end_revenue"] += 100.0
    tampered["reconciliation_diff"] = (
        tampered["start_revenue"] + tampered["new"] + tampered["expansion"] + tampered["contraction"]
        + tampered["churn"] + tampered["price"] - tampered["end_revenue"]
    )
    with pytest.raises(AssertionError):
        validate_bridge(tampered)
