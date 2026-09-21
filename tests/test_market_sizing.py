"""Tests for module 3: market sizing (TAM/SAM/SOM reconciliation)."""

from __future__ import annotations

import pytest


def test_topdown_funnel_is_monotonically_decreasing(market):
    td = market["topdown"]
    assert td["som"] <= td["sam"] <= td["tam"]


def test_bottomsup_som_uses_real_smb_unit_economics(dataset, market):
    smb = dataset.customers[dataset.customers["company_size_band"] == "SMB"]
    expected_avg_acv = smb["base_arr"].mean()
    assert market["bottomsup"]["avg_existing_smb_acv"] == pytest.approx(expected_avg_acv, rel=1e-9)


def test_reconciliation_gap_pct_matches_definition(market):
    rec = market["reconciliation"]
    expected_gap = (rec["som_bottomsup"] - rec["som_topdown"]) / rec["som_topdown"] * 100.0
    assert rec["gap_pct"] == pytest.approx(expected_gap, rel=1e-9)


def test_reconciled_som_is_between_the_two_estimates(market):
    rec = market["reconciliation"]
    lo, hi = sorted([rec["som_topdown"], rec["som_bottomsup"]])
    assert lo <= rec["reconciled_som"] <= hi
