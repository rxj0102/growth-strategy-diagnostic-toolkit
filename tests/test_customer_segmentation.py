"""Tests for module 1: customer segmentation."""

from __future__ import annotations

import pytest

from gsdt.customer_segmentation import MIN_SEGMENTS, MAX_SEGMENTS


def test_chosen_k_within_requested_range(segmentation):
    assert MIN_SEGMENTS <= segmentation["k"] <= MAX_SEGMENTS


def test_every_customer_gets_a_segment(dataset, segmentation):
    labeled = segmentation["labeled"]
    assert len(labeled) == len(dataset.rfm)
    assert labeled["segment_id"].notna().all()


def test_profile_customer_counts_sum_to_total(dataset, segmentation):
    profile = segmentation["profile"]
    assert profile["size"].sum() == len(dataset.rfm)


def test_profile_revenue_sums_to_total_monetary(dataset, segmentation):
    profile = segmentation["profile"]
    assert profile["revenue"].sum() == pytest.approx(dataset.rfm["monetary_12m"].sum(), rel=1e-9)


def test_growth_potential_is_bounded(segmentation):
    profile = segmentation["profile"]
    assert (profile["growth_potential"] >= 0).all()
    assert (profile["growth_potential"] <= 100).all()
