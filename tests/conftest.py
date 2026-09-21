"""Shared, session-scoped fixtures so the (somewhat expensive) synthetic
dataset and downstream analyses are computed once and reused across the
whole test suite.
"""

from __future__ import annotations

import pytest

from gsdt.synthetic_data import generate_full_dataset
from gsdt.customer_segmentation import run_segmentation
from gsdt.growth_decomposition import compute_multi_year_bridge, cumulative_bridge
from gsdt.market_sizing import market_sizing_summary
from gsdt.pricing_elasticity import generate_wtp_survey, run_pricing_scenarios
from gsdt.hypothesis_tree import build_hypothesis_tree
from gsdt.business_case import build_business_case


@pytest.fixture(scope="session")
def dataset():
    return generate_full_dataset(n=900, seed=42)


@pytest.fixture(scope="session")
def segmentation(dataset):
    labeled, profile, k = run_segmentation(dataset.rfm)
    return {"labeled": labeled, "profile": profile, "k": k}


@pytest.fixture(scope="session")
def segment_map(segmentation):
    return segmentation["labeled"].set_index("customer_id")["segment_id"]


@pytest.fixture(scope="session")
def bridges(dataset, segment_map):
    return compute_multi_year_bridge(dataset.panel, segment_map=segment_map)


@pytest.fixture(scope="session")
def latest_bridge(bridges):
    return bridges[max(bridges.keys())]


@pytest.fixture(scope="session")
def cum_bridge(bridges):
    return cumulative_bridge(bridges)


@pytest.fixture(scope="session")
def market(dataset):
    return market_sizing_summary(dataset.customers)


@pytest.fixture(scope="session")
def pricing_base(dataset):
    return generate_wtp_survey(dataset.customers, dataset.rfm)


@pytest.fixture(scope="session")
def pricing_results(pricing_base):
    return run_pricing_scenarios(pricing_base)


@pytest.fixture(scope="session")
def tree(segmentation, cum_bridge, market, pricing_results):
    return build_hypothesis_tree(segmentation["profile"], cum_bridge, market, pricing_results)


@pytest.fixture(scope="session")
def business_case_result(segmentation, latest_bridge, market, pricing_results):
    return build_business_case(segmentation["profile"], latest_bridge, market, pricing_results)
