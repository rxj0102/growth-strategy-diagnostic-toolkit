"""End-to-end runner: executes all seven modules against a fresh synthetic
dataset and writes the executive summary memo to ``executive_summary.md``.

Usage:
    python run_diagnostic.py
"""

from __future__ import annotations

import pandas as pd

from gsdt.synthetic_data import generate_full_dataset
from gsdt.customer_segmentation import run_segmentation
from gsdt.growth_decomposition import compute_multi_year_bridge, cumulative_bridge, validate_bridge
from gsdt.market_sizing import market_sizing_summary
from gsdt.pricing_elasticity import generate_wtp_survey, run_pricing_scenarios
from gsdt.hypothesis_tree import build_hypothesis_tree, render_tree
from gsdt.business_case import build_business_case, impact_effort_matrix
from gsdt.executive_summary import generate_executive_summary, write_summary


def main() -> None:
    pd.set_option("display.width", 140)

    print("1/7 Generating synthetic dataset...")
    data = generate_full_dataset()

    print("2/7 Segmenting customers...")
    labeled, profile, k = run_segmentation(data.rfm)
    print(f"    Chosen segments: {k}")
    print(profile[["segment_id", "label", "size", "pct_of_revenue", "growth_potential"]].round(1).to_string(index=False))

    print("\n3/7 Decomposing growth...")
    seg_map = labeled.set_index("customer_id")["segment_id"]
    bridges = compute_multi_year_bridge(data.panel, segment_map=seg_map)
    for bridge in bridges.values():
        validate_bridge(bridge)
    cum = cumulative_bridge(bridges)
    latest_bridge = bridges[max(bridges.keys())]
    print(cum.round(0).to_string())

    print("\n4/7 Sizing the Healthcare SMB market opportunity...")
    market = market_sizing_summary(data.customers)
    rec = market["reconciliation"]
    print(f"    TAM ${market['topdown']['tam']:,.0f} | SAM ${market['topdown']['sam']:,.0f} | "
          f"SOM top-down ${rec['som_topdown']:,.0f} | SOM bottoms-up ${rec['som_bottomsup']:,.0f} | "
          f"reconciled ${rec['reconciled_som']:,.0f}")

    print("\n5/7 Running pricing scenarios...")
    pricing_base = generate_wtp_survey(data.customers, data.rfm)
    pricing_results = run_pricing_scenarios(pricing_base)
    print(pricing_results.round(2).to_string(index=False))

    print("\n6/7 Building the hypothesis tree...")
    tree = build_hypothesis_tree(profile, cum, market, pricing_results)
    print(render_tree(tree))

    print("\n7/7 Prioritizing initiatives and writing the executive summary...")
    top3, all_ranked = build_business_case(profile, latest_bridge, market, pricing_results)
    print(impact_effort_matrix(all_ranked).to_string(index=False))

    memo = generate_executive_summary(tree, top3, profile, cum, market, pricing_results)
    write_summary(memo, "executive_summary.md")
    print("\nExecutive summary written to executive_summary.md")


if __name__ == "__main__":
    main()
