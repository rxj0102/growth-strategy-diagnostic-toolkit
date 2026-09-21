"""Module 7: Executive summary memo.

Synthesizes the hypothesis tree (module 5), the underlying data (modules
1-4), and the prioritized initiatives (module 6) into a single-page,
source-traceable markdown memo. Every quantitative claim in the memo is
tagged with the module/function that produced it, so a reader (or a
reviewer) can trace every number back to its analysis.
"""

from __future__ import annotations

from gsdt.business_case import Initiative
from gsdt.hypothesis_tree import HypothesisNode, flatten_tree


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def generate_executive_summary(
    tree: HypothesisNode,
    top_initiatives: list[Initiative],
    segment_profile,
    cumulative_bridge_df,
    market_sizing_result: dict,
    pricing_results,
    company_name: str = "Meridian Works",
) -> str:
    """Build the one-page markdown executive memo.

    Returns the memo as a markdown string; use ``write_summary`` to persist
    it to disk.
    """
    total_prior_revenue = float(cumulative_bridge_df.loc["TOTAL", "start_revenue"])
    total_end_revenue = float(cumulative_bridge_df.loc["TOTAL", "end_revenue"])
    realized_growth_pct = (total_end_revenue / total_prior_revenue - 1.0) * 100.0

    top_segment = segment_profile.sort_values("revenue", ascending=False).iloc[0]
    reconciliation = market_sizing_result["reconciliation"]

    lines: list[str] = []
    lines.append(f"# {company_name}: Growth Strategy Executive Recommendation")
    lines.append("")
    lines.append(f"**Core question:** {tree.issue}")
    lines.append("")
    lines.append(
        f"**Bottom line:** Revenue grew {_fmt_money(total_prior_revenue)} -> "
        f"{_fmt_money(total_end_revenue)} ({realized_growth_pct:+.1f}%) over the review window. "
        f"The three initiatives below, pursued together, are projected to add "
        f"{_fmt_money(sum(i.revenue_impact_annual for i in top_initiatives))}/year in incremental "
        "revenue against the growth target."
    )
    lines.append("")

    # --- Situation -------------------------------------------------------
    lines.append("## Situation")
    lines.append(
        f"- Customer base: {int(segment_profile['size'].sum()):,} customers across "
        f"{len(segment_profile)} behavioral segments "
        "*(source: customer_segmentation.profile_segments)*."
    )
    lines.append(
        f"- Largest segment by revenue: **{top_segment['label']}** "
        f"({top_segment['pct_of_customers']:.1f}% of customers, {top_segment['pct_of_revenue']:.1f}% of revenue) "
        "*(source: customer_segmentation.profile_segments)*."
    )
    lines.append(
        f"- Growth bridge: new {_fmt_money(cumulative_bridge_df.loc['TOTAL', 'new'])}, "
        f"expansion {_fmt_money(cumulative_bridge_df.loc['TOTAL', 'expansion'])}, "
        f"contraction {_fmt_money(cumulative_bridge_df.loc['TOTAL', 'contraction'])}, "
        f"churn {_fmt_money(cumulative_bridge_df.loc['TOTAL', 'churn'])}, "
        f"price {_fmt_money(cumulative_bridge_df.loc['TOTAL', 'price'])} "
        "*(source: growth_decomposition.cumulative_bridge)*."
    )
    lines.append(
        f"- Adjacent-market opportunity (Healthcare SMB): reconciled 3-year SOM of "
        f"{_fmt_money(reconciliation['reconciled_som'])} (top-down {_fmt_money(reconciliation['som_topdown'])}, "
        f"bottoms-up {_fmt_money(reconciliation['som_bottomsup'])}, {reconciliation['gap_pct']:.0f}% gap) "
        "*(source: market_sizing.market_sizing_summary)*."
    )
    lines.append("")

    # --- Hypothesis tree summary -------------------------------------------
    lines.append("## Where to focus: hypothesis tree")
    for branch in tree.children:
        lines.append(f"- **[{branch.status}]** {branch.issue} -- {branch.finding}")
    lines.append("")

    # --- Recommended initiatives --------------------------------------------
    lines.append("## Recommended initiatives (ranked by impact/effort)")
    lines.append("")
    lines.append("| Rank | Initiative | Quadrant | Revenue Impact/yr | Cost | Payback (mo) |")
    lines.append("|---|---|---|---|---|---|")
    for init in top_initiatives:
        payback = f"{init.payback_months:.1f}" if init.payback_months != float("inf") else "n/a"
        lines.append(
            f"| {init.priority_rank} | {init.name} | {init.quadrant} | "
            f"{_fmt_money(init.revenue_impact_annual)} | {_fmt_money(init.implementation_cost)} | {payback} |"
        )
    lines.append("")
    for init in top_initiatives:
        sources = ", ".join(init.source_refs)
        lines.append(f"**#{init.priority_rank} {init.name}** *(source: {sources})*")
        lines.append(f"  {init.description}")
        lines.append("")

    # --- Next steps -----------------------------------------------------------
    lines.append("## Next steps")
    lines.append(
        "1. Validate the retention/upsell/pricing cost assumptions with Finance and CS leadership before "
        "committing budget."
    )
    lines.append(
        "2. Sequence the Quick Win / lowest-payback initiatives first to fund the Major Project "
        "(adjacent-market entry)."
    )
    lines.append(
        "3. Re-run this diagnostic quarterly as new transaction data lands, to track realized vs. "
        "projected impact."
    )
    lines.append("")
    lines.append(
        "*This memo was generated programmatically from the growth-strategy-diagnostic-toolkit; every "
        "figure above is traceable to the cited module/function and to synthetic (not real) underlying data.*"
    )

    return "\n".join(lines)


def write_summary(markdown_text: str, path: str) -> None:
    """Persist the memo to disk as a markdown file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown_text)


if __name__ == "__main__":
    from gsdt.synthetic_data import generate_full_dataset
    from gsdt.customer_segmentation import run_segmentation
    from gsdt.growth_decomposition import compute_multi_year_bridge, cumulative_bridge
    from gsdt.market_sizing import market_sizing_summary
    from gsdt.pricing_elasticity import generate_wtp_survey, run_pricing_scenarios
    from gsdt.hypothesis_tree import build_hypothesis_tree
    from gsdt.business_case import build_business_case

    data = generate_full_dataset()
    labeled, profile, k = run_segmentation(data.rfm)
    seg_map = labeled.set_index("customer_id")["segment_id"]
    bridges = compute_multi_year_bridge(data.panel, segment_map=seg_map)
    cum = cumulative_bridge(bridges)
    latest_bridge = bridges[max(bridges.keys())]

    market = market_sizing_summary(data.customers)
    pricing_base = generate_wtp_survey(data.customers, data.rfm)
    pricing_results = run_pricing_scenarios(pricing_base)

    tree = build_hypothesis_tree(profile, cum, market, pricing_results)
    top3, _ = build_business_case(profile, latest_bridge, market, pricing_results)

    memo = generate_executive_summary(tree, top3, profile, cum, market, pricing_results)
    print(memo)
