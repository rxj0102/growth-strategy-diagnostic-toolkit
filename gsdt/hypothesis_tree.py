"""Module 5: Hypothesis tree.

Structures the core growth question --

    "Where should Meridian Works focus to grow revenue {GROWTH_TARGET_PCT:.0%}
     over the next {GROWTH_WINDOW_YEARS} years?"

-- as an issue tree with four top-level branches, each carrying a testable
hypothesis and one or two sub-issues. Every node's ``finding`` and
``status`` are computed from the ACTUAL outputs of modules 1-4 (segment
profiles, the growth bridge, market sizing, and pricing scenarios) rather
than being hard-coded prose, so re-running the tree against a different
random seed or a different dataset produces different findings.

Status logic
-------------
Each node estimates the dollar contribution its underlying lever could make
toward the 3-year growth target, expressed as a % of that target
(``materiality_pct``), and buckets it:
    >= 40%  -> "Strongly Supported"
    >= 15%  -> "Supported"
    >= 5%   -> "Partially Supported"
    <  5%   -> "Not Supported"
"""

from __future__ import annotations

from dataclasses import dataclass, field

GROWTH_TARGET_PCT = 0.25
GROWTH_WINDOW_YEARS = 3

STATUS_THRESHOLDS = (
    (40.0, "Strongly Supported"),
    (15.0, "Supported"),
    (5.0, "Partially Supported"),
    (0.0, "Not Supported"),
)


def _status_from_materiality(materiality_pct: float) -> str:
    for threshold, label in STATUS_THRESHOLDS:
        if materiality_pct >= threshold:
            return label
    return "Not Supported"


@dataclass
class HypothesisNode:
    issue: str
    hypothesis: str
    analysis_source: str
    finding: str
    materiality_pct: float
    status: str
    children: list["HypothesisNode"] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "issue": self.issue,
            "hypothesis": self.hypothesis,
            "analysis_source": self.analysis_source,
            "finding": self.finding,
            "materiality_pct": round(self.materiality_pct, 1),
            "status": self.status,
            "children": [c.to_dict() for c in self.children],
        }


def _leaf(issue, hypothesis, analysis_source, finding, driver_dollars, growth_target_dollars) -> HypothesisNode:
    materiality_pct = (driver_dollars / growth_target_dollars * 100.0) if growth_target_dollars else 0.0
    status = _status_from_materiality(materiality_pct)
    return HypothesisNode(issue, hypothesis, analysis_source, finding, materiality_pct, status)


def build_hypothesis_tree(
    segment_profile,
    cumulative_bridge_df,
    market_sizing_result: dict,
    pricing_results,
    growth_target_pct: float = GROWTH_TARGET_PCT,
    growth_window_years: int = GROWTH_WINDOW_YEARS,
) -> HypothesisNode:
    """Build the full hypothesis tree from the outputs of modules 1-4.

    Parameters mirror what each module produces:
        segment_profile        -> customer_segmentation.profile_segments(...)
        cumulative_bridge_df   -> growth_decomposition.cumulative_bridge(...)
        market_sizing_result   -> market_sizing.market_sizing_summary(...)
        pricing_results        -> pricing_elasticity.run_pricing_scenarios(...)
    """
    total_prior_revenue = float(cumulative_bridge_df.loc["TOTAL", "start_revenue"])
    growth_target_dollars = total_prior_revenue * growth_target_pct

    core_question = (
        f"Where should Meridian Works focus to grow revenue {growth_target_pct:.0%} "
        f"(~${growth_target_dollars:,.0f}) over the next {growth_window_years} years?"
    )

    # --- Branch 1: Expand wallet share in the existing base -----------------
    cumulative_expansion = float(cumulative_bridge_df.loc["TOTAL", "expansion"])
    top_segments = segment_profile.sort_values("growth_potential", ascending=False).head(2)
    expansion_children = []
    for _, seg in top_segments.iterrows():
        seg_id = seg["segment_id"]
        if seg_id in cumulative_bridge_df.index:
            seg_expansion = float(cumulative_bridge_df.loc[seg_id, "expansion"])
        else:
            seg_expansion = 0.0
        expansion_children.append(
            _leaf(
                issue=f"Segment '{seg['label']}' upsell headroom",
                hypothesis=(
                    f"The '{seg['label']}' segment (growth potential score "
                    f"{seg['growth_potential']:.0f}/100) has enough expansion headroom to be a "
                    "primary upsell target."
                ),
                analysis_source="customer_segmentation.profile_segments + growth_decomposition.cumulative_bridge",
                finding=(
                    f"'{seg['label']}' holds {seg['pct_of_customers']:.1f}% of customers and "
                    f"{seg['pct_of_revenue']:.1f}% of revenue, and generated ${seg_expansion:,.0f} of "
                    "cumulative expansion revenue over the review window."
                ),
                driver_dollars=seg_expansion,
                growth_target_dollars=growth_target_dollars,
            )
        )
    branch_expansion = HypothesisNode(
        issue="Expand wallet share within the existing customer base",
        hypothesis="Upsell/cross-sell into already-engaged accounts can materially contribute to the growth target.",
        analysis_source="growth_decomposition.cumulative_bridge (expansion driver)",
        finding=(
            f"Expansion contributed ${cumulative_expansion:,.0f} of cumulative revenue growth over the "
            f"review window (base revenue ${total_prior_revenue:,.0f})."
        ),
        materiality_pct=(cumulative_expansion / growth_target_dollars * 100.0) if growth_target_dollars else 0.0,
        status=_status_from_materiality((cumulative_expansion / growth_target_dollars * 100.0) if growth_target_dollars else 0.0),
        children=expansion_children,
    )

    # --- Branch 2: Reduce churn / contraction --------------------------------
    cumulative_churn = float(cumulative_bridge_df.loc["TOTAL", "churn"])
    cumulative_contraction = float(cumulative_bridge_df.loc["TOTAL", "contraction"])
    at_risk = segment_profile[segment_profile["label"].isin(["At-Risk", "Dormant / Lapsed"])]
    at_risk_revenue = float(at_risk["revenue"].sum())
    at_risk_children = []
    if not at_risk.empty:
        at_risk_children.append(
            _leaf(
                issue="Concentration of revenue at churn/contraction risk",
                hypothesis="Churn and contraction are concentrated in identifiable, targetable segments.",
                analysis_source="customer_segmentation.profile_segments",
                finding=(
                    f"At-risk / dormant segments hold ${at_risk_revenue:,.0f} of trailing-12-month revenue "
                    f"across {int(at_risk['size'].sum())} customers."
                ),
                driver_dollars=abs(cumulative_churn) + abs(cumulative_contraction),
                growth_target_dollars=growth_target_dollars,
            )
        )
    recoverable = abs(cumulative_churn) + abs(cumulative_contraction)
    branch_retention = HypothesisNode(
        issue="Reduce churn and contraction",
        hypothesis="A meaningful share of lost revenue is recoverable through a targeted retention program.",
        analysis_source="growth_decomposition.cumulative_bridge (churn + contraction drivers)",
        finding=(
            f"Churn removed ${abs(cumulative_churn):,.0f} and contraction removed ${abs(cumulative_contraction):,.0f} "
            f"of revenue over the review window (${recoverable:,.0f} combined)."
        ),
        materiality_pct=(recoverable / growth_target_dollars * 100.0) if growth_target_dollars else 0.0,
        status=_status_from_materiality((recoverable / growth_target_dollars * 100.0) if growth_target_dollars else 0.0),
        children=at_risk_children,
    )

    # --- Branch 3: Accelerate new-market entry -------------------------------
    reconciliation = market_sizing_result["reconciliation"]
    reconciled_som = float(reconciliation["reconciled_som"])
    cumulative_new = float(cumulative_bridge_df.loc["TOTAL", "new"])
    branch_new_market = HypothesisNode(
        issue="Accelerate new-customer acquisition into adjacent markets",
        hypothesis="An adjacent-market expansion (Healthcare SMB) can add a new, sizeable growth vector beyond organic new-logo growth in the existing footprint.",
        analysis_source="market_sizing.market_sizing_summary + growth_decomposition.cumulative_bridge (new driver)",
        finding=(
            f"Existing-footprint new-logo acquisition contributed ${cumulative_new:,.0f} over the review window. "
            f"The Healthcare SMB expansion reconciles to a ${reconciled_som:,.0f} 3-year SOM "
            f"(top-down ${reconciliation['som_topdown']:,.0f} vs. bottoms-up ${reconciliation['som_bottomsup']:,.0f}, "
            f"{reconciliation['gap_pct']:.0f}% gap)."
        ),
        materiality_pct=(reconciled_som / growth_target_dollars * 100.0) if growth_target_dollars else 0.0,
        status=_status_from_materiality((reconciled_som / growth_target_dollars * 100.0) if growth_target_dollars else 0.0),
        children=[],
    )

    # --- Branch 4: Optimize pricing ------------------------------------------
    baseline_row = pricing_results.iloc[(pricing_results["price_change_pct"] - 0.0).abs().idxmin()]
    best_row = pricing_results.loc[pricing_results["projected_gross_profit"].idxmax()]
    annual_uplift = float(best_row["projected_revenue"] - baseline_row["projected_revenue"])
    three_year_uplift = annual_uplift * growth_window_years
    pricing_children = [
        _leaf(
            issue=f"Scenario: {row['scenario']}",
            hypothesis="This pricing scenario improves revenue and/or margin without unacceptable retention loss.",
            analysis_source="pricing_elasticity.run_pricing_scenarios",
            finding=(
                f"Projected retention {row['retention_rate_pct']:.1f}%, revenue "
                f"{row['revenue_change_vs_baseline_pct']:+.1f}% vs. baseline, gross margin {row['gross_margin_pct']:.1f}%."
            ),
            driver_dollars=max(row["projected_revenue"] - baseline_row["projected_revenue"], 0.0) * growth_window_years,
            growth_target_dollars=growth_target_dollars,
        )
        for _, row in pricing_results.iterrows()
        if row["price_change_pct"] != 0.0
    ]
    branch_pricing = HypothesisNode(
        issue="Optimize pricing and monetization",
        hypothesis="A disciplined price increase captures incremental revenue and margin with acceptable retention impact.",
        analysis_source="pricing_elasticity.run_pricing_scenarios",
        finding=(
            f"Best-margin scenario ('{best_row['scenario']}') projects ${annual_uplift:,.0f}/yr incremental revenue "
            f"vs. baseline (${three_year_uplift:,.0f} over {growth_window_years} years) at "
            f"{best_row['retention_rate_pct']:.1f}% customer retention."
        ),
        materiality_pct=(three_year_uplift / growth_target_dollars * 100.0) if growth_target_dollars else 0.0,
        status=_status_from_materiality((three_year_uplift / growth_target_dollars * 100.0) if growth_target_dollars else 0.0),
        children=pricing_children,
    )

    root = HypothesisNode(
        issue=core_question,
        hypothesis="Revenue growth targets are achievable through a combination of the four levers below.",
        analysis_source="synthesis of modules 1-4",
        finding=(
            f"Base revenue ${total_prior_revenue:,.0f}; growth target ${growth_target_dollars:,.0f} "
            f"over {growth_window_years} years."
        ),
        materiality_pct=100.0,
        status="Root",
        children=[branch_expansion, branch_retention, branch_new_market, branch_pricing],
    )
    return root


def render_tree(node: HypothesisNode, indent: int = 0) -> str:
    """Render the tree as indented plain text."""
    pad = "  " * indent
    lines = [f"{pad}- [{node.status}] {node.issue}"]
    if indent > 0:
        lines.append(f"{pad}    Hypothesis: {node.hypothesis}")
        lines.append(f"{pad}    Finding:    {node.finding}")
        lines.append(f"{pad}    Source:     {node.analysis_source}")
    for child in node.children:
        lines.append(render_tree(child, indent + 1))
    return "\n".join(lines)


def flatten_tree(node: HypothesisNode) -> list[HypothesisNode]:
    """Return every node in the tree (including the root) as a flat list."""
    out = [node]
    for child in node.children:
        out.extend(flatten_tree(child))
    return out


if __name__ == "__main__":
    from gsdt.synthetic_data import generate_full_dataset
    from gsdt.customer_segmentation import run_segmentation
    from gsdt.growth_decomposition import compute_multi_year_bridge, cumulative_bridge
    from gsdt.market_sizing import market_sizing_summary
    from gsdt.pricing_elasticity import generate_wtp_survey, run_pricing_scenarios

    data = generate_full_dataset()
    labeled, profile, k = run_segmentation(data.rfm)
    seg_map = labeled.set_index("customer_id")["segment_id"]
    bridges = compute_multi_year_bridge(data.panel, segment_map=seg_map)
    cum = cumulative_bridge(bridges)

    market = market_sizing_summary(data.customers)
    pricing_base = generate_wtp_survey(data.customers, data.rfm)
    pricing_results = run_pricing_scenarios(pricing_base)

    tree = build_hypothesis_tree(profile, cum, market, pricing_results)
    print(render_tree(tree))
