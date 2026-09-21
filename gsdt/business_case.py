"""Module 6: Business case -- prioritized growth initiatives.

Derives a set of candidate growth initiatives directly from the outputs of
modules 1-4 (no numbers are hand-typed), scores each on an impact/effort
matrix, and returns the top-N (default 3) as the recommended business case.

Candidates considered
-----------------------
1. Retention program        -- recover a share of the churn + contraction
                                revenue concentrated in At-Risk/Dormant
                                segments (module 1 + module 2).
2. Structured upsell motion  -- accelerate organic expansion in the
                                highest growth-potential segment (module 1 +
                                module 2).
3. Adjacent-market entry     -- launch the Healthcare SMB expansion sized in
                                module 3.
4. Pricing action            -- take the best risk-adjusted pricing scenario
                                from module 4.

All four are scored; ``prioritize_initiatives`` returns the ranked list, and
the business case headline is the top 3 (a 4th, lower-priority candidate can
still be inspected via ``derive_initiatives``).

Costs and paybacks are modeled with a small number of clearly-documented
unit-cost assumptions (e.g. cost per at-risk account of a retention
outreach motion). These are illustrative planning assumptions, not fitted
to any real benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Documented cost / margin assumptions
# ---------------------------------------------------------------------------

BLENDED_GROSS_MARGIN = 0.72
NEW_MARKET_GROSS_MARGIN = 0.65  # lower during ramp: onboarding, localization support

RETENTION_RECOVERY_RATE = 0.30  # share of at-risk churn+contraction a CS motion can recover in year 1
RETENTION_COST_PER_ACCOUNT = 220.0  # annual outreach/success cost per at-risk account

UPSELL_UPLIFT_FACTOR = 0.40  # incremental acceleration of organic expansion from a structured motion
UPSELL_COST_PER_ACCOUNT = 150.0  # annual campaign + CSM time per account in the target segment

NEW_MARKET_GTM_INVESTMENT = 650_000.0  # one-time: GTM hires, compliance certification, launch marketing

PRICING_IMPLEMENTATION_COST = 75_000.0  # one-time: billing changes, contract amendments, comms

GROWTH_WINDOW_YEARS = 3


@dataclass
class Initiative:
    name: str
    description: str
    revenue_impact_annual: float
    gross_profit_impact_annual: float
    implementation_cost: float
    payback_months: float
    source_refs: tuple[str, ...]
    impact_score: float = 0.0
    effort_score: float = 0.0
    quadrant: str = ""
    priority_rank: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "revenue_impact_annual": round(self.revenue_impact_annual, 0),
            "gross_profit_impact_annual": round(self.gross_profit_impact_annual, 0),
            "implementation_cost": round(self.implementation_cost, 0),
            "payback_months": round(self.payback_months, 1) if np.isfinite(self.payback_months) else None,
            "impact_score": round(self.impact_score, 1),
            "effort_score": round(self.effort_score, 1),
            "quadrant": self.quadrant,
            "priority_rank": self.priority_rank,
            "source_refs": list(self.source_refs),
        }


def _payback_months(implementation_cost: float, gross_profit_impact_annual: float) -> float:
    if gross_profit_impact_annual <= 0:
        return float("inf")
    return implementation_cost / (gross_profit_impact_annual / 12.0)


def derive_initiatives(
    segment_profile: pd.DataFrame,
    bridge_latest: pd.DataFrame,
    market_sizing_result: dict,
    pricing_results: pd.DataFrame,
) -> list[Initiative]:
    """Build the four candidate initiatives from real module outputs."""

    # --- 1. Retention program -------------------------------------------------
    at_risk = segment_profile[segment_profile["label"].isin(["At-Risk", "Dormant / Lapsed"])]
    at_risk_seg_ids = [s for s in at_risk["segment_id"] if s in bridge_latest.index]
    recoverable_run_rate = float(
        sum(abs(bridge_latest.loc[s, "churn"]) + abs(bridge_latest.loc[s, "contraction"]) for s in at_risk_seg_ids)
    )
    at_risk_accounts = int(at_risk["size"].sum())
    retention_revenue_impact = recoverable_run_rate * RETENTION_RECOVERY_RATE
    retention_cost = at_risk_accounts * RETENTION_COST_PER_ACCOUNT
    retention_gp_impact = retention_revenue_impact * BLENDED_GROSS_MARGIN
    retention_initiative = Initiative(
        name="Proactive retention program for At-Risk accounts",
        description=(
            f"Stand up a targeted customer-success motion (health scoring, executive business reviews, "
            f"win-back offers) for the {at_risk_accounts:,} accounts in At-Risk/Dormant segments, recovering "
            f"an estimated {RETENTION_RECOVERY_RATE:.0%} of their churn+contraction run-rate."
        ),
        revenue_impact_annual=retention_revenue_impact,
        gross_profit_impact_annual=retention_gp_impact,
        implementation_cost=retention_cost,
        payback_months=_payback_months(retention_cost, retention_gp_impact),
        source_refs=("customer_segmentation.profile_segments", "growth_decomposition.compute_growth_bridge"),
    )

    # --- 2. Structured upsell motion ------------------------------------------
    eligible = segment_profile[~segment_profile["label"].isin(["At-Risk", "Dormant / Lapsed"])]
    top_segment = eligible.sort_values("growth_potential", ascending=False).iloc[0]
    top_seg_id = top_segment["segment_id"]
    base_expansion_run_rate = float(bridge_latest.loc[top_seg_id, "expansion"]) if top_seg_id in bridge_latest.index else 0.0
    upsell_revenue_impact = base_expansion_run_rate * UPSELL_UPLIFT_FACTOR
    upsell_cost = int(top_segment["size"]) * UPSELL_COST_PER_ACCOUNT
    upsell_gp_impact = upsell_revenue_impact * BLENDED_GROSS_MARGIN
    upsell_initiative = Initiative(
        name=f"Structured upsell motion for '{top_segment['label']}' segment",
        description=(
            f"Deploy a proactive expansion playbook (usage-based upsell triggers, tiered packaging) into the "
            f"'{top_segment['label']}' segment ({int(top_segment['size']):,} accounts, growth potential "
            f"{top_segment['growth_potential']:.0f}/100), accelerating its organic expansion rate by "
            f"{UPSELL_UPLIFT_FACTOR:.0%}."
        ),
        revenue_impact_annual=upsell_revenue_impact,
        gross_profit_impact_annual=upsell_gp_impact,
        implementation_cost=upsell_cost,
        payback_months=_payback_months(upsell_cost, upsell_gp_impact),
        source_refs=("customer_segmentation.profile_segments", "growth_decomposition.compute_growth_bridge"),
    )

    # --- 3. Adjacent-market entry ----------------------------------------------
    reconciliation = market_sizing_result["reconciliation"]
    reconciled_som_3yr = float(reconciliation["reconciled_som"])
    new_market_revenue_impact = reconciled_som_3yr / GROWTH_WINDOW_YEARS
    new_market_gp_impact = new_market_revenue_impact * NEW_MARKET_GROSS_MARGIN
    new_market_initiative = Initiative(
        name="Launch Healthcare SMB adjacent-market expansion",
        description=(
            f"Stand up a focused go-to-market motion (reference customers, compliance certification, channel "
            f"partnership) to capture the reconciled ${reconciled_som_3yr:,.0f} 3-year SOM in Healthcare SMB "
            "across North America and EMEA, ramping to run-rate over 3 years."
        ),
        revenue_impact_annual=new_market_revenue_impact,
        gross_profit_impact_annual=new_market_gp_impact,
        implementation_cost=NEW_MARKET_GTM_INVESTMENT,
        payback_months=_payback_months(NEW_MARKET_GTM_INVESTMENT, new_market_gp_impact),
        source_refs=("market_sizing.market_sizing_summary",),
    )

    # --- 4. Pricing action -------------------------------------------------------
    baseline_row = pricing_results.iloc[(pricing_results["price_change_pct"] - 0.0).abs().idxmin()]
    candidate_rows = pricing_results[pricing_results["price_change_pct"] != 0.0]
    best_row = candidate_rows.loc[candidate_rows["projected_gross_profit"].idxmax()]
    pricing_revenue_impact = float(best_row["projected_revenue"] - baseline_row["projected_revenue"])
    pricing_gp_impact = float(best_row["projected_gross_profit"] - baseline_row["projected_gross_profit"])
    pricing_initiative = Initiative(
        name=f"Execute pricing action: {best_row['scenario']}",
        description=(
            f"Roll out a {best_row['price_change_pct']:.0%} list-price change at next renewal, projected to "
            f"retain {best_row['retention_rate_pct']:.1f}% of customers at a {best_row['gross_margin_pct']:.1f}% "
            "gross margin."
        ),
        revenue_impact_annual=pricing_revenue_impact,
        gross_profit_impact_annual=pricing_gp_impact,
        implementation_cost=PRICING_IMPLEMENTATION_COST,
        payback_months=_payback_months(PRICING_IMPLEMENTATION_COST, pricing_gp_impact),
        source_refs=("pricing_elasticity.run_pricing_scenarios",),
    )

    return [retention_initiative, upsell_initiative, new_market_initiative, pricing_initiative]


def _classify_quadrant(impact_score: float, effort_score: float, impact_threshold: float, effort_threshold: float) -> str:
    if impact_score >= impact_threshold and effort_score < effort_threshold:
        return "Quick Win"
    if impact_score >= impact_threshold and effort_score >= effort_threshold:
        return "Major Project"
    if impact_score < impact_threshold and effort_score < effort_threshold:
        return "Fill-in"
    return "Reconsider"


def prioritize_initiatives(initiatives: list[Initiative]) -> list[Initiative]:
    """Score every candidate on an impact/effort matrix and rank them.

    - impact_score (0-100): gross profit impact scaled against the highest
      of the candidates.
    - effort_score (0-100): implementation cost scaled against the highest
      of the candidates (higher cost = higher effort).
    - quadrant: Quick Win / Major Project / Fill-in / Reconsider, using the
      median impact and effort scores as the quadrant boundaries.
    - priority_rank: sorted descending by (impact_score - effort_score).

    Returns a NEW list (does not mutate the input), sorted best-first.
    """
    if not initiatives:
        return []

    max_gp = max(i.gross_profit_impact_annual for i in initiatives) or 1.0
    max_cost = max(i.implementation_cost for i in initiatives) or 1.0

    scored = []
    for init in initiatives:
        impact_score = max(init.gross_profit_impact_annual, 0.0) / max_gp * 100.0
        effort_score = init.implementation_cost / max_cost * 100.0
        scored.append((init, impact_score, effort_score))

    impact_median = float(np.median([s[1] for s in scored]))
    effort_median = float(np.median([s[2] for s in scored]))

    ranked = sorted(scored, key=lambda s: (s[1] - s[2]), reverse=True)

    result = []
    for rank, (init, impact_score, effort_score) in enumerate(ranked, start=1):
        init.impact_score = impact_score
        init.effort_score = effort_score
        init.quadrant = _classify_quadrant(impact_score, effort_score, impact_median, effort_median)
        init.priority_rank = rank
        result.append(init)
    return result


def impact_effort_matrix(initiatives: list[Initiative]) -> pd.DataFrame:
    """Tabular view of the impact/effort matrix, one row per initiative."""
    return pd.DataFrame(
        [
            {
                "priority_rank": i.priority_rank,
                "name": i.name,
                "impact_score": round(i.impact_score, 1),
                "effort_score": round(i.effort_score, 1),
                "quadrant": i.quadrant,
                "revenue_impact_annual": round(i.revenue_impact_annual, 0),
                "gross_profit_impact_annual": round(i.gross_profit_impact_annual, 0),
                "implementation_cost": round(i.implementation_cost, 0),
                "payback_months": round(i.payback_months, 1) if np.isfinite(i.payback_months) else None,
            }
            for i in sorted(initiatives, key=lambda i: i.priority_rank)
        ]
    )


def build_business_case(
    segment_profile: pd.DataFrame,
    bridge_latest: pd.DataFrame,
    market_sizing_result: dict,
    pricing_results: pd.DataFrame,
    top_n: int = 3,
) -> tuple[list[Initiative], list[Initiative]]:
    """Convenience entry point.

    Returns (top_n_initiatives, all_ranked_initiatives).
    """
    candidates = derive_initiatives(segment_profile, bridge_latest, market_sizing_result, pricing_results)
    ranked = prioritize_initiatives(candidates)
    return ranked[:top_n], ranked


if __name__ == "__main__":
    from gsdt.synthetic_data import generate_full_dataset
    from gsdt.customer_segmentation import run_segmentation
    from gsdt.growth_decomposition import compute_multi_year_bridge
    from gsdt.market_sizing import market_sizing_summary
    from gsdt.pricing_elasticity import generate_wtp_survey, run_pricing_scenarios

    data = generate_full_dataset()
    labeled, profile, k = run_segmentation(data.rfm)
    seg_map = labeled.set_index("customer_id")["segment_id"]
    bridges = compute_multi_year_bridge(data.panel, segment_map=seg_map)
    latest_bridge = bridges[max(bridges.keys())]

    market = market_sizing_summary(data.customers)
    pricing_base = generate_wtp_survey(data.customers, data.rfm)
    pricing_results = run_pricing_scenarios(pricing_base)

    top3, all_ranked = build_business_case(profile, latest_bridge, market, pricing_results)
    pd.set_option("display.width", 140)
    print("=== Full ranked candidate list ===")
    print(impact_effort_matrix(all_ranked).to_string(index=False))
    print("\n=== Top 3 recommended initiatives ===")
    for init in top3:
        print(f"\n#{init.priority_rank} {init.name} [{init.quadrant}]")
        print(f"  {init.description}")
        print(
            f"  Revenue impact: ${init.revenue_impact_annual:,.0f}/yr | "
            f"Cost: ${init.implementation_cost:,.0f} | "
            f"Payback: {init.payback_months:.1f} months"
        )
