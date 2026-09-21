"""Module 3: Market sizing (TAM / SAM / SOM) for a specific growth opportunity.

Growth opportunity analyzed
----------------------------
Meridian Works is well established with Mid-Market and Enterprise customers
in Professional Services, Technology, and Financial Services. This module
sizes a concrete *adjacent* expansion opportunity:

    "Expand the SMB-tier product into the Healthcare vertical across
     North America and EMEA."

Both a top-down (industry-data driven) and a bottoms-up (unit-economics x
addressable-accounts driven) estimate are built independently, then
reconciled. All industry figures below are SYNTHETIC placeholders standing
in for what would, in a real engagement, come from analyst reports (e.g.
IBISWorld, Gartner) or government business-count statistics -- they are
documented explicitly so the assumption, not just the output, can be
challenged.

Top-down approach
------------------
TAM = target establishments x category annual spend per establishment
SAM = TAM narrowed to the company's ICP-fit and currently-serviceable slice
SOM = SAM x a benchmark 3-year obtainable share

Bottoms-up approach
---------------------
SOM (bottoms-up) = reachable accounts (via direct sales + channel capacity)
                    x realistic win rate
                    x average new-logo ACV
                      (ACV is estimated FROM THE ACTUAL SMB CUSTOMER BASE
                       in ``gsdt.synthetic_data``, not assumed)

Reconciliation
---------------
The two SOM estimates are built from unrelated inputs and will not match
exactly -- that gap is the point: it tells a strategy team where the
top-down market story and the bottoms-up go-to-market plan disagree. This
module reports both, the % gap, and a blended reconciled estimate.
"""

from __future__ import annotations

import pandas as pd

# ---------------------------------------------------------------------------
# Top-down assumptions (synthetic "industry report" inputs -- documented)
# ---------------------------------------------------------------------------

TARGET_INDUSTRY = "Healthcare"
TARGET_REGIONS = ("North America", "EMEA")

# Estimated count of healthcare establishments in the target regions that
# are in Meridian's general size range (synthetic government/analyst-style
# business-count figure).
TOTAL_TARGET_ESTABLISHMENTS = 84_000

# Fraction of those establishments that fit Meridian's ideal customer
# profile (headcount band, distributed/shift-based workforce, etc.).
ICP_FIT_RATE = 0.35

# Average annual category spend per ICP-fit establishment on
# workforce-operations software + services (synthetic analyst benchmark).
CATEGORY_ANNUAL_SPEND_PER_ESTABLISHMENT = 3_200.0

# Fraction of ICP-fit establishments Meridian can realistically serve given
# current localization, compliance certifications, and channel presence.
SERVICEABLE_RATE = 0.55

# Benchmark 3-year obtainable share of SAM, informed by Meridian's
# historical penetration when entering a comparably-sized adjacent vertical
# (documented assumption, not derived from this module's own data).
OBTAINABLE_SHARE_3YR_TOPDOWN = 0.12


def topdown_sizing(
    total_establishments: float = TOTAL_TARGET_ESTABLISHMENTS,
    icp_fit_rate: float = ICP_FIT_RATE,
    category_spend_per_establishment: float = CATEGORY_ANNUAL_SPEND_PER_ESTABLISHMENT,
    serviceable_rate: float = SERVICEABLE_RATE,
    obtainable_share_3yr: float = OBTAINABLE_SHARE_3YR_TOPDOWN,
) -> dict:
    """Top-down TAM/SAM/SOM for the Healthcare-vertical expansion opportunity."""
    if not (0 <= icp_fit_rate <= 1 and 0 <= serviceable_rate <= 1 and 0 <= obtainable_share_3yr <= 1):
        raise ValueError("Rate parameters must be between 0 and 1")

    tam = total_establishments * category_spend_per_establishment
    icp_fit_establishments = total_establishments * icp_fit_rate
    serviceable_establishments = icp_fit_establishments * serviceable_rate
    sam = serviceable_establishments * category_spend_per_establishment
    som = sam * obtainable_share_3yr

    return {
        "approach": "top_down",
        "target_industry": TARGET_INDUSTRY,
        "target_regions": TARGET_REGIONS,
        "total_establishments": total_establishments,
        "icp_fit_establishments": icp_fit_establishments,
        "serviceable_establishments": serviceable_establishments,
        "tam": tam,
        "sam": sam,
        "som": som,
        "obtainable_share_3yr": obtainable_share_3yr,
    }


# ---------------------------------------------------------------------------
# Bottoms-up assumptions
# ---------------------------------------------------------------------------

# Reachable accounts over 3 years via direct sales capacity allocated to the
# vertical push plus one channel-partner relationship (synthetic GTM
# capacity plan, independent of the top-down establishment count above).
REACHABLE_ACCOUNTS_3YR = 9_000

# Realistic win rate for a focused vertical entry with reference customers.
WIN_RATE = 0.18

# Healthcare accounts are assumed to carry a modest premium over the
# existing SMB book due to a compliance/audit-trail add-on.
HEALTHCARE_ACV_PREMIUM = 1.10


def bottomsup_sizing(
    customers: pd.DataFrame,
    reachable_accounts_3yr: float = REACHABLE_ACCOUNTS_3YR,
    win_rate: float = WIN_RATE,
    acv_premium: float = HEALTHCARE_ACV_PREMIUM,
) -> dict:
    """Bottoms-up SOM using unit economics observed in the actual SMB base.

    ``customers`` is the synthetic customer roster (``base_arr`` column) from
    ``gsdt.synthetic_data.generate_customers``. The average SMB ACV is
    computed from that real (synthetic) data, not assumed.
    """
    if not (0 <= win_rate <= 1):
        raise ValueError("win_rate must be between 0 and 1")
    if "base_arr" not in customers.columns or "company_size_band" not in customers.columns:
        raise ValueError("customers must contain base_arr and company_size_band columns")

    smb = customers[customers["company_size_band"] == "SMB"]
    if smb.empty:
        raise ValueError("No SMB customers found to derive bottoms-up unit economics")

    avg_smb_acv = float(smb["base_arr"].mean())
    avg_new_logo_acv = avg_smb_acv * acv_premium

    won_accounts = reachable_accounts_3yr * win_rate
    som_bottomsup = won_accounts * avg_new_logo_acv

    return {
        "approach": "bottoms_up",
        "reachable_accounts_3yr": reachable_accounts_3yr,
        "win_rate": win_rate,
        "avg_existing_smb_acv": avg_smb_acv,
        "avg_new_logo_acv": avg_new_logo_acv,
        "won_accounts_3yr": won_accounts,
        "som": som_bottomsup,
    }


def reconcile_sizing(topdown: dict, bottomsup: dict, weight_bottomsup: float = 0.5) -> dict:
    """Reconcile the top-down and bottoms-up SOM estimates.

    Returns the two source SOMs, the % gap between them, and a blended
    ``reconciled_som`` (weighted average, default equal weight -- top-down
    reflects market-level benchmarks, bottoms-up reflects Meridian's actual
    unit economics and go-to-market capacity, so neither is strictly more
    trustworthy).
    """
    if not 0 <= weight_bottomsup <= 1:
        raise ValueError("weight_bottomsup must be between 0 and 1")

    som_topdown = topdown["som"]
    som_bottomsup = bottomsup["som"]
    if som_topdown <= 0:
        raise ValueError("Top-down SOM must be positive to compute a reconciliation gap")

    gap_pct = (som_bottomsup - som_topdown) / som_topdown * 100.0
    reconciled_som = weight_bottomsup * som_bottomsup + (1 - weight_bottomsup) * som_topdown

    bottomsup_exceeds_sam = som_bottomsup > topdown["sam"]

    return {
        "som_topdown": som_topdown,
        "som_bottomsup": som_bottomsup,
        "gap_pct": gap_pct,
        "weight_bottomsup": weight_bottomsup,
        "reconciled_som": reconciled_som,
        "sam_topdown": topdown["sam"],
        "tam_topdown": topdown["tam"],
        "bottomsup_exceeds_sam": bottomsup_exceeds_sam,
    }


def market_sizing_summary(customers: pd.DataFrame) -> dict:
    """Convenience entry point: run both approaches and reconcile them."""
    topdown = topdown_sizing()
    bottomsup = bottomsup_sizing(customers)
    reconciliation = reconcile_sizing(topdown, bottomsup)
    return {"topdown": topdown, "bottomsup": bottomsup, "reconciliation": reconciliation}


if __name__ == "__main__":
    from gsdt.synthetic_data import generate_full_dataset

    data = generate_full_dataset()
    summary = market_sizing_summary(data.customers)

    td, bu, rec = summary["topdown"], summary["bottomsup"], summary["reconciliation"]
    print(f"Opportunity: Expand into {TARGET_INDUSTRY} SMB, {', '.join(TARGET_REGIONS)}\n")
    print("-- Top-down --")
    print(f"  TAM: ${td['tam']:,.0f}")
    print(f"  SAM: ${td['sam']:,.0f}")
    print(f"  SOM (3yr): ${td['som']:,.0f}")
    print("\n-- Bottoms-up --")
    print(f"  Avg existing SMB ACV: ${bu['avg_existing_smb_acv']:,.0f}")
    print(f"  Avg new-logo (Healthcare) ACV: ${bu['avg_new_logo_acv']:,.0f}")
    print(f"  Won accounts (3yr): {bu['won_accounts_3yr']:,.0f}")
    print(f"  SOM (3yr): ${bu['som']:,.0f}")
    print("\n-- Reconciliation --")
    print(f"  Gap: {rec['gap_pct']:.1f}%")
    print(f"  Reconciled SOM (3yr): ${rec['reconciled_som']:,.0f}")
