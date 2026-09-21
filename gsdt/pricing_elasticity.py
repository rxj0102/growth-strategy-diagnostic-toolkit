"""Module 4: Pricing elasticity / willingness-to-pay analysis.

Builds a synthetic willingness-to-pay (WTP) survey layered on top of the
active customer base (from ``gsdt.synthetic_data``), then simulates several
price-change scenarios using a constant-elasticity demand model:

    retained_fraction(price_change_pct) = clip((1 + price_change_pct) ** (-elasticity), 0, 1)

Elasticity varies by company-size band -- SMB accounts are the most
price-sensitive, Enterprise accounts the least, which is the typical
pattern in B2B subscription pricing research. Gross margin is computed by
holding each customer's dollar cost-to-serve fixed at its current-price
baseline (short-run cost structure does not change just because list price
changes) and comparing it against the *retention-weighted expected*
revenue under the new price.

The WTP survey is reported as a complementary, independent cross-check
(what % of customers say a scenario's price exceeds what they'd pay) rather
than being multiplied directly into the elasticity math, so the two
methodologies (stated preference vs. revealed/assumed elasticity) can be
compared rather than conflated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED_OFFSET = 3

ELASTICITY_BY_BAND = {"SMB": 1.4, "Mid-Market": 0.9, "Enterprise": 0.4}
GROSS_MARGIN_BY_BAND = {"SMB": 0.68, "Mid-Market": 0.74, "Enterprise": 0.78}

DEFAULT_SCENARIOS = (
    {"name": "Baseline (no change)", "price_change_pct": 0.00},
    {"name": "Moderate increase (+5%)", "price_change_pct": 0.05},
    {"name": "Aggressive increase (+12%)", "price_change_pct": 0.12},
)


def generate_wtp_survey(
    customers: pd.DataFrame,
    rfm: pd.DataFrame,
    seed: int | None = None,
) -> pd.DataFrame:
    """Build a synthetic per-customer pricing base: current price, a
    simulated maximum willingness-to-pay, and baseline cost-to-serve.

    Restricted to currently-active customers (``rfm.is_active``), since
    willingness-to-pay for churned customers is not observable/meaningful.
    WTP is simulated as the current price times a lognormal multiplier
    whose center rises with engagement (more engaged customers tolerate a
    higher price before switching away).
    """
    rng = np.random.default_rng(42 + SEED_OFFSET if seed is None else seed)

    active = rfm[rfm["is_active"]].merge(
        customers[["customer_id", "company_size_band"]].rename(
            columns={"company_size_band": "company_size_band_ref"}
        ),
        on="customer_id",
        how="left",
    )
    if active.empty:
        raise ValueError("No active customers available to build a pricing survey")

    band = active["company_size_band"].values
    engagement = active["engagement_score"].values
    current_price = active["monetary_12m"].values

    # WTP multiplier center rises modestly with engagement (0-100 -> +0% to +22%).
    wtp_center = 1.0 + (engagement / 100.0) * 0.22
    wtp_multiplier = rng.lognormal(mean=np.log(wtp_center), sigma=0.08)
    wtp_price = current_price * wtp_multiplier

    margin = np.array([GROSS_MARGIN_BY_BAND[b] for b in band])
    baseline_cost = current_price * (1.0 - margin)

    pricing_base = pd.DataFrame(
        {
            "customer_id": active["customer_id"].values,
            "company_size_band": band,
            "current_price": current_price,
            "wtp_price": np.round(wtp_price, 2),
            "baseline_cost": np.round(baseline_cost, 2),
        }
    )
    return pricing_base


def simulate_pricing_scenario(
    pricing_base: pd.DataFrame,
    price_change_pct: float,
    elasticity_by_band: dict = ELASTICITY_BY_BAND,
) -> pd.DataFrame:
    """Apply one price-change scenario and return per-customer expected
    outcomes: new_price, retained_fraction, expected_revenue,
    expected_gross_profit, wtp_exceeded (bool, informational).
    """
    df = pricing_base.copy()
    elasticity = df["company_size_band"].map(elasticity_by_band)
    if elasticity.isna().any():
        raise ValueError("elasticity_by_band does not cover every company_size_band present")

    df["new_price"] = df["current_price"] * (1.0 + price_change_pct)
    df["retained_fraction"] = np.clip((1.0 + price_change_pct) ** (-elasticity), 0.0, 1.0)
    df["expected_revenue"] = df["retained_fraction"] * df["new_price"]
    df["expected_gross_profit"] = df["retained_fraction"] * (df["new_price"] - df["baseline_cost"])
    df["wtp_exceeded"] = df["new_price"] > df["wtp_price"]
    return df


def run_pricing_scenarios(
    pricing_base: pd.DataFrame,
    scenarios: tuple[dict, ...] = DEFAULT_SCENARIOS,
    elasticity_by_band: dict = ELASTICITY_BY_BAND,
) -> pd.DataFrame:
    """Run each scenario and return a one-row-per-scenario comparison table."""
    n_customers = len(pricing_base)
    baseline_revenue = None

    rows = []
    for scenario in scenarios:
        detail = simulate_pricing_scenario(pricing_base, scenario["price_change_pct"], elasticity_by_band)
        revenue = detail["expected_revenue"].sum()
        gross_profit = detail["expected_gross_profit"].sum()
        retained_customers = detail["retained_fraction"].sum()

        if baseline_revenue is None and scenario["price_change_pct"] == 0.0:
            baseline_revenue = revenue

        rows.append(
            {
                "scenario": scenario["name"],
                "price_change_pct": scenario["price_change_pct"],
                "retained_customers": retained_customers,
                "retention_rate_pct": retained_customers / n_customers * 100.0,
                "projected_revenue": revenue,
                "projected_gross_profit": gross_profit,
                "gross_margin_pct": (gross_profit / revenue * 100.0) if revenue else 0.0,
                "pct_wtp_exceeded": detail["wtp_exceeded"].mean() * 100.0,
            }
        )

    result = pd.DataFrame(rows)
    if baseline_revenue is None:
        baseline_revenue = result.loc[result["price_change_pct"].idxmin(), "projected_revenue"]
    result["revenue_change_vs_baseline_pct"] = (result["projected_revenue"] / baseline_revenue - 1.0) * 100.0
    return result


if __name__ == "__main__":
    from gsdt.synthetic_data import generate_full_dataset

    data = generate_full_dataset()
    pricing_base = generate_wtp_survey(data.customers, data.rfm)
    results = run_pricing_scenarios(pricing_base)
    pd.set_option("display.width", 140)
    print(results.round(2).to_string(index=False))
