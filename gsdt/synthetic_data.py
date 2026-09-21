"""Synthetic data generation shared by every module in the toolkit.

BUSINESS SCENARIO
------------------
"Meridian Works" is a fictional mid-market B2B subscription/services company
that sells a workforce-operations platform (software subscription + an
attached services/support tier) to companies of varying size, across a
handful of industries and three geographic regions.

All data in this project is SYNTHETIC and generated deterministically from a
fixed random seed (``SEED``) so every module, and every test, sees the same
customer base and the same financial history. Nothing here is real customer
or market data.

HOW THE DATA IS GENERATED (read this before trusting a number)
----------------------------------------------------------------
1. ``generate_customers`` creates a static roster of customers with
   firmographic attributes (company size band, industry, region), a join
   (acquisition) year, and two latent traits that drive everything
   downstream: a ``growth_trend`` group (expanding / flat / contracting
   account) and an ``engagement_score`` (0-100, simulated product usage).
2. ``generate_revenue_panel`` expands the roster into a long
   customer-by-fiscal-year panel (FY2023-FY2025) of realized subscription
   revenue. Revenue compounds the customer's growth trend over tenure,
   applies a company-wide annual price increase (``PRICE_INDEX``), and
   stops the moment a customer's simulated churn event fires. This panel is
   the single source of truth for ``growth_decomposition.py``.
3. ``generate_rfm_snapshot`` looks at the panel as of ``ANALYSIS_DATE``
   (2025-12-31) and derives Recency/Frequency/Monetary/Engagement features
   per customer -- the input to ``customer_segmentation.py``. Because the
   RFM features are *derived from the same panel*, a customer's segment,
   its FY2025 revenue, and its churn status are always mutually consistent.

All dollar figures are annualized (ARR-style). All key assumptions are
named module-level constants so they can be inspected, or overridden in a
script/test, instead of being buried in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Documented assumptions
# ---------------------------------------------------------------------------

SEED = 42
N_CUSTOMERS_DEFAULT = 1_200  # within the requested 800-1,500 range

YEARS = (2023, 2024, 2025)
ANALYSIS_DATE = date(2025, 12, 31)

# Company-wide list-price index applied to every active customer each fiscal
# year (a ~4% then ~3% price increase). Used by growth_decomposition.py to
# isolate the "Price" driver of the growth bridge.
PRICE_INDEX = {2023: 1.00, 2024: 1.04, 2025: 1.07}

# Firmographic mix
COMPANY_SIZE_BANDS = ("SMB", "Mid-Market", "Enterprise")
COMPANY_SIZE_WEIGHTS = (0.55, 0.35, 0.10)
# Approximate ARR multiple by band relative to a base SMB subscription.
COMPANY_SIZE_ARR_MULTIPLE = {"SMB": 1.0, "Mid-Market": 3.5, "Enterprise": 12.0}
BASE_ARR = 4_800.0  # base annual subscription value for a 1.0x (SMB) account

INDUSTRIES = (
    "Professional Services",
    "Healthcare",
    "Retail & eCommerce",
    "Financial Services",
    "Technology",
    "Manufacturing",
    "Education",
)
INDUSTRY_WEIGHTS = (0.22, 0.14, 0.16, 0.12, 0.14, 0.13, 0.09)

REGIONS = ("North America", "EMEA", "APAC")
REGION_WEIGHTS = (0.55, 0.27, 0.18)

# Acquisition-year mix. Customers who "joined" before 2023 are the existing
# base at the start of the growth-decomposition window; customers joining in
# 2023-2025 are the "New" driver in the growth bridge.
JOIN_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
JOIN_YEAR_WEIGHTS = (0.10, 0.15, 0.20, 0.20, 0.17, 0.18)

# Growth-trend group mix and the per-customer annual compounding rate drawn
# for each group (mean, std of a normal distribution, clipped).
GROWTH_TREND_GROUPS = ("expand", "flat", "contract")
GROWTH_TREND_WEIGHTS = (0.30, 0.45, 0.25)
GROWTH_TREND_PARAMS = {
    "expand": {"mean": 0.18, "std": 0.07, "clip": (0.02, 0.45)},
    "flat": {"mean": 0.00, "std": 0.04, "clip": (-0.10, 0.10)},
    "contract": {"mean": -0.15, "std": 0.08, "clip": (-0.45, -0.02)},
}
CUMULATIVE_GROWTH_CLIP = (0.40, 3.00)

# Base annual churn hazard by company-size band, before growth-trend
# adjustment. Contracting accounts are more likely to churn; expanding
# accounts less so.
BASE_CHURN_HAZARD = {"SMB": 0.12, "Mid-Market": 0.06, "Enterprise": 0.02}
CHURN_HAZARD_ADJUSTMENT = {"expand": -0.03, "flat": 0.00, "contract": 0.05}

REVENUE_NOISE_STD = 0.05  # idiosyncratic lognormal noise on realized revenue


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(SEED if seed is None else seed)


# ---------------------------------------------------------------------------
# Step 1: base customer roster
# ---------------------------------------------------------------------------

def generate_customers(n: int = N_CUSTOMERS_DEFAULT, seed: int | None = None) -> pd.DataFrame:
    """Generate the static customer roster with firmographics and latent traits.

    Returns a DataFrame with one row per customer and columns:
        customer_id, company_size_band, industry, region, join_year,
        growth_trend_group, annual_growth_rate, base_arr, engagement_score,
        churn_year (NaN if the customer is still active at end of 2025)
    """
    if not 1 <= n <= 100_000:
        raise ValueError("n must be a positive, reasonably-sized customer count")

    rng = _rng(seed)

    customer_id = [f"CUST-{i:05d}" for i in range(1, n + 1)]
    company_size_band = rng.choice(COMPANY_SIZE_BANDS, size=n, p=COMPANY_SIZE_WEIGHTS)
    industry = rng.choice(INDUSTRIES, size=n, p=INDUSTRY_WEIGHTS)
    region = rng.choice(REGIONS, size=n, p=REGION_WEIGHTS)
    join_year = rng.choice(JOIN_YEARS, size=n, p=JOIN_YEAR_WEIGHTS)
    growth_trend_group = rng.choice(GROWTH_TREND_GROUPS, size=n, p=GROWTH_TREND_WEIGHTS)

    annual_growth_rate = np.empty(n, dtype=float)
    for group, params in GROWTH_TREND_PARAMS.items():
        mask = growth_trend_group == group
        draws = rng.normal(params["mean"], params["std"], size=mask.sum())
        annual_growth_rate[mask] = np.clip(draws, *params["clip"])

    size_multiple = np.array([COMPANY_SIZE_ARR_MULTIPLE[b] for b in company_size_band])
    # lognormal idiosyncratic sizing noise within a band
    size_noise = rng.lognormal(mean=0.0, sigma=0.25, size=n)
    base_arr = BASE_ARR * size_multiple * size_noise

    # Engagement score: correlated with growth trend (expanding accounts use
    # the product more) plus noise, bounded to [0, 100].
    trend_engagement_shift = {"expand": 18.0, "flat": 0.0, "contract": -16.0}
    engagement_center = 55.0 + np.array(
        [trend_engagement_shift[g] for g in growth_trend_group]
    )
    engagement_score = np.clip(rng.normal(engagement_center, 12.0), 0.0, 100.0)

    # Churn simulation: for each customer, roll annual churn hazard for each
    # year they are active (from join_year through 2025) until it fires.
    churn_hazard = np.array(
        [
            BASE_CHURN_HAZARD[b] + CHURN_HAZARD_ADJUSTMENT[g]
            for b, g in zip(company_size_band, growth_trend_group)
        ]
    )
    churn_hazard = np.clip(churn_hazard, 0.005, 0.60)

    churn_year = np.full(n, np.nan)
    last_simulated_year = max(YEARS)
    for i in range(n):
        for yr in range(int(join_year[i]), last_simulated_year + 1):
            if rng.random() < churn_hazard[i]:
                churn_year[i] = yr
                break

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "company_size_band": company_size_band,
            "industry": industry,
            "region": region,
            "join_year": join_year.astype(int),
            "growth_trend_group": growth_trend_group,
            "annual_growth_rate": annual_growth_rate,
            "base_arr": base_arr,
            "engagement_score": engagement_score,
            "churn_hazard": churn_hazard,
            "churn_year": churn_year,
        }
    )
    return df


# ---------------------------------------------------------------------------
# Step 2: multi-year revenue panel
# ---------------------------------------------------------------------------

def generate_revenue_panel(customers: pd.DataFrame, years=YEARS, seed: int | None = None) -> pd.DataFrame:
    """Expand the customer roster into a long (customer_id, year) revenue panel.

    A customer is "active" in year ``t`` iff ``join_year <= t`` and
    (``churn_year`` is NaN or ``t < churn_year``). Revenue for an active
    customer compounds its individual annual growth rate over tenure and
    applies the company-wide ``PRICE_INDEX`` for that year, plus a small
    idiosyncratic noise term.
    """
    rng = _rng(seed if seed is not None else SEED + 1)

    rows = []
    for _, c in customers.iterrows():
        noise = rng.lognormal(mean=0.0, sigma=REVENUE_NOISE_STD, size=len(years))
        for noise_val, year in zip(noise, years):
            active = (year >= c["join_year"]) and (
                pd.isna(c["churn_year"]) or year < c["churn_year"]
            )
            if not active:
                rows.append(
                    {"customer_id": c["customer_id"], "year": year, "revenue": 0.0, "active": False}
                )
                continue
            tenure = year - c["join_year"]
            cumulative_growth = np.clip(
                (1.0 + c["annual_growth_rate"]) ** tenure, *CUMULATIVE_GROWTH_CLIP
            )
            revenue = c["base_arr"] * cumulative_growth * PRICE_INDEX[year] * noise_val
            rows.append(
                {"customer_id": c["customer_id"], "year": year, "revenue": round(float(revenue), 2), "active": True}
            )

    panel = pd.DataFrame(rows)
    return panel


# ---------------------------------------------------------------------------
# Step 3: RFM snapshot derived from the panel
# ---------------------------------------------------------------------------

def generate_rfm_snapshot(
    customers: pd.DataFrame,
    panel: pd.DataFrame,
    as_of: date = ANALYSIS_DATE,
    seed: int | None = None,
) -> pd.DataFrame:
    """Derive Recency/Frequency/Monetary/Engagement features as of ``as_of``.

    - Monetary: trailing-twelve-month revenue, i.e. the final panel year's
      revenue for that customer (0 for churned customers).
    - Recency: days since last purchase/renewal event. Active customers get
      a small recency (more engaged customers transact more recently);
      churned customers get a recency proportional to time since churn.
    - Frequency: simulated count of billing/usage transactions in the
      trailing twelve months, scaled by company size band and engagement.
    - Engagement: carried over from the customer roster's latent trait.
    """
    rng = _rng(seed if seed is not None else SEED + 2)
    final_year = max(panel["year"])

    final_year_revenue = (
        panel[panel["year"] == final_year]
        .set_index("customer_id")["revenue"]
        .reindex(customers["customer_id"])
        .fillna(0.0)
        .values
    )
    final_year_active = (
        panel[panel["year"] == final_year]
        .set_index("customer_id")["active"]
        .reindex(customers["customer_id"])
        .fillna(False)
        .values
    )

    n = len(customers)
    engagement = customers["engagement_score"].values
    band = customers["company_size_band"].values
    churn_year = customers["churn_year"].values

    band_frequency_base = {"SMB": 6.0, "Mid-Market": 14.0, "Enterprise": 30.0}
    freq_lambda = np.array([band_frequency_base[b] for b in band]) * (
        0.4 + engagement / 100.0
    )
    frequency_12m = rng.poisson(np.clip(freq_lambda, 0.1, None))
    frequency_12m = np.where(final_year_active, frequency_12m, rng.integers(0, 3, size=n))

    recency_days = np.empty(n, dtype=float)
    for i in range(n):
        if final_year_active[i]:
            # More engaged customers renewed/transacted more recently.
            recency_days[i] = rng.uniform(1, 20) + (100 - engagement[i]) * 0.6
        else:
            years_since_churn = final_year - churn_year[i] if not pd.isna(churn_year[i]) else 1
            years_since_churn = max(years_since_churn, 0) + 1
            recency_days[i] = years_since_churn * 365 + rng.uniform(0, 120)

    rfm = pd.DataFrame(
        {
            "customer_id": customers["customer_id"].values,
            "company_size_band": band,
            "industry": customers["industry"].values,
            "region": customers["region"].values,
            "recency_days": np.round(recency_days, 1),
            "frequency_12m": frequency_12m.astype(int),
            "monetary_12m": np.round(final_year_revenue, 2),
            "engagement_score": np.round(engagement, 1),
            "is_active": final_year_active,
        }
    )
    return rfm


# ---------------------------------------------------------------------------
# Convenience bundle
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDataset:
    customers: pd.DataFrame
    panel: pd.DataFrame
    rfm: pd.DataFrame


def generate_full_dataset(n: int = N_CUSTOMERS_DEFAULT, seed: int = SEED) -> SyntheticDataset:
    """Generate the full, internally-consistent synthetic dataset used across
    every module: the customer roster, the multi-year revenue panel, and the
    RFM snapshot derived from that same panel.
    """
    customers = generate_customers(n=n, seed=seed)
    panel = generate_revenue_panel(customers, seed=seed + 1)
    rfm = generate_rfm_snapshot(customers, panel, seed=seed + 2)
    return SyntheticDataset(customers=customers, panel=panel, rfm=rfm)
