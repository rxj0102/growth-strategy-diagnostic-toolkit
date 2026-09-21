"""Module 2: Growth decomposition (revenue growth bridge).

Decomposes year-over-year revenue change into five drivers that reconcile
*exactly* to the observed change, for the total company and for each
customer segment:

    start_revenue + new + expansion + contraction + churn + price = end_revenue

Definitions (per customer, for a transition from year t-1 to year t)
----------------------------------------------------------------------
- ``new``: revenue in year t from customers who were not active in t-1
  (new-logo acquisition). Always >= 0.
- ``churn``: negative of revenue in year t-1 from customers who were active
  in t-1 but are no longer active in t (full-account churn). Always <= 0.
- For customers active in BOTH years, the year-over-year revenue change is
  split into a price component and a volume component:
    price_effect  = revenue[t-1] * (price_index[t] / price_index[t-1] - 1)
    volume_effect = (revenue[t] - revenue[t-1]) - price_effect
  ``volume_effect`` is bucketed into ``expansion`` (>= 0, upsell/usage
  growth) or ``contraction`` (< 0, downsell/usage decline).
- ``price``: sum of ``price_effect`` across customers active in both years
  -- the portion of revenue change attributable to the company-wide list
  price change, isolated from volume/usage change.

This decomposition is an *exact arithmetic identity* on realized revenue
(see the docstring proof in the README / tests) -- it holds regardless of
how the underlying revenue was generated, which is why the reconciliation
tests can assert equality to floating-point tolerance rather than "roughly
matches".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gsdt.synthetic_data import PRICE_INDEX, YEARS

RECONCILIATION_TOLERANCE = 1e-6


def _wide_panel(panel: pd.DataFrame) -> pd.DataFrame:
    revenue_wide = panel.pivot(index="customer_id", columns="year", values="revenue")
    active_wide = panel.pivot(index="customer_id", columns="year", values="active")
    return revenue_wide, active_wide


def compute_growth_bridge(
    panel: pd.DataFrame,
    year_from: int,
    year_to: int,
    price_index: dict[int, float] = PRICE_INDEX,
    segment_map: pd.Series | dict | None = None,
) -> pd.DataFrame:
    """Compute the growth bridge for a single year-over-year transition.

    ``segment_map`` optionally maps customer_id -> segment label; when
    provided, one row per segment is returned in addition to a ``TOTAL``
    row. When omitted, only the ``TOTAL`` row is returned.

    Returns a DataFrame indexed by segment with columns:
        start_revenue, new, expansion, contraction, churn, price,
        end_revenue, reconciled_end, reconciliation_diff
    """
    revenue_wide, active_wide = _wide_panel(panel)
    if year_from not in revenue_wide.columns or year_to not in revenue_wide.columns:
        raise ValueError(f"panel does not contain both years {year_from} and {year_to}")

    rev_from = revenue_wide[year_from].fillna(0.0)
    rev_to = revenue_wide[year_to].fillna(0.0)
    act_from = active_wide[year_from].fillna(False)
    act_to = active_wide[year_to].fillna(False)

    is_new = (~act_from) & act_to
    is_churn = act_from & (~act_to)
    is_existing = act_from & act_to

    price_ratio = price_index[year_to] / price_index[year_from] - 1.0

    detail = pd.DataFrame(index=revenue_wide.index)
    detail["new"] = np.where(is_new, rev_to, 0.0)
    detail["churn"] = np.where(is_churn, -rev_from, 0.0)

    price_effect = np.where(is_existing, rev_from * price_ratio, 0.0)
    delta = np.where(is_existing, rev_to - rev_from, 0.0)
    volume_effect = np.where(is_existing, delta - price_effect, 0.0)

    detail["price"] = price_effect
    detail["expansion"] = np.where(volume_effect > 0, volume_effect, 0.0)
    detail["contraction"] = np.where(volume_effect < 0, volume_effect, 0.0)
    detail["start_revenue"] = rev_from
    detail["end_revenue"] = rev_to

    if segment_map is not None:
        detail["segment"] = detail.index.map(
            segment_map if isinstance(segment_map, dict) else segment_map.to_dict()
        )
        if detail["segment"].isna().any():
            raise ValueError("segment_map does not cover every customer_id in the panel")
    else:
        detail["segment"] = "TOTAL"

    agg_cols = ["start_revenue", "new", "expansion", "contraction", "churn", "price", "end_revenue"]
    bridge = detail.groupby("segment")[agg_cols].sum()

    if segment_map is not None:
        total_row = bridge[agg_cols].sum()
        total_row.name = "TOTAL"
        bridge = pd.concat([bridge, total_row.to_frame().T])

    bridge["reconciled_end"] = (
        bridge["start_revenue"] + bridge["new"] + bridge["expansion"] + bridge["contraction"] + bridge["churn"] + bridge["price"]
    )
    bridge["reconciliation_diff"] = bridge["reconciled_end"] - bridge["end_revenue"]
    bridge["year_from"] = year_from
    bridge["year_to"] = year_to
    return bridge


def compute_multi_year_bridge(
    panel: pd.DataFrame,
    years: tuple[int, ...] = YEARS,
    price_index: dict[int, float] = PRICE_INDEX,
    segment_map: pd.Series | dict | None = None,
) -> dict[tuple[int, int], pd.DataFrame]:
    """Compute a growth bridge for every consecutive pair of years.

    Returns a dict keyed by (year_from, year_to) -> bridge DataFrame.
    """
    years = tuple(sorted(years))
    if len(years) < 2:
        raise ValueError("Need at least two years to compute a growth bridge")
    bridges = {}
    for year_from, year_to in zip(years[:-1], years[1:]):
        bridges[(year_from, year_to)] = compute_growth_bridge(
            panel, year_from, year_to, price_index=price_index, segment_map=segment_map
        )
    return bridges


def cumulative_bridge(bridges: dict[tuple[int, int], pd.DataFrame]) -> pd.DataFrame:
    """Roll multiple single-year bridges up into one cumulative bridge per
    segment, spanning the earliest start year to the latest end year.

    Because each single-year bridge reconciles exactly, the sum of
    components across transitions reconciles exactly to
    (last end_revenue - first start_revenue) as well (telescoping sum).
    """
    pairs = sorted(bridges.keys())
    first_start = bridges[pairs[0]][["start_revenue"]].rename(columns={"start_revenue": "start_revenue"})
    component_cols = ["new", "expansion", "contraction", "churn", "price"]

    summed = None
    for pair in pairs:
        df = bridges[pair][component_cols]
        summed = df.copy() if summed is None else summed.add(df, fill_value=0.0)

    last_end = bridges[pairs[-1]][["end_revenue"]]

    out = first_start.join(summed, how="outer").join(last_end, how="outer").fillna(0.0)
    out["reconciled_end"] = out["start_revenue"] + out[component_cols].sum(axis=1)
    out["reconciliation_diff"] = out["reconciled_end"] - out["end_revenue"]
    return out


def validate_bridge(bridge: pd.DataFrame, tol: float = RECONCILIATION_TOLERANCE) -> bool:
    """Raise AssertionError if any row of a bridge DataFrame fails to
    reconcile within ``tol``; otherwise return True.
    """
    bad = bridge[bridge["reconciliation_diff"].abs() > tol]
    if not bad.empty:
        raise AssertionError(f"Growth bridge failed to reconcile for segments: {list(bad.index)}")
    return True


if __name__ == "__main__":
    from gsdt.synthetic_data import generate_full_dataset

    data = generate_full_dataset()
    bridges = compute_multi_year_bridge(data.panel)
    pd.set_option("display.width", 120)
    for (yf, yt), bridge in bridges.items():
        validate_bridge(bridge)
        print(f"\n=== Growth bridge FY{yf} -> FY{yt} ===")
        print(bridge.round(0).to_string())

    cum = cumulative_bridge(bridges)
    validate_bridge(cum)
    print(f"\n=== Cumulative bridge FY{YEARS[0]} -> FY{YEARS[-1]} ===")
    print(cum.round(0).to_string())
