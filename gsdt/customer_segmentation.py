"""Module 1: Customer segmentation.

Segments the synthetic Meridian Works customer base (see
``gsdt.synthetic_data``) into 4-6 behavioral segments using RFM-style
features: Recency (days since last transaction), Frequency (transactions in
trailing 12 months), Monetary (trailing-12-month revenue), and Engagement
(0-100 product-usage score).

Methodology
-----------
1. Standardize the four RFM features (zero mean, unit variance) so no single
   feature dominates the distance metric.
2. Fit K-Means for each candidate k in ``k_range`` and pick the k with the
   best silhouette score (a data-driven choice within the requested 4-6
   segment range, rather than an arbitrary fixed k).
3. Profile each resulting cluster: customer count, % of base, total and %
   of trailing-12-month revenue, average RFM values, and a heuristic 0-100
   "growth potential" score.
4. Assign each cluster a human-readable label from its centroid
   characteristics (e.g. high monetary + high engagement + low recency =
   "Champions / Growth Engine").

Growth potential score
-----------------------
A simple, transparent composite (not a black box): high engagement and high
frequency and low recency all raise growth potential; it intentionally does
NOT reward high monetary directly, so already-large accounts don't
automatically look "high potential" -- the score is about *headroom and
propensity to expand*, not current size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RFM_FEATURES = ["recency_days", "frequency_12m", "monetary_12m", "engagement_score"]

MIN_SEGMENTS = 4
MAX_SEGMENTS = 6
RANDOM_STATE = 42


def segment_customers(
    rfm: pd.DataFrame,
    k_range: range = range(MIN_SEGMENTS, MAX_SEGMENTS + 1),
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, int]:
    """Cluster customers on standardized RFM features.

    Returns (labeled_df, chosen_k) where ``labeled_df`` is ``rfm`` with an
    added integer ``segment_id`` column.
    """
    if not set(RFM_FEATURES).issubset(rfm.columns):
        missing = set(RFM_FEATURES) - set(rfm.columns)
        raise ValueError(f"rfm is missing required columns: {missing}")
    if len(rfm) < max(k_range) * 2:
        raise ValueError("Not enough customers to fit the requested k_range")

    X = StandardScaler().fit_transform(rfm[RFM_FEATURES].values)

    best_k, best_score, best_labels = None, -np.inf, None
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    labeled = rfm.copy()
    labeled["segment_id"] = best_labels
    return labeled, best_k


def _growth_potential_score(row: pd.Series, recency_p95: float) -> float:
    """Heuristic 0-100 growth-potential score for a single customer.

    Rewards high engagement, high transaction frequency, and recent
    activity; explicitly excludes current spend so it measures headroom /
    propensity rather than existing size.
    """
    recency_component = 100.0 * (1.0 - min(row["recency_days"], recency_p95) / recency_p95)
    frequency_component = min(row["frequency_12m"] / 20.0, 1.0) * 100.0
    engagement_component = row["engagement_score"]
    score = 0.45 * engagement_component + 0.30 * recency_component + 0.25 * frequency_component
    return float(np.clip(score, 0, 100))


def profile_segments(labeled: pd.DataFrame) -> pd.DataFrame:
    """Build a per-segment profile: size, revenue contribution, RFM
    centroids, and growth potential.
    """
    if "segment_id" not in labeled.columns:
        raise ValueError("labeled must contain a segment_id column (run segment_customers first)")

    df = labeled.copy()
    recency_p95 = float(df["recency_days"].quantile(0.95)) or 1.0
    df["growth_potential"] = df.apply(_growth_potential_score, axis=1, recency_p95=recency_p95)

    total_customers = len(df)
    total_revenue = df["monetary_12m"].sum()

    rows = []
    for seg_id, g in df.groupby("segment_id"):
        rows.append(
            {
                "segment_id": seg_id,
                "size": len(g),
                "pct_of_customers": len(g) / total_customers * 100.0,
                "revenue": g["monetary_12m"].sum(),
                "pct_of_revenue": (g["monetary_12m"].sum() / total_revenue * 100.0) if total_revenue else 0.0,
                "avg_recency_days": g["recency_days"].mean(),
                "avg_frequency_12m": g["frequency_12m"].mean(),
                "avg_monetary_12m": g["monetary_12m"].mean(),
                "avg_engagement_score": g["engagement_score"].mean(),
                "pct_active": g["is_active"].mean() * 100.0 if "is_active" in g else np.nan,
                "growth_potential": g["growth_potential"].mean(),
            }
        )
    profile = pd.DataFrame(rows).sort_values("revenue", ascending=False).reset_index(drop=True)
    profile["label"] = profile.apply(_label_segment, axis=1, profile=profile)
    return profile


def _label_segment(row: pd.Series, profile: pd.DataFrame) -> str:
    """Assign a human-readable name to a segment from its centroid shape,
    relative to the other segments in the same profile.
    """
    monetary_rank = profile["avg_monetary_12m"].rank(ascending=False)[row.name]
    recency_rank = profile["avg_recency_days"].rank(ascending=True)[row.name]  # low recency = good
    engagement_rank = profile["avg_engagement_score"].rank(ascending=False)[row.name]
    growth_rank = profile["growth_potential"].rank(ascending=False)[row.name]
    pct_active = row.get("pct_active", 100.0)

    n = len(profile)
    if pct_active < 40:
        return "Dormant / Lapsed"
    if monetary_rank <= n / 3 and engagement_rank <= n / 3 and recency_rank <= n / 3:
        return "Champions / Growth Engine"
    if growth_rank <= n / 3 and monetary_rank > n / 3:
        return "Rising / High-Potential"
    if recency_rank > 2 * n / 3 or engagement_rank > 2 * n / 3:
        return "At-Risk"
    if monetary_rank <= n / 3:
        return "Loyal Core"
    return "Steady / Onboarding"


def run_segmentation(rfm: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Convenience entry point: cluster and profile in one call.

    Returns (labeled_df, profile_df, chosen_k).
    """
    labeled, k = segment_customers(rfm)
    profile = profile_segments(labeled)
    return labeled, profile, k


if __name__ == "__main__":
    from gsdt.synthetic_data import generate_full_dataset

    data = generate_full_dataset()
    labeled, profile, k = run_segmentation(data.rfm)
    pd.set_option("display.width", 120)
    print(f"Chosen number of segments: {k}\n")
    print(profile.to_string(index=False))
