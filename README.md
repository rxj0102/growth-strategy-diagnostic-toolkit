# Growth Strategy Diagnostic Toolkit

[![CI](https://github.com/rxj0102/growth-strategy-diagnostic-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/rxj0102/growth-strategy-diagnostic-toolkit/actions/workflows/ci.yml)

A Python toolkit that reproduces the analytical workstream of a
**customer-insight-driven growth strategy engagement** for a mid-market
subscription/services business, end-to-end: segmentation, a growth bridge,
market sizing, pricing elasticity, a hypothesis tree, a prioritized business
case, and a source-traceable executive memo.

All data is **synthetic** — there is no real company, customer, or market
data anywhere in this repository. The synthetic data is generated
deterministically (fixed random seed) and is fully documented so every
downstream number can be traced back to how it was produced.

## The business scenario

**"Meridian Works"** is a fictional mid-market B2B company that sells a
workforce-operations subscription platform (software + an attached
services/support tier) to companies of varying size. It has ~1,200
customers across three company-size bands (SMB / Mid-Market / Enterprise),
seven industries, and three regions (North America, EMEA, APAC).

Leadership has asked: **"Where should we focus to grow revenue 25% over
the next 3 years?"** This toolkit answers that question the way a strategy
consulting team would: segment the base, understand what actually drove
recent growth, size a specific adjacent-market opportunity, test pricing
levers, structure the answer as a hypothesis tree, and land on a
prioritized, costed set of initiatives.

## Methodology and how the data is generated

Everything starts from `gsdt/synthetic_data.py`, the single source of truth
for every module:

1. **Customer roster** (`generate_customers`) — firmographics (company size
   band, industry, region), an acquisition year, and two latent traits: a
   `growth_trend` group (expanding / flat / contracting account) and an
   `engagement_score`. All distributions and rates (company-size mix,
   churn hazards, growth-rate parameters, price-increase schedule, etc.)
   are named module-level constants with inline documentation of the
   assumption behind each one.
2. **Multi-year revenue panel** (`generate_revenue_panel`) — expands the
   roster into a customer x fiscal-year (FY2023-FY2025) panel of realized
   revenue, compounding each customer's individual growth trend over
   tenure, applying a company-wide annual price increase, and stopping the
   moment a customer's simulated churn event fires. This panel is the
   single source of truth for the growth bridge.
3. **RFM snapshot** (`generate_rfm_snapshot`) — Recency/Frequency/
   Monetary/Engagement features derived *from that same panel* as of
   2025-12-31, so segmentation, growth decomposition, and pricing are
   always looking at internally-consistent numbers for the same customers.

Because every module reads from this one generator, a customer's segment,
its churn status, its trailing-12-month revenue, and its role in the
growth bridge are always mutually consistent — there is no risk of, say, a
"churned" customer showing up with positive current-year revenue in the
segmentation output.

## Modules

| # | Module | What it does |
|---|--------|---------------|
| 1 | `gsdt/customer_segmentation.py` | Clusters customers into 4-6 segments on standardized RFM features (K-Means, k chosen by silhouette score); profiles each segment's size, revenue contribution, and a transparent 0-100 "growth potential" score. |
| 2 | `gsdt/growth_decomposition.py` | Builds a year-over-year **growth bridge**: `start_revenue + new + expansion + contraction + churn + price = end_revenue`, by segment and in total. This is an exact arithmetic identity on realized revenue (see docstring), not a fitted model — it reconciles to the cent. |
| 3 | `gsdt/market_sizing.py` | TAM/SAM/SOM for a concrete adjacent-market opportunity (expand the SMB tier into the Healthcare vertical, NA + EMEA), built two ways — top-down from synthetic industry benchmarks, bottoms-up from the *actual* SMB unit economics in the customer base — then reconciled. |
| 4 | `gsdt/pricing_elasticity.py` | A synthetic willingness-to-pay survey plus a constant-elasticity demand model (elasticity varies by company-size band). Simulates 3 pricing scenarios and their revenue/margin impact. |
| 5 | `gsdt/hypothesis_tree.py` | Structures "where should we focus to grow revenue 25% over 3 years?" as an issue tree with 4 branches (expand wallet share, reduce churn, enter adjacent markets, optimize pricing) and sub-issues, each backed by a real finding computed from modules 1-4. |
| 6 | `gsdt/business_case.py` | Derives 4 candidate initiatives straight from modules 1-4's outputs, scores each on an impact/effort matrix (Quick Win / Major Project / Fill-in / Reconsider), and returns the top 3, each with revenue impact, implementation cost, and payback period. |
| 7 | `gsdt/executive_summary.py` | Synthesizes the hypothesis tree, the data, and the prioritized initiatives into a one-page, source-tagged markdown memo — every quantitative claim is annotated with the module/function that produced it. |

## Growth bridge: why the reconciliation is exact

For a customer active in both year *t-1* and *t*:

```
price_effect  = revenue[t-1] * (price_index[t] / price_index[t-1] - 1)
volume_effect = (revenue[t] - revenue[t-1]) - price_effect
```

`volume_effect` buckets into `expansion` (>= 0) or `contraction` (< 0);
`price_effect` sums into the `price` driver. New customers contribute their
full current-year revenue to `new`; customers who left contribute their
full prior-year revenue (negated) to `churn`. Summed across all customers,
this telescopes to `start_revenue + new + expansion + contraction + churn +
price == end_revenue` exactly — see `tests/test_growth_decomposition.py`.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+.

## Running the full diagnostic

```bash
python run_diagnostic.py
```

This runs all seven modules end-to-end against a fresh synthetic dataset,
prints each module's output, and writes the executive memo to
`executive_summary.md`.

Each module can also be run standalone, e.g.:

```bash
python -m gsdt.customer_segmentation
python -m gsdt.growth_decomposition
python -m gsdt.market_sizing
python -m gsdt.pricing_elasticity
python -m gsdt.hypothesis_tree
python -m gsdt.business_case
python -m gsdt.executive_summary
```

## Tests

```bash
python -m pytest -v
```

35 tests across 8 files, covering:
- **Segmentation logic** — chosen segment count within the requested 4-6
  range, every customer assigned, profile totals reconcile to the full
  base, growth-potential score is bounded.
- **Growth-bridge arithmetic reconciliation** — every segment and the
  total reconcile exactly, cumulative multi-year bridges telescope
  correctly, driver signs are structurally correct (new/expansion >= 0,
  churn <= 0), and a tampered bridge is correctly rejected by the
  validator.
- **Market-sizing reconciliation** — the top-down funnel is monotonically
  decreasing (SOM <= SAM <= TAM), bottoms-up unit economics are pulled
  from the real synthetic customer data, the reconciliation gap formula is
  correct, and the reconciled SOM always falls between the two source
  estimates.
- **Pricing scenario math** — revenue equals price times retained
  quantity, higher price increases monotonically reduce retention,
  reported gross margin % matches profit/revenue, and more price-elastic
  segments retain less at the same price increase.
- Structural checks on the hypothesis tree, the business case ranking, and
  the executive summary's section headers and source-traceability tags.

## Repository layout

```
gsdt/
  synthetic_data.py         # shared, seeded data generator (all assumptions documented here)
  customer_segmentation.py  # module 1
  growth_decomposition.py   # module 2
  market_sizing.py          # module 3
  pricing_elasticity.py     # module 4
  hypothesis_tree.py        # module 5
  business_case.py          # module 6
  executive_summary.py      # module 7
tests/
  conftest.py                # session-scoped fixtures (dataset computed once, shared)
  test_synthetic_data.py
  test_customer_segmentation.py
  test_growth_decomposition.py
  test_market_sizing.py
  test_pricing_elasticity.py
  test_hypothesis_tree.py
  test_business_case.py
  test_executive_summary.py
run_diagnostic.py           # end-to-end runner
requirements.txt
```

## Caveats

This is a demonstration toolkit built on synthetic data to illustrate a
growth-strategy analytical methodology. The specific numbers it produces
(revenue figures, market sizes, elasticities) are not real and should not
be used for actual business decisions. The *structure* of the analysis —
how a growth bridge reconciles, how TAM/SAM/SOM should be built two ways
and reconciled, how a hypothesis tree ties issues to evidence, how an
impact/effort matrix prioritizes initiatives — is the reusable part.
