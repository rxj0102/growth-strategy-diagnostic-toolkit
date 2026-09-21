"""Tests for module 7: executive summary memo."""

from __future__ import annotations

import os

from gsdt.executive_summary import generate_executive_summary, write_summary


def _build_memo(tree, business_case_result, segmentation, cum_bridge, market, pricing_results):
    top3, _all = business_case_result
    return generate_executive_summary(tree, top3, segmentation["profile"], cum_bridge, market, pricing_results)


def test_memo_contains_core_sections(tree, business_case_result, segmentation, cum_bridge, market, pricing_results):
    memo = _build_memo(tree, business_case_result, segmentation, cum_bridge, market, pricing_results)
    for heading in ["## Situation", "## Where to focus", "## Recommended initiatives", "## Next steps"]:
        assert heading in memo


def test_memo_is_source_traceable(tree, business_case_result, segmentation, cum_bridge, market, pricing_results):
    memo = _build_memo(tree, business_case_result, segmentation, cum_bridge, market, pricing_results)
    assert "*(source:" in memo


def test_write_summary_persists_to_disk(tmp_path, tree, business_case_result, segmentation, cum_bridge, market, pricing_results):
    memo = _build_memo(tree, business_case_result, segmentation, cum_bridge, market, pricing_results)
    out_path = tmp_path / "executive_summary.md"
    write_summary(memo, str(out_path))
    assert os.path.exists(out_path)
    with open(out_path, encoding="utf-8") as f:
        assert f.read() == memo
