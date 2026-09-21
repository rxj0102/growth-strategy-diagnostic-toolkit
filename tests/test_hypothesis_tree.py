"""Tests for module 5: hypothesis tree."""

from __future__ import annotations

from gsdt.hypothesis_tree import flatten_tree, _status_from_materiality


def test_root_has_four_top_level_branches(tree):
    assert len(tree.children) == 4


def test_at_least_one_branch_has_sub_issues(tree):
    assert any(len(branch.children) > 0 for branch in tree.children)


def test_every_node_has_non_empty_finding_and_hypothesis(tree):
    for node in flatten_tree(tree):
        assert node.finding.strip() != ""
        assert node.hypothesis.strip() != ""
        assert node.analysis_source.strip() != ""


def test_status_thresholds_are_monotonic_and_bucket_correctly():
    assert _status_from_materiality(50.0) == "Strongly Supported"
    assert _status_from_materiality(20.0) == "Supported"
    assert _status_from_materiality(7.0) == "Partially Supported"
    assert _status_from_materiality(1.0) == "Not Supported"
