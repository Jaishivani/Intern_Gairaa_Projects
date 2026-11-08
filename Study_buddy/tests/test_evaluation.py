"""Retrieval metric tests."""

from evaluation.eval_retrieval import _measure


def test_precision_recall_and_mrr_use_expected_pages() -> None:
    """Page hits are measured at fixed k and ranked reciprocal position."""
    ranked = [
        [{"page": 4}, {"page": 3}],
        [{"page": 2}, {"page": 5}],
    ]
    precision, recall, mrr = _measure(ranked, [4, 5], top_k=4)
    assert precision == 0.25
    assert recall == 1.0
    assert mrr == 0.75
