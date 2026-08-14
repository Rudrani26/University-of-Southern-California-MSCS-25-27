"""
tests/test_iaa.py
-----------------
Tests for src/iaa.py — cohen_kappa and summarize_disagreements.

Run with:
    pytest tests/test_iaa.py -v
"""

import pytest
from src.iaa import cohen_kappa, summarize_disagreements


# ---------------------------------------------------------------------------
# cohen_kappa
# ---------------------------------------------------------------------------

def test_cohen_kappa_identical_lists():
    labels = ["RF", "RsF", "INC", "RF", "RsF"]
    assert cohen_kappa(labels, labels) == 1.0


def test_cohen_kappa_fully_opposed():
    a = ["RF",  "RF",  "RF",  "RF",  "RF"]
    b = ["RsF", "RsF", "RsF", "RsF", "RsF"]
    assert cohen_kappa(a, b) < 0.5


def test_cohen_kappa_mixed_returns_valid_range():
    a = ["RF", "RsF", "INC", "RF",  "RsF", "INC", "RF",  "RsF"]
    b = ["RF", "RF",  "INC", "RsF", "RsF", "RF",  "RF",  "INC"]
    kappa = cohen_kappa(a, b)
    assert -1.0 <= kappa <= 1.0


def test_cohen_kappa_invalid_label_raises():
    with pytest.raises(ValueError, match="invalid label"):
        cohen_kappa(["RF", "XYZ"], ["RF", "RF"])


def test_cohen_kappa_unequal_lengths_raises():
    with pytest.raises(ValueError, match="equal length"):
        cohen_kappa(["RF", "RsF"], ["RF"])


def test_cohen_kappa_empty_lists_raises():
    with pytest.raises(ValueError, match="empty"):
        cohen_kappa([], [])


def test_cohen_kappa_invalid_label_in_b_raises():
    with pytest.raises(ValueError, match="invalid label"):
        cohen_kappa(["RF", "RF"], ["RF", "BAD"])


# ---------------------------------------------------------------------------
# summarize_disagreements
# ---------------------------------------------------------------------------

def test_summarize_disagreements_correct_count():
    la  = ["RF",  "RsF", "INC", "RF",  "RsF"]
    lb  = ["RF",  "RF",  "INC", "RsF", "RsF"]
    ids = ["q1",  "q2",  "q3",  "q4",  "q5"]
    result = summarize_disagreements(la, lb, ids)
    assert len(result) == 2


def test_summarize_disagreements_correct_instances():
    la  = ["RF",  "RsF", "INC", "RF",  "RsF"]
    lb  = ["RF",  "RF",  "INC", "RsF", "RsF"]
    ids = ["q1",  "q2",  "q3",  "q4",  "q5"]
    result = summarize_disagreements(la, lb, ids)
    disagreed_ids = {d["instance_id"] for d in result}
    assert disagreed_ids == {"q2", "q4"}


def test_summarize_disagreements_no_false_positives():
    la  = ["RF",  "RsF", "INC", "RF",  "RsF"]
    lb  = ["RF",  "RF",  "INC", "RsF", "RsF"]
    ids = ["q1",  "q2",  "q3",  "q4",  "q5"]
    result = summarize_disagreements(la, lb, ids)
    disagreed_ids = {d["instance_id"] for d in result}
    assert "q1" not in disagreed_ids
    assert "q3" not in disagreed_ids
    assert "q5" not in disagreed_ids


def test_summarize_disagreements_no_disagreements_returns_empty():
    la  = ["RF", "RsF"]
    lb  = ["RF", "RsF"]
    ids = ["q1", "q2"]
    assert summarize_disagreements(la, lb, ids) == []


def test_summarize_disagreements_result_keys():
    la  = ["RF",  "RsF"]
    lb  = ["RsF", "RsF"]
    ids = ["q1",  "q2"]
    result = summarize_disagreements(la, lb, ids)
    assert set(result[0].keys()) == {"instance_id", "annotator_a", "annotator_b"}


def test_summarize_disagreements_invalid_label_raises():
    with pytest.raises(ValueError, match="invalid label"):
        summarize_disagreements(["RF", "BAD"], ["RF", "RF"], ["q1", "q2"])


def test_summarize_disagreements_unequal_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        summarize_disagreements(["RF"], ["RF", "RsF"], ["q1", "q2"])


def test_summarize_disagreements_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        summarize_disagreements([], [], [])
