"""
src/iaa.py
----------
Inter-annotator agreement utilities for the LongBench diagnostic pilot.

Functions
---------
cohen_kappa(labels_a, labels_b) -> float
    Compute Cohen's κ over two annotation lists.

summarize_disagreements(labels_a, labels_b, instance_ids) -> list[dict]
    Return records for every position where the two annotators disagree.
"""

from sklearn.metrics import cohen_kappa_score

VALID_LABELS = {"RF", "RsF", "INC"}


def _validate(labels: list[str], name: str) -> None:
    """Raise ValueError if any label is not in VALID_LABELS."""
    invalid = [l for l in labels if l not in VALID_LABELS]
    if invalid:
        raise ValueError(
            f"Annotator {name} contains invalid label(s): {invalid}. "
            f"Allowed values are: {sorted(VALID_LABELS)}"
        )


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """
    Compute Cohen's κ between two annotators.

    Parameters
    ----------
    labels_a : list[str]
        Labels from annotator A. Each element must be one of RF, RsF, INC.
    labels_b : list[str]
        Labels from annotator B. Same length as labels_a.

    Returns
    -------
    float
        Cohen's κ score in the range [-1, 1].

    Raises
    ------
    ValueError
        If the lists have different lengths or contain invalid labels.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"Label lists must have equal length. "
            f"Got len(labels_a)={len(labels_a)}, len(labels_b)={len(labels_b)}."
        )
    if len(labels_a) == 0:
        raise ValueError("Label lists must not be empty.")

    _validate(labels_a, "A")
    _validate(labels_b, "B")

    return cohen_kappa_score(labels_a, labels_b)


def summarize_disagreements(
    labels_a: list[str],
    labels_b: list[str],
    instance_ids: list[str],
) -> list[dict]:
    """
    Return a list of disagreement records.

    Parameters
    ----------
    labels_a : list[str]
        Labels from annotator A.
    labels_b : list[str]
        Labels from annotator B.
    instance_ids : list[str]
        Instance identifiers, one per position. Must have the same length as
        labels_a and labels_b.

    Returns
    -------
    list[dict]
        Each dict has keys: instance_id, annotator_a, annotator_b.
        Only positions where labels_a[i] != labels_b[i] are included.
        Returns an empty list when there are no disagreements.

    Raises
    ------
    ValueError
        If lists have unequal lengths or contain invalid labels.
    """
    if not (len(labels_a) == len(labels_b) == len(instance_ids)):
        raise ValueError(
            "labels_a, labels_b, and instance_ids must all have the same length. "
            f"Got {len(labels_a)}, {len(labels_b)}, {len(instance_ids)}."
        )
    if len(labels_a) == 0:
        raise ValueError("Input lists must not be empty.")

    _validate(labels_a, "A")
    _validate(labels_b, "B")

    return [
        {
            "instance_id": instance_ids[i],
            "annotator_a": labels_a[i],
            "annotator_b": labels_b[i],
        }
        for i in range(len(labels_a))
        if labels_a[i] != labels_b[i]
    ]
