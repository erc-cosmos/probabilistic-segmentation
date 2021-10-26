"""Tests for scoring.py."""
import hypothesis
import hypothesis.strategies as st
import pytest
import scoring

segmentations = st.lists(st.floats(min_value=0, allow_infinity=False))


def test_count_matches_tolerance():
    """Check countMatches with a known example."""
    ref = [0, 10, 20]
    guess = [5, 16]
    assert scoring.count_matches(ref, guess, 3) == 0
    assert scoring.count_matches(ref, guess, 4) == 1
    assert scoring.count_matches(ref, guess, 6) == 2


@hypothesis.given(segmentations, st.floats(min_value=0))
def test_count_matches_reflexively(ref, tolerance):
    """Check that everything matches when the reference and estimate are the same."""
    assert scoring.count_matches(ref, ref, tolerance) == len(set(ref))


@hypothesis.given(segmentations, segmentations)
def test_count_matches_strict_not_max(ref, guess):
    """Check that different sets yield do not yield maximal score when tolerance is 0."""
    hypothesis.assume(set(ref) != set(guess))
    assert scoring.count_matches(ref, guess, 0) < max(len(set(ref)), len(set(guess)))


@hypothesis.given(segmentations, segmentations)
def test_count_matches_no_more_than_minlength(ref, guess):
    """Check that the number of matches is lower than the number of elements in either argument."""
    assert scoring.count_matches(ref, guess, 0) <= min(len(ref), len(guess))


@hypothesis.given(segmentations, segmentations, st.floats(min_value=0))
def test_count_matches_symmetry(ref, guess, tolerance):
    """Check that countMatches is symmetric."""
    assert scoring.count_matches(ref, guess, tolerance) == scoring.count_matches(guess, ref, tolerance)


@hypothesis.given(segmentations, segmentations, st.floats(min_value=0), st.floats(min_value=0))
def test_count_matches_weak_monotonic(ref, guess, tolerance1, tolerance2):
    """Check that increasing tolerance doesn't decrease matches."""
    score1 = scoring.count_matches(ref, guess, tolerance1)
    score2 = scoring.count_matches(ref, guess, tolerance2)
    assert (score1 <= score2) or (tolerance1 > tolerance2)


def test_precision_known_examples():
    """Check precision with a known example."""
    ref = [0, 10, 20]
    guess = [5, 19]
    assert scoring.precision(ref, guess, 3) == 0.5


def test_recall_known_examples():
    """Check recall with a known example."""
    ref = [0, 10, 20]
    guess = [5, 19]
    assert scoring.recall(ref, guess, 3) == 1/3.0


@hypothesis.given(segmentations, st.integers(min_value=0))
def test_recall_empty(guess, tol):
    """Check recall with an empty reference."""
    assert scoring.recall([], guess, tol) == 0


@hypothesis.given(segmentations, st.integers(min_value=0))
def test_precision_empty(ref, tol):
    """Check precision with an empty guess."""
    assert scoring.precision(ref, [], tol) == 0


@hypothesis.given(segmentations, segmentations, st.integers(min_value=0))
def test_precision_range(ref, guess, tol):
    """Check the range of F-measure."""
    result = scoring.precision(ref, guess, tol)
    assert 0 <= result <= 1


@hypothesis.given(segmentations, segmentations, st.integers(min_value=0))
def test_recall_range(ref, guess, tol):
    """Check the range of F-measure."""
    result = scoring.recall(ref, guess, tol)
    assert 0 <= result <= 1


@hypothesis.given(segmentations, segmentations, st.integers(min_value=0))
def test_precision_recall_symmetry(ref, guess, tol):
    """Check that precision and recall are symmetric."""
    assert scoring.precision(ref, guess, tol) == pytest.approx(
        scoring.recall(guess, ref, tol), nan_ok=True)
    assert scoring.recall(ref, guess, tol) == pytest.approx(
        scoring.precision(guess, ref, tol), nan_ok=True)


def test_fmeasure_1():
    """Check F1 with a known example."""
    ref = [0, 10, 20]
    guess = [5, 19]
    assert scoring.f_measure(ref, guess, 3) == 0.4


@hypothesis.given(segmentations, segmentations, st.floats(min_value=0), st.floats(min_value=1e-5, max_value=1e5))
def test_fmeasure_range(ref, guess, tol, weight):
    """Check the range of F-measure."""
    hypothesis.assume(ref != [] or guess != [])
    result = scoring.f_measure(ref, guess, tol, weight)
    assert 0 <= result <= 1


@hypothesis.given(segmentations, st.floats(min_value=0))
def test_perfect_score(ref, tol):
    """Check that an exact guess gets a perfect score."""
    hypothesis.assume(ref != [])
    assert scoring.frp_measures(ref, ref, tol) == (1, 1, 1)
