from context import *
import scoring


def test_countMatches_noMatch():
    ref = [0, 10, 20]
    guess = [5, 16]
    assert scoring.countMatches(ref, guess, 3) == 0
    assert scoring.countMatches(ref, guess, 4) == 1
    assert scoring.countMatches(ref, guess, 6) == 2


def test_precision():
    ref = [0, 10, 20]
    guess = [5, 19]
    assert scoring.precision(ref, guess, 3) == 0.5


def test_recall():
    ref = [0, 10, 20]
    guess = [5, 19]
    assert scoring.recall(ref, guess, 3) == 1/3.0


@hypothesis.given(st.lists(st.floats(min_value=0),max_size=0), st.lists(st.floats(min_value=0)), st.integers(min_value=0))
def test_recall_empty(ref, guess, tol):
    assert scoring.recall(ref, guess, tol) == 0


@hypothesis.given(st.lists(st.floats(min_value=0)), st.lists(st.floats(min_value=0),max_size=0), st.integers(min_value=0))
def test_precision_empty(ref, guess, tol):
    assert scoring.precision(ref, guess, tol) == 0


@hypothesis.given(st.lists(st.floats(min_value=0)), st.lists(st.floats(min_value=0)), st.integers(min_value=0))
def test_precision_range(ref, guess, tol):
    result = scoring.precision(ref, guess, tol)
    assert 0 <= result <= 1


@hypothesis.given(st.lists(st.floats(min_value=0)), st.lists(st.floats(min_value=0)), st.integers(min_value=0))
def test_recall_range(ref, guess, tol):
    result = scoring.recall(ref, guess, tol)
    assert 0 <= result <= 1


@hypothesis.given(st.lists(st.floats(min_value=0)), st.lists(st.floats(min_value=0)), st.integers(min_value=0))
def test_precision_recall_symmetry(ref, guess, tol):
    assert scoring.precision(ref, guess, tol) == pytest.approx(
        scoring.recall(guess, ref, tol), nan_ok=True)
    assert scoring.recall(ref, guess, tol) == pytest.approx(
        scoring.precision(guess, ref, tol), nan_ok=True)


def test_fmeasure_1():
    ref = [0, 10, 20]
    guess = [5, 19]
    assert scoring.fMeasure(ref, guess, 3) == 0.4

@hypothesis.given(st.lists(st.floats(min_value=0)), st.lists(st.floats(min_value=0)), st.integers(min_value=0))
def test_fmeasure_range(ref, guess, tol):
    result = scoring.fMeasure(ref, guess, tol)
    assert 0 <= result <= 1