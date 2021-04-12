"""Tests for the classes in lengthPriors."""
import pytest
import hypothesis
import hypothesis.strategies as st
import lengthPriors

means = st.floats(min_value=0, allow_infinity=False)
stddevs = st.floats(min_value=1, exclude_min=True, allow_infinity=False)
xinputs = st.lists(st.floats(min_value=0), min_size=2, unique=True).map(sorted)
normalPriors = st.builds(lengthPriors.NormalLengthPrior,
                         means, stddevs, xinputs,
                         st.integers(min_value=2, max_value=50))
segmentLengthSets = st.lists(st.integers(
    min_value=2, max_value=100), min_size=2)
pieceLengths = st.integers(min_value=2, max_value=10000)
empiricalPriors = st.builds(lengthPriors.EmpiricalLengthPrior,
                            dataLength=pieceLengths,
                            data=segmentLengthSets)


@hypothesis.given(normalPriors,
                  st.integers(min_value=0),
                  st.integers(min_value=0))
def test_proba_Normal(distribution, i, j):
    """Check that evalCond returns probability-like values."""
    try:
        result = distribution.evalCond(i, j)
        assert 0 <= result <= 1
    except lengthPriors.ImpossibleCondition:
        pass


@hypothesis.given(normalPriors,
                  st.integers(min_value=0))
def test_proba_sum_Normal_linear(distribution, i):
    """Check that evalCond sums to 1 over j (if i is possible)."""
    try:
        result = sum(distribution.evalCond(i, i+j+1)
                     for j in range(len(distribution.x)))
        assert result == pytest.approx(1, abs=1e-10) or result == 0
    except lengthPriors.ImpossibleCondition:
        hypothesis.reject()


@hypothesis.given(empiricalPriors,
                  st.integers(min_value=0))
def test_proba_sum_Empirical_linear(distribution, i):
    """Check that evalCond sums to 1 over j (if i is possible)."""
    try:
        result = sum(distribution.evalCond(i, j)
                     for j in range(distribution.dataLength))
        assert result == pytest.approx(1, abs=1e-10) or result == 0
    except lengthPriors.ImpossibleCondition:
        # TODO: Check that it is indeed an impossible condition
        hypothesis.reject()


@hypothesis.given(segmentLengthSets, pieceLengths)
def test_build_Empirical(lengths, dataLength):
    """Check that EmpiricalPriors get properly constructed from data."""
    assert lengthPriors.EmpiricalLengthPrior(
        dataLength=dataLength, data=lengths)


def test_empirical_distribution_by_value():
    """Check a distribution after construction."""
    dist = lengthPriors.inferDiscreteDistribution([5, 5, 3, 8, 4])
    assert dist[5] == 0.4
    assert dist[4] == 0.2


@hypothesis.given(segmentLengthSets)
def test_empirical_distribution_is_distribution(segmentLengthSet):
    """Check that the inferred distribution is a distribution."""
    dist = lengthPriors.inferDiscreteDistribution(segmentLengthSet)
    assert all(0 <= p <= 1 for p in dist.values())
    assert sum(dist.values()) == pytest.approx(1)
