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
indices = st.integers(min_value=0)


@st.composite
def continuousPriorsWithIndices(draw, priorStrategy=normalPriors):
    """Generate a continuous prior and an index into it."""
    prior = draw(priorStrategy.filter(lambda p: p.x != []))
    index = draw(st.integers(min_value=0, max_value=len(prior.x)-1))
    return (prior, index)


@st.composite
def discretePriorsWithIndices(draw, priorStrategy=empiricalPriors):
    """Generate a continuous prior and an index into it."""
    prior = draw(priorStrategy.filter(lambda p: p.dataLength != 0))
    index = draw(st.integers(min_value=0, max_value=prior.dataLength-1))
    return (prior, index)


@st.composite
def continuousPriorsWithInvalidIndices(draw, priorStrategy=normalPriors):
    """Generate a continuous prior and an index into it."""
    prior = draw(priorStrategy)
    index = draw(st.integers(max_value=-1) |
                 st.integers(min_value=len(prior.x)))
    return (prior, index)


@st.composite
def discretePriorsWithInvalidIndices(draw, priorStrategy=empiricalPriors):
    """Generate a continuous prior and an index into it."""
    prior = draw(priorStrategy)
    index = draw(st.integers(max_value=-1) |
                 st.integers(min_value=prior.dataLength))
    return (prior, index)


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


# evalCond does not give proper priors, but this is fine (it cancels out)
# @hypothesis.given(normalPriors,
#                   st.integers(min_value=0))
# def test_proba_sum_Normal_linear(distribution, i):
#     """Check that evalCond sums to 1 over j (if i is possible)."""
#     try:
#         result = sum(distribution.evalCond(i, j)
#                      for j in range(len(distribution.x)))
#         assert result == pytest.approx(1, abs=1e-10) or result == 0
#     except lengthPriors.ImpossibleCondition:
#         hypothesis.reject()


# @hypothesis.given(empiricalPriors,
#                   st.integers(min_value=0))
# def test_proba_sum_Empirical_linear(distribution, i):
#     """Check that evalCond sums to 1 over j (if i is possible)."""
#     try:
#         result = sum([distribution.evalCond(i, j)
#                      for j in range(distribution.dataLength)])
#         assert result == pytest.approx(1, abs=1e-10) or result == 0
#     except lengthPriors.ImpossibleCondition:
#         # TODO: Check that it is indeed an impossible condition
#         hypothesis.reject()


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


@hypothesis.given(discretePriorsWithIndices())
def test_discrete_min_index_in_data(priorandposition):
    """Check that min index is within the data range for discrete priors."""
    prior, position = priorandposition
    minpos = prior.getMinIndex(position)
    assert 0 <= minpos < prior.dataLength


@hypothesis.given(continuousPriorsWithIndices())
def test_continuous_min_index_in_data(priorandposition):
    """Check that min index is within the data range for continuous priors."""
    prior, position = priorandposition
    minpos = prior.getMinIndex(position)
    assert 0 <= minpos < len(prior.x)


@hypothesis.given(discretePriorsWithInvalidIndices())
def test_discrete_min_reject_invalid_input(priorandposition):
    """Check that min index rejects invalid arguments."""
    prior, position = priorandposition
    with pytest.raises(IndexError):
        prior.getMinIndex(position)


@hypothesis.given(continuousPriorsWithInvalidIndices())
def test_continuous_min_reject_invalid_input(priorandposition):
    """Check that min index rejects invalid arguments."""
    prior, position = priorandposition
    with pytest.raises(IndexError):
        prior.getMinIndex(position)


@hypothesis.given(discretePriorsWithIndices())
def test_discrete_max_index_in_data(priorandposition):
    """Check that max index is within the data range for discrete priors."""
    prior, position = priorandposition
    maxpos = prior.getMaxIndex(position)
    assert 0 <= maxpos < prior.dataLength


@hypothesis.given(continuousPriorsWithIndices())
def test_continuous_max_index_in_data(priorandposition):
    """Check that max index is within the data range for continuous priors."""
    prior, position = priorandposition
    maxpos = prior.getMaxIndex(position)
    assert 0 <= maxpos < len(prior.x)


@hypothesis.given(discretePriorsWithInvalidIndices())
def test_discrete_max_reject_invalid_input(priorandposition):
    """Check that max index rejects invalid arguments."""
    prior, position = priorandposition
    with pytest.raises(IndexError):
        prior.getMaxIndex(position)


@hypothesis.given(continuousPriorsWithInvalidIndices())
def test_continuous_max_reject_invalid_input(priorandposition):
    """Check that max index rejects invalid arguments."""
    prior, position = priorandposition
    with pytest.raises(IndexError):
        prior.getMaxIndex(position)


@hypothesis.given(continuousPriorsWithIndices(), st.data())
def test_min_evolves_inversly_with_maxlength_(priorandposition, increment):
    """Check that changing maxLength can only cause inverse change to min index."""
    prior, position = priorandposition
    initialMin = prior.getMinIndex(position)
    increment = increment.draw(st.floats(min_value=-prior.maxLength, max_value=10000, exclude_min=True))
    hypothesis.assume(increment != 0)
    prior.maxLength += increment
    newMin = prior.getMinIndex(position)
    minDifference = newMin - initialMin
    assert (increment >= 0 and minDifference <= 0) or (increment <= 0 and minDifference >= 0)
