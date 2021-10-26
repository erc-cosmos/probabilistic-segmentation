"""Tests for the classes in lengthPriors."""
import hypothesis
import hypothesis.strategies as st
import length_priors
import pytest

means = st.floats(min_value=0, allow_infinity=False)
stddevs = st.floats(min_value=1, exclude_min=True, allow_infinity=False)
xinputs = st.lists(st.floats(min_value=0), min_size=2, unique=True).map(sorted)
normal_priors = st.builds(length_priors.NormalLengthPrior,
                          means, stddevs, xinputs,
                          st.integers(min_value=2, max_value=50))
segment_length_sets = st.lists(st.integers(
    min_value=2, max_value=100), min_size=2)
piece_lengths = st.integers(min_value=2, max_value=10000)
empirical_priors = st.builds(length_priors.EmpiricalLengthPrior,
                             data_length=piece_lengths,
                             data=segment_length_sets)
indices = st.integers(min_value=0)


@st.composite
def continuous_priors_with_indices(draw, prior_strategy=normal_priors):
    """Generate a continuous prior and an index into it."""
    prior = draw(prior_strategy.filter(lambda p: p.x != []))
    index = draw(st.integers(min_value=0, max_value=len(prior.x)-1))
    return (prior, index)


@st.composite
def discrete_priors_with_indices(draw, prior_strategy=empirical_priors):
    """Generate a continuous prior and an index into it."""
    prior = draw(prior_strategy.filter(lambda p: p.data_length != 0))
    index = draw(st.integers(min_value=0, max_value=prior.data_length-1))
    return (prior, index)


@st.composite
def continuous_priors_with_invalid_indices(draw, prior_strategy=normal_priors):
    """Generate a continuous prior and an index into it."""
    prior = draw(prior_strategy)
    index = draw(st.integers(max_value=-1) |
                 st.integers(min_value=len(prior.x)))
    return (prior, index)


@st.composite
def discrete_priors_with_invalid_indices(draw, prior_strategy=empirical_priors):
    """Generate a continuous prior and an index into it."""
    prior = draw(prior_strategy)
    index = draw(st.integers(max_value=-1) |
                 st.integers(min_value=prior.data_length))
    return (prior, index)


@hypothesis.given(normal_priors,
                  st.integers(min_value=0),
                  st.integers(min_value=0))
def test_proba_normal(distribution, i, j):
    """Check that eval_cond returns probability-like values."""
    try:
        result = distribution.eval_cond(i, j)
        assert 0 <= result <= 1
    except length_priors.ImpossibleCondition:
        pass


@hypothesis.given(segment_length_sets, piece_lengths)
def test_build_empirical(lengths, data_length):
    """Check that EmpiricalPriors get properly constructed from data."""
    assert length_priors.EmpiricalLengthPrior(
        data_length=data_length, data=lengths)


def test_empirical_distribution_by_value():
    """Check a distribution after construction."""
    dist = length_priors.infer_discrete_distribution([5, 5, 3, 8, 4])
    assert dist[5] == 0.4
    assert dist[4] == 0.2


@hypothesis.given(segment_length_sets)
def test_empirical_distribution_is_distribution(segment_length_set):
    """Check that the inferred distribution is a distribution."""
    dist = length_priors.infer_discrete_distribution(segment_length_set)
    assert all(0 <= p <= 1 for p in dist.values())
    assert sum(dist.values()) == pytest.approx(1)


@hypothesis.given(discrete_priors_with_indices())
def test_discrete_min_index_in_data(priorandposition):
    """Check that min index is within the data range for discrete priors."""
    prior, position = priorandposition
    minpos = prior.get_min_index(position)
    assert 0 <= minpos < prior.data_length


@hypothesis.given(continuous_priors_with_indices())
def test_continuous_min_index_in_data(priorandposition):
    """Check that min index is within the data range for continuous priors."""
    prior, position = priorandposition
    minpos = prior.get_min_index(position)
    assert 0 <= minpos < len(prior.x)


@hypothesis.given(discrete_priors_with_invalid_indices())
def test_discrete_min_reject_invalid_input(priorandposition):
    """Check that min index rejects invalid arguments."""
    prior, position = priorandposition
    with pytest.raises(IndexError):
        prior.get_min_index(position)


@hypothesis.given(continuous_priors_with_invalid_indices())
def test_continuous_min_reject_invalid_input(priorandposition):
    """Check that min index rejects invalid arguments."""
    prior, position = priorandposition
    with pytest.raises(IndexError):
        prior.get_min_index(position)


@hypothesis.given(discrete_priors_with_indices())
def test_discrete_max_index_in_data(priorandposition):
    """Check that max index is within the data range for discrete priors."""
    prior, position = priorandposition
    maxpos = prior.get_max_index(position)
    assert 0 <= maxpos < prior.data_length


@hypothesis.given(continuous_priors_with_indices())
def test_continuous_max_index_in_data(priorandposition):
    """Check that max index is within the data range for continuous priors."""
    prior, position = priorandposition
    maxpos = prior.get_max_index(position)
    assert 0 <= maxpos < len(prior.x)


@hypothesis.given(discrete_priors_with_invalid_indices())
def test_discrete_max_reject_invalid_input(priorandposition):
    """Check that max index rejects invalid arguments."""
    prior, position = priorandposition
    with pytest.raises(IndexError):
        prior.get_max_index(position)


@hypothesis.given(continuous_priors_with_invalid_indices())
def test_continuous_max_reject_invalid_input(priorandposition):
    """Check that max index rejects invalid arguments."""
    prior, position = priorandposition
    with pytest.raises(IndexError):
        prior.get_max_index(position)


@hypothesis.given(continuous_priors_with_indices(), st.data())
def test_min_evolves_inversly_with_maxlength_(priorandposition, increment):
    """Check that changing maxLength can only cause inverse change to min index."""
    prior, position = priorandposition
    initial_min = prior.get_min_index(position)
    increment = increment.draw(st.floats(min_value=-prior.max_length, max_value=10000, exclude_min=True))
    hypothesis.assume(increment != 0)
    prior.max_length += increment
    new_min = prior.get_min_index(position)
    min_difference = new_min - initial_min
    assert (increment >= 0 and min_difference <= 0) or (increment <= 0 and min_difference >= 0)
