"""Tests for the dynamicComputation module."""
import functools

import dynamicComputation as dc
import hypothesis
import hypothesis.strategies as st
import lengthPriors
import numpy as np
import syntheticData as sd
import pytest

from defaultVars import arcPrior

means = st.floats(min_value=0, allow_infinity=False)
stddevs = st.floats(min_value=1, exclude_min=True, allow_infinity=False)
xinputs = st.lists(st.floats(min_value=0), min_size=2, unique=True).map(sorted)


def normalPriors(dataLength):
    """Generate normal priors bound to given data length."""
    return st.builds(functools.partial(lengthPriors.NormalLengthPrior, xinputs=list(range(dataLength))),
                     means, stddevs, st.integers(min_value=2, max_value=50))


segmentLengthSets = st.lists(st.integers(min_value=2, max_value=15), min_size=2, max_size=5)
features = st.builds(functools.partial(sd.genData, arcPrior=arcPrior), segmentLengthSets)


# alphabeta takes a little longer than most tests
@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_marginals_are_probability(data):
    """Check that marginals are probabilities."""
    hidden, data = data
    lengthPrior = lengthPriors.NormalLengthPrior(15, 5, list(range(len(data))), maxLength=20)
    posteriorMarginals = dc.runAlphaBeta(data, arcPrior, lengthPrior)
    assert 0 <= np.nanmin(posteriorMarginals)
    assert 1.0001 >= np.nanmax(posteriorMarginals)


# alphabeta takes a little longer than most tests
@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_beta_are_logprobability(data):
    """Check that betas are log-probabilities."""
    hidden, data = data
    lengthPrior = lengthPriors.NormalLengthPrior(15, 5, list(range(len(data))), maxLength=20)
    DLs = dc.computeDataLikelihood(data, arcPrior, lengthPrior)
    betas = dc.computeBetas(arcPrior, lengthPrior, DLs)
    assert 0 >= np.nanmax(betas)


# alphabeta takes a little longer than most tests
@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_alpha_are_logprobability(data):
    """Check that alphas are log-probabilities."""
    hidden, data = data
    lengthPrior = lengthPriors.NormalLengthPrior(15, 5, list(range(len(data))), maxLength=20)
    DLs = dc.computeDataLikelihood(data, arcPrior, lengthPrior)
    alphas = dc.computeAlphas(arcPrior, lengthPrior, DLs)
    assert all([0 >= alpha for alpha in alphas])


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_dataLikelihood_is_upper_triangular(data):
    """Check that the dataLikelihood matrix is (strictly) upper triangular."""
    hidden, data = data
    lengthPrior = lengthPriors.NormalLengthPrior(15, 5, list(range(len(data))), maxLength=20)
    DLmatrix = dc.computeDataLikelihood(data, arcPrior, lengthPrior)

    np.testing.assert_equal(DLmatrix[np.tril_indices(len(data), k=-1)], np.NINF)
    assert np.nanmax(DLmatrix) <= 0  # Also assert logproba


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_alphaN_is_beta0(data):
    """Check that the last value of alpha is the same as the first value of beta."""
    hidden, data = data
    lengthPrior = lengthPriors.NormalLengthPrior(15, 5, list(range(len(data))), maxLength=20)
    DLs = dc.computeDataLikelihood(data, arcPrior, lengthPrior)
    alphas = dc.computeAlphas(arcPrior, lengthPrior, DLs)
    betas = dc.computeBetas(arcPrior, lengthPrior, DLs)
    assert alphas[-1] == pytest.approx(betas[0])
    assert alphas[-1] <= 0


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_alpha_length(data):
    """Check that the length of alphas is the expected length."""
    hidden, data = data
    lengthPrior = lengthPriors.NormalLengthPrior(15, 5, list(range(len(data))), maxLength=20)
    DLs = dc.computeDataLikelihood(data, arcPrior, lengthPrior)
    alphas = dc.computeAlphas(arcPrior, lengthPrior, DLs)
    assert len(alphas) == len(data) + 1


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_beta_length(data):
    """Check that the length of betas is the expected length."""
    hidden, data = data
    lengthPrior = lengthPriors.NormalLengthPrior(15, 5, list(range(len(data))), maxLength=20)
    DLs = dc.computeDataLikelihood(data, arcPrior, lengthPrior)
    betas = dc.computeBetas(arcPrior, lengthPrior, DLs)
    assert len(betas) == len(data) + 1


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_marginals_sum_over_minimum_segment_count(data):
    """Check that the marginals sum over the minimum number of boundaries."""
    hidden, data = data
    lengthPrior = lengthPriors.EmpiricalLengthPrior(range(11), dataLength=len(data), maxLength=10)
    marginals = dc.runAlphaBeta(data, arcPrior, lengthPrior)
    assert sum(marginals) >= np.ceil(lengthPrior.dataLength/float(lengthPrior.maxLength+1))


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_marginals_sum_over_minimum_segment_count_with_NormalPrior(data):
    """Check that the marginals sum over the minimum number of boundaries."""
    hidden, data = data
    lengthPrior = lengthPriors.NormalLengthPrior(15, 5, list(range(len(data))), maxLength=20)
    marginals = dc.runAlphaBeta(data, arcPrior, lengthPrior)
    assert sum(marginals) >= np.ceil(lengthPrior.dataLength/float(lengthPrior.maxLength+1))
