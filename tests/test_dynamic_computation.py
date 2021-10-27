"""Tests for the dynamicComputation module."""
import functools

from bayes_arcs import dynamic_computation as dc
from bayes_arcs import length_priors
from bayes_arcs import synthetic_data as sd
import hypothesis
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest

import default_vars

means = st.floats(min_value=0, allow_infinity=False)
stddevs = st.floats(min_value=1, exclude_min=True, allow_infinity=False)
xinputs = st.lists(st.floats(min_value=0), min_size=2, unique=True).map(sorted)


def normal_priors(data_length):
    """Generate normal priors bound to given data length."""
    return st.builds(functools.partial(length_priors.NormalLengthPrior, xinputs=list(range(data_length))),
                     means, stddevs, st.integers(min_value=2, max_value=50))


segment_length_sets = st.lists(st.integers(min_value=2, max_value=15), min_size=2, max_size=5)
features = st.builds(functools.partial(sd.gen_data, arc_prior=default_vars.arc_prior), segment_length_sets)


# alphabeta takes a little longer than most tests
@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_marginals_are_probability(data):
    """Check that marginals are probabilities."""
    hidden, data = data
    length_prior = length_priors.NormalLengthPrior(15, 5, list(range(len(data))), max_length=20)
    posterior_marginals = dc.run_alpha_beta(data, default_vars.arc_prior, length_prior)
    assert 0 <= np.nanmin(posterior_marginals)
    assert 1.0001 >= np.nanmax(posterior_marginals)


# alphabeta takes a little longer than most tests
@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_beta_are_logprobability(data):
    """Check that betas are log-probabilities."""
    hidden, data = data
    length_prior = length_priors.NormalLengthPrior(15, 5, list(range(len(data))), max_length=20)
    data_likelihoods = dc.compute_data_likelihood(data, default_vars.arc_prior, length_prior)
    betas = dc.compute_betas(length_prior, data_likelihoods)
    assert 0 >= np.nanmax(betas)


# alphabeta takes a little longer than most tests
@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_alpha_are_logprobability(data):
    """Check that alphas are log-probabilities."""
    hidden, data = data
    length_prior = length_priors.NormalLengthPrior(15, 5, list(range(len(data))), max_length=20)
    data_likelihoods = dc.compute_data_likelihood(data, default_vars.arc_prior, length_prior)
    alphas = dc.compute_alphas(length_prior, data_likelihoods)
    assert all([0 >= alpha for alpha in alphas])


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_data_likelihood_is_upper_triangular(data):
    """Check that the dataLikelihood matrix is (strictly) upper triangular."""
    hidden, data = data
    length_prior = length_priors.NormalLengthPrior(15, 5, list(range(len(data))), max_length=20)
    data_likelihoods = dc.compute_data_likelihood(data, default_vars.arc_prior, length_prior)

    np.testing.assert_equal(data_likelihoods[np.tril_indices(len(data), k=-1)], np.NINF)
    assert np.nanmax(data_likelihoods) <= 0  # Also assert logproba


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_alpha_n_is_beta0(data):
    """Check that the last value of alpha is the same as the first value of beta."""
    hidden, data = data
    length_prior = length_priors.NormalLengthPrior(15, 5, list(range(len(data))), max_length=20)
    data_likelihoods = dc.compute_data_likelihood(data, default_vars.arc_prior, length_prior)
    alphas = dc.compute_alphas(length_prior, data_likelihoods)
    betas = dc.compute_betas(length_prior, data_likelihoods)
    assert alphas[-1] == pytest.approx(betas[0])
    assert alphas[-1] <= 0


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_alpha_length(data):
    """Check that the length of alphas is the expected length."""
    hidden, data = data
    length_prior = length_priors.NormalLengthPrior(15, 5, list(range(len(data))), max_length=20)
    data_likelihoods = dc.compute_data_likelihood(data, default_vars.arc_prior, length_prior)
    alphas = dc.compute_alphas(length_prior, data_likelihoods)
    assert len(alphas) == len(data) + 1


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_beta_length(data):
    """Check that the length of betas is the expected length."""
    hidden, data = data
    length_prior = length_priors.NormalLengthPrior(15, 5, list(range(len(data))), max_length=20)
    data_likelihoods = dc.compute_data_likelihood(data, default_vars.arc_prior, length_prior)
    betas = dc.compute_betas(length_prior, data_likelihoods)
    assert len(betas) == len(data) + 1


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_marginals_sum_over_minimum_segment_count(data):
    """Check that the marginals sum over the minimum number of boundaries."""
    hidden, data = data
    length_prior = length_priors.EmpiricalLengthPrior(range(11), data_length=len(data), max_length=10)
    marginals = dc.run_alpha_beta(data, default_vars.arc_prior, length_prior)
    assert sum(marginals) >= np.ceil(length_prior.data_length/float(length_prior.max_length+1))


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_marginals_sum_over_minimum_segment_count_with_normal_prior(data):
    """Check that the marginals sum over the minimum number of boundaries."""
    hidden, data = data
    length_prior = length_priors.NormalLengthPrior(15, 5, list(range(len(data))), max_length=20)
    marginals = dc.run_alpha_beta(data, default_vars.arc_prior, length_prior)
    assert sum(marginals) >= np.ceil(length_prior.data_length/float(length_prior.max_length+1))


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_marginals_2d_sum_to_marginals(data):
    """Check that the marginals sum over the minimum number of boundaries."""
    hidden, data = data
    length_prior = length_priors.NormalLengthPrior(15, 5, list(range(len(data))), max_length=20)
    marginals, bidim_marginals = dc.run_alpha_beta(data, default_vars.arc_prior, length_prior, return_2d=True)

    npt.assert_allclose(sum(bidim_marginals), marginals)


@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_reverse_commutes(data):
    """Check that reversing then marginalising is the same as marginalising then reversing."""
    hidden, data = data
    length_prior = length_priors.NormalLengthPrior(15, 5, list(range(len(data))), max_length=20)
    arc_prior = {
        # Gaussian priors on the parameters of ax^2 + bx + c
        'aMean': 0,
        'aStd': 0,
        'bMean': 0,
        'bStd': 0,
        'cMean': 0,
        'cStd': 5,
        'noiseStd': 5
    }
    marginals = dc.run_alpha_beta(data, arc_prior, length_prior)

    r_data = list(reversed(data))
    r_marginals = dc.run_alpha_beta(r_data, arc_prior, length_prior)

    npt.assert_allclose(list(reversed(r_marginals[:-1])), marginals[:-1])
