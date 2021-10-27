"""Tests for the singleArc submodule."""
from bayes_arcs import single_arc as sa
import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest

import default_vars


def test_likelihood_1d():
    """Test likelihood of known 1D data."""
    prior = default_vars.arc_prior
    data = default_vars.data_1d
    assert sa.arc_likelihood(prior, data) == pytest.approx(-37.52804497877863)


def test_likelihood_2d():
    """Test likelihood of known 2D data."""
    priors = [default_vars.arc_prior, default_vars.arc_prior2]
    data = default_vars.data_multidim
    assert sa.arc_likelihood(priors=priors, data=data) == pytest.approx(-161.83384451749782)


def test_mean_vect_1d():
    """Test that mean priors are properly vectorized in 1D."""
    prior = default_vars.arc_prior
    assert list(sa.make_mean_vect(prior)) == [10, 20, 30]


def test_mean_vect_2d():
    """Test that mean priors are properly vectorized in 2D."""
    priors = [default_vars.arc_prior, default_vars.arc_prior2]
    assert list(sa.make_mean_vect(priors)) == [10, 20, 30, -10, -20, -30]


def test_var_vect_1d():
    """Test that variance priors are properly vectorized in 1D."""
    prior = default_vars.arc_prior
    assert list(sa.make_var_vect(prior)) == [1, 4, 9]


def test_var_vect_2d():
    """Test that variance priors are properly vectorized in 2D."""
    priors = [default_vars.arc_prior, default_vars.arc_prior2]
    assert list(sa.make_var_vect(priors)) == [1, 4, 9, 1, 4, 9]


def test_design_format_1d():
    """Check the shape of the design matrix in 1D."""
    x, _ = zip(*default_vars.data_1d)
    assert sa.make_design_matrix(x).shape == (5, 3)


def test_design_format_2d():
    """Check the shape of the design matrix in 2D."""
    x, _ = zip(*default_vars.data_multidim)
    output = sa.make_design_matrix(x, output_dims=2)
    assert output.shape == (10, 6)  # General dimensions
    assert not output[:5, 3:].any()  # Is block diagonal
    assert not output[5:, :3].any()


def test_noise_cov_1d_is_diagonal():
    """Check that the covariance matrix for 1D is diagonal."""
    prior = default_vars.arc_prior
    x, _ = zip(*default_vars.data_1d)
    output = sa.make_noise_cov(prior, x)
    assert not np.any(output-np.diag(np.diagonal(output)))


def test_noise_cov_2d_is_diagonal():
    """Check that the covariance matrix for 2D is diagonal."""
    priors = [default_vars.arc_prior, default_vars.arc_prior2]
    x, _ = zip(*default_vars.data_multidim)
    output = sa.make_noise_cov(priors, x)
    assert not np.any(output-np.diag(np.diagonal(output)))


@hypothesis.given(st.lists(st.floats(min_value=-1e5, max_value=1e5), min_size=2),
                  st.floats(min_value=-1e5, max_value=1e5),
                  st.floats(min_value=1e-3, max_value=1e3),
                  st.floats(min_value=1e-3, max_value=1e3))
def test_static_optimisation_is_equivalent(data, mean, std, noise):
    """Check that static signal optimisation does not change the result."""
    prior = {
        # Gaussian priors on the parameters of ax^2 + bx + c
        'aMean': 0,
        'aStd': 0,
        'bMean': 0,
        'bStd': 0,
        'cMean': mean,
        'cStd': std,
        'noiseStd': noise
    }
    loglik_opt = sa.arc_likelihood(prior, sa.normalize_x(data))
    try:
        loglik_no_opt = sa.arc_likelihood(prior, sa.normalize_x(data), disable_opti=True)
    except np.linalg.LinAlgError:
        hypothesis.reject()  # Reject if the variances are too extreme and result in a singular covariance matrix
    else:
        assert loglik_opt == pytest.approx(loglik_no_opt)
