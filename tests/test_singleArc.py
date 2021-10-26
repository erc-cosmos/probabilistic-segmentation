"""Tests for the singleArc submodule."""
import singleArc as sa
import defaultVars
import numpy as np
import hypothesis
import hypothesis.strategies as st
import pytest


def test_likelihood_1D():
    prior = defaultVars.arcPrior
    data = defaultVars.data1D
    assert sa.arcLikelihood(prior, data) == pytest.approx(-37.52804497877863)


def test_likelihood_2D():
    priors = [defaultVars.arcPrior, defaultVars.arcPrior2]
    data = defaultVars.dataMultidim
    assert sa.arcLikelihood(priors=priors, data=data) == pytest.approx(-161.83384451749782)


def test_meanVect_1D():
    prior = defaultVars.arcPrior
    assert list(sa.makeMeanVect(prior)) == [10, 20, 30]


def test_meanVect_2D():
    priors = [defaultVars.arcPrior, defaultVars.arcPrior2]
    assert list(sa.makeMeanVect(priors)) == [10, 20, 30, -10, -20, -30]


def test_varVect_1D():
    prior = defaultVars.arcPrior
    assert list(sa.makeVarVect(prior)) == [1, 4, 9]


def test_varVect_2D():
    priors = [defaultVars.arcPrior, defaultVars.arcPrior2]
    assert list(sa.makeVarVect(priors)) == [1, 4, 9, 1, 4, 9]


def test_design_format_1D():
    x, _ = zip(*defaultVars.data1D)
    assert sa.makeDesignMatrix(x).shape == (5, 3)


def test_design_format_2D():
    x, _ = zip(*defaultVars.dataMultidim)
    output = sa.makeDesignMatrix(x, outputDims=2)
    assert output.shape == (10, 6)  # General dimensions
    assert not output[:5, 3:].any()  # Is block diagonal
    assert not output[5:, :3].any()


def test_noiseCov_1D_is_diagonal():
    prior = defaultVars.arcPrior
    x, _ = zip(*defaultVars.data1D)
    output = sa.makeNoiseCov(prior, x)
    assert not np.any(output-np.diag(np.diagonal(output)))


def test_noiseCov_2D_is_diagonal():
    priors = [defaultVars.arcPrior, defaultVars.arcPrior2]
    x, _ = zip(*defaultVars.dataMultidim)
    output = sa.makeNoiseCov(priors, x)
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
    loglik_opt = sa.arcLikelihood(prior, sa.normalizeX(data))
    try:
        loglik_no_opt = sa.arcLikelihood(prior, sa.normalizeX(data), disable_opti=True)
    except np.linalg.LinAlgError:
        hypothesis.reject()  # Reject if the variances are too extreme and result in a singular covariance matrix
    assert loglik_opt == pytest.approx(loglik_no_opt)
