import context
import singleArc as sa
import defaultVars
import numpy as np

def test_likelihood_1D():
    prior = defaultVars.arcPrior
    data = defaultVars.data1D
    assert sa.arcLikelihood(prior,data) == -37.52804497877863

def test_likelihood_2D():
    priors = [defaultVars.arcPrior, defaultVars.arcPrior2]
    data = defaultVars.dataMultidim
    assert sa.arcLikelihood(priors=priors,data=data) == -161.83384451749782
    
def test_meanVect_1D():
    prior = defaultVars.arcPrior
    assert list(sa.makeMeanVect(prior)) == [10,20,30]

def test_meanVect_2D():
    priors = [defaultVars.arcPrior, defaultVars.arcPrior2]
    assert list(sa.makeMeanVect(priors)) == [10,20,30,-10,-20,-30]

def test_varVect_1D():
    prior = defaultVars.arcPrior
    assert list(sa.makeVarVect(prior)) == [1,4,9]

def test_varVect_2D():
    priors = [defaultVars.arcPrior, defaultVars.arcPrior2]
    assert list(sa.makeVarVect(priors)) == [1,4,9,1,4,9]

def test_design_format_1D():
    x,_ = zip(*defaultVars.data1D)
    assert sa.makeDesignMatrix(x).shape == (5,3)

def test_design_format_2D():
    x,_ = zip(*defaultVars.dataMultidim)
    output = sa.makeDesignMatrix(x, outputDims=2)
    assert output.shape == (10,6) # General dimensions
    assert not output[:5,3:].any() # Is block diagonal
    assert not output[5:,:3].any()

def test_noiseCov_1D_is_diagonal():
    prior = defaultVars.arcPrior
    x,_ = zip(*defaultVars.data1D)
    output = sa.makeNoiseCov(prior, x)
    assert not np.any(output-np.diag(np.diagonal(output)))

def test_noiseCov_2D_is_diagonal():
    priors = [defaultVars.arcPrior, defaultVars.arcPrior2]
    x,_ = zip(*defaultVars.dataMultidim)
    output = sa.makeNoiseCov(priors, x)
    assert not np.any(output-np.diag(np.diagonal(output)))
