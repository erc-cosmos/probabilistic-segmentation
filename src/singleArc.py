"""Functions for single arcs with multivariate output."""
import numpy as np
import numpy.polynomial.polynomial
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from mydecorators import singleOrList


@singleOrList(kw='priors')
def makeMeanVect(priors):
    """Return an array of the prior means of the model parameters."""
    result = []
    for prior in priors:
        result.extend([prior['aMean'], prior['bMean'], prior['cMean']])
    return np.array(result)


@singleOrList(kw='priors')
def makeVarVect(priors):
    """Return an array of the prior variance of the model parameters."""
    result = []
    for prior in priors:
        result.extend([prior['aStd'], prior['bStd'], prior['cStd']])
    return np.square(np.array(result))


def makeDesignMatrix(xarray, outputDims=1):
    """Build the design matrix for the problem at hand."""
    return block_diag(*[np.array([[x**2, x, 1] for x in xarray]) for i in range(outputDims)])


@singleOrList(kw='priors')
def makeNoiseCov(priors, inputVector):
    """Build the gaussian noise's covariance matrix."""
    return block_diag(*[(prior['noiseStd']**2)*np.identity(len(inputVector)) for prior in priors])


def is_static_prior(prior):
    """Tell if a prior is static in time."""
    return all(prior[param] == 0 for param in ['aMean', 'bMean', 'aStd', 'bStd'])


@singleOrList(kw='priors')
def arcLikelihood(priors, data, *, disable_opti=False):
    """Take a prior and a set of input/output values and return the log-likelihood of the data."""
    # 1 input, variable number of outputs
    (inputVector, outputVectors) = zip(*data)
    outputDim = len(priors)

    if len(priors) == 1 and is_static_prior(priors[0]) and not disable_opti:
        # Optimized version for static priors
        return _arc_likelihood_static_prior(priors[0], np.array(outputVectors))

    # Capital Phi in the doc
    designMatrix = makeDesignMatrix(inputVector, outputDim)
    # Bold mu in the doc
    meanVectPrior = makeMeanVect(priors)
    # Means vector for the data
    meanVectData = np.matmul(designMatrix, meanVectPrior)
    # Bold sigma^2 in the doc
    varVect = makeVarVect(priors)
    # Noise component of covariance matrix
    noiseCov = makeNoiseCov(priors, inputVector)
    # Covariance matrix for the data
    covMatData = noiseCov + \
        (designMatrix @ np.diag(varVect) @ np.transpose(designMatrix))
    # Bold t in the doc
    targetValues = np.array(outputVectors).flatten('F')  # Flatten column first

    return multivariate_normal.logpdf(targetValues, mean=meanVectData, cov=covMatData)


def _arc_likelihood_static_prior(prior, data):
    """Take a static prior and a set of input/output values and return the log-likelihood of the data."""
    d = len(data)
    # Centered data w.r.t the mean mean
    cdata = data - prior['cMean']
    # The covariance matrix is xId + y
    x = prior['noiseStd'] ** 2
    y = prior['cStd'] ** 2
    # Its inverse is vId - w
    v = 1/x
    w = y/x/(x+d*y)
    # Its determinant has a closed formula
    det = (x**(d-1)*(x+d*y))
    # This is the exponent in the multivariate Gaussian density formula
    exponent = v * (cdata @ cdata) - w * sum(cdata) ** 2

    loglik = -(exponent + d * np.log(2 * np.pi) + np.log(det))/2
    return loglik


def arcMAP(prior, data):
    """Take a prior and a set of input/output values and return the most likely arc with its loglikelihood."""
    # NYI
    return {"LL": None, "Arc": None}


def arcML(data, returnEstimates=False):
    """Return the maximum likelihood arc for a set of input/output values."""
    (inputVector, outputVector) = zip(*data)
    polyfit = np.polynomial.polynomial.polyfit(inputVector, outputVector, 2)
    if returnEstimates:
        return list(reversed(polyfit)), np.polynomial.polynomial.polyval(inputVector, polyfit)
    else:
        return list(reversed(polyfit))


def normalizeX(data_slice, linearSampling=True):
    """Normalize input variable (or generate it if needed) to range from 0 to 1."""
    # TODO: Automatically detect if linear sampling (beat-wise) or not (note-wise)
    if linearSampling:
        if len(data_slice) == 1:
            return [(0, data_slice[0])]
        else:
            return [(float(i)/(len(data_slice)-1), dataPoint) for (i, dataPoint) in enumerate(data_slice)]
    else:
        maxX, _ = data_slice[-1]
        return [(float(x)/maxX, y) for x, y in data_slice]


def knownSegmentationML(data, segmentation):
    """Perform a ML estimation with known boundaries."""
    y = []  # ML estimation of the denoised data
    models = []  # ML coefficient estimates
    lengths = []
    for (bound_curr, bound_next) in zip(segmentation, segmentation[1:]):
        dataPairs = normalizeX(data[(bound_curr+1):bound_next+1])
        model, values = arcML(dataPairs, returnEstimates=True)
        models.append(model)
        y.extend(values)
        lengths.append(bound_next-bound_curr)
    return y, models, lengths
