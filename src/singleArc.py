"""Functions for single arcs with multivariate output"""
import numpy as np
import numpy.polynomial.polynomial
from scipy.linalg import block_diag
from scipy.stats import multivariate_normal

from mydecorators import singleOrList


@singleOrList(kw='priors')
def makeMeanVect(priors):
    """ Returns an array of the prior means of the model parameters """
    result = []
    for prior in priors:
        result.extend([prior['aMean'], prior['bMean'], prior['cMean']])
    return np.array(result)


@singleOrList(kw='priors')
def makeVarVect(priors):
    """ Returns an array of the prior variance of the model parameters """
    result = []
    for prior in priors:
        result.extend([prior['aStd'], prior['bStd'], prior['cStd']])
    return np.square(np.array(result))


def makeDesignMatrix(xarray, outputDims=1):
    """ Builds the design matrix for the problem at hand """
    return block_diag(*[np.array([[x**2, x, 1] for x in xarray]) for i in range(outputDims)])


@singleOrList(kw='priors')
def makeNoiseCov(priors, inputVector):
    """ Builds the gaussian noise's covariance matrix """
    return block_diag(*[(prior['noiseStd']**2)*np.identity(len(inputVector)) for prior in priors])


@singleOrList(kw='priors')
def arcLikelihood(priors, data):
    """ Takes a prior and a set of input/output values and return the log-likelihood of the data """
    # 1 input, variable number of outputs
    (inputVector, outputVectors) = zip(*data)
    outputDim = len(priors)
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


def arcMAP(prior, data):
    """ Takes a prior and a set of input/output values and return the most likely arc with its loglikelihood
    """
    # NYI
    return {"LL": None, "Arc": None}


def arcML(data, returnEstimates=False):
    """ Returns the maximum likelihood arc for a set of input/output values
    """
    (inputVector, outputVector) = zip(*data)
    polyfit = np.polynomial.polynomial.polyfit(inputVector, outputVector, 2)
    if returnEstimates:
        return list(reversed(polyfit)), np.polynomial.polynomial.polyval(inputVector, polyfit)
    else:
        return list(reversed(polyfit))


def normalizeX(dataSlice, linearSampling=True):
    # TODO: Automatically detect if linear sampling (beat-wise) or not (note-wise)
    if linearSampling:
        if len(dataSlice) == 1:
            return [(0, dataSlice[0])]
        else:
            return [(float(i)/(len(dataSlice)-1), dataPoint) for (i, dataPoint) in enumerate(dataSlice)]
    else:
        maxX, _ = data[-1]
        return [(float(x)/maxX, y) for x, y in data]


def knownSegmentationML(data, segmentation):
    y = []  # ML estimation of the denoised data
    models = []  # ML coefficient estimates
    lengths = []
    for (bound_curr, bound_next) in zip(segmentation, segmentation[1:]):
        dataPairs = normalizeX(data[(sscurr+1):ssnext+1])
        model, values = arcML(dataPairs, returnEstimates=True)
        models.append(model)
        y.extend(values)
        lengths.append(ssnext-sscurr)
    return y, models, lengths
