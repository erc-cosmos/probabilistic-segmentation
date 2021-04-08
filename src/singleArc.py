# Functions for single arcs
from scipy.stats import multivariate_normal
import numpy as np
import numpy.polynomial.polynomial

def makeMeanVect(prior):
    return np.array([prior['aMean'], prior['bMean'], prior['cMean']])

def makeVarVect(prior):
    """ Returns an array of the prior variance of the model parameters
    """
    return np.square(np.array([prior['aStd'], prior['bStd'], prior['cStd']]))

def makeDesignMatrix(xarray):
    return np.array([[x**2,x, 1] for x in xarray])

def arcLikelihood(prior, data):
    """ Takes a prior and a set of input/output values and return the likelihood of the data
    """
    (inputVector, outputVector) = zip(*data)
    # Capital Phi in the doc
    designMatrix = makeDesignMatrix(inputVector)
    #print(designMatrix)
    # bold mu in the doc
    meanVectPrior = makeMeanVect(prior)
    #print(meanVectPrior)
    # Means vector for the data
    meanVectData = np.matmul(designMatrix,meanVectPrior)
    #print(meanVectData)
    # bold sigma^2 in the doc
    varVect = makeVarVect(prior)
    #print(varVect)
    # Covariance matrix for the data
    covMatData = (prior['noiseStd']**2)*np.identity(len(inputVector)) + (designMatrix @ np.diag(varVect) @ np.transpose(designMatrix))
    # print(covMatData)
    # bold t in the doc
    targetValues = outputVector

    return multivariate_normal.logpdf(targetValues, mean = meanVectData, cov = covMatData)
    


def arcMAP(prior, data):
    """ Takes a prior and a set of input/output values and return the most likely arc with its loglikelihood
    """
    # NYI
    return {"LL":None,"Arc":None}

def arcML(data, returnEstimates = False):
    """ Returns the maximum likelihood arc for a set of input/output values
    """
    (inputVector, outputVector) = zip(*data)
    polyfit = np.polynomial.polynomial.polyfit(inputVector,outputVector,2)
    if returnEstimates:
        return list(reversed(polyfit)), np.polynomial.polynomial.polyval(inputVector,polyfit)
    else:
        return list(reversed(polyfit))


def normalizeX(dataSlice, linearSampling=True):
    # TODO: Automatically detect if linear sampling (beat-wise) or not (note-wise)
    if linearSampling:
        if len(dataSlice) == 1:
            return [(0,dataSlice[0])]
        else:
            return [(float(i)/(len(dataSlice)-1), dataPoint) for (i, dataPoint) in enumerate(dataSlice)]
    else:
        maxX,_ = data[-1]
        return [(float(x)/maxX, y) for x,y in data]


def knownSegmentationML(data, segmentation):
    y = [] # ML estimation of the denoised data
    models = [] # ML coefficient estimates
    lengths = []
    for (bound_curr, bound_next) in zip(segmentation,segmentation[1:]):
        dataPairs = normalizeX(data[(sscurr+1):ssnext+1])
        model, values = arcML(dataPairs,returnEstimates=True)
        models.append(model)
        y.extend(values)
        lengths.append(ssnext-sscurr)
    return y,models,lengths

if __name__ == "__main__":
    from testHelpers import *
    priors = arcPrior
    
    x,y = zip(*dataMultidim)
    y = np.array(y)[:,1]
    dataMultidim = zip(x,y)
    print(y)

    print("MeanVect",makeMeanVect(priors))
    print("VarVect",makeVarVect(priors))
    print("DesignMatrix",makeDesignMatrix(x))
    # print("NoiseCov",makeNoiseCov(priors,x))
    print("Likelihood",arcLikelihood(priors,dataMultidim))