#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import csv

from dynamicComputation import *
from scoring import *
from readers import *
from writers import *
from lengthPrior import *

############################################
# Prior structures, TODO:should end up somewhere else
# Set for tempo
# arcPrior = { # Set after first 10 each of M06-1 and M06-2
#     # Gaussian priors on the parameters of ax^2 + bx + c
#     # TODO: Put in reasonable values
#     'aMean': -181,
#     'aStd': 93,
#     'bMean': 159,
#     'bStd': 106,
#     'cMean': 107,
#     'cStd': 31,
#     'noiseStd': 18.1
# }
# lengthPriorParams = {
#     'mean':14.7,
#     'stddev':5.95,
#     'maxLength':30
# }

# Set for loudness
arcPrior = { # Set after first 10 each of M06-1 and M06-2
    # Gaussian priors on the parameters of ax^2 + bx + c
    'aMean': -0.73,
    'aStd': 0.55,
    'bMean': 0.68,
    'bStd': 0.60,
    'cMean': 0.41,
    'cStd': 0.19,
    'noiseStd': 0.039
}
lengthPriorParams = {
    'mean':11.8,
    'stddev':5.53,
    'maxLength':30
}


#####################

def batchRun(keyList,dataList, verbose = False):
    results = {}
    if verbose:
        N = len(keyList)
        i = 0
        print("Running segmentation")
    for key,data in zip(keyList,dataList):
        lengthPrior = NormalLengthPrior(lengthPriorParams['mean'],lengthPriorParams['stddev'],range(len(data)),lengthPriorParams['maxLength'])
        posteriorMarginals = runAlphaBeta(data,arcPrior,lengthPrior)
        guess,_ = marginal2guess(posteriorMarginals,3, .5)
        results[key] = (posteriorMarginals,guess)
        if verbose:
            i=i+1
            print("Completed run "+str(i)+"/"+str(N))
    return results

def batchEvaluate(keyList, refList, rawResults):
    measures = {}
    for key,ref in zip(keyList,refList):
        marginals, guess = rawResults[key]
        Q = scoreProbSegmentation(ref,marginals)
        F,R,P = frpMeasures(ref,guess, 3)
        measures[key] = Q,F,R,P
    return measures

fullData = readAllMazurkaDataAndSeg()
# Minirun for debugging
# fullData = fullData[0:3]

keyList = [piece+"//"+pid for piece,pid,tim,seg in fullData]
_,_,dataList,segList = zip(*fullData)

rawResults = batchRun(keyList,dataList,verbose=True)

for key,(marginals,_) in rawResults.items():
    writeMarginals("results/"+key,marginals)

measures = batchEvaluate(keyList,segList,rawResults)

print(measures)

writeMeasures("results/early_newprior_loudness_2021-02-08.csv",measures)
