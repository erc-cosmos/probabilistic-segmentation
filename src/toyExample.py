# Test with synthetic data
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from dynamicComputation import *
from scoring import *


# Prior structures, TODO:should end up somewhere else
arcPrior = {
    # Gaussian priors on the parameters of ax^2 + bx + c
    # TODO: Put in reasonable values
    'aMean': -2,
    'aStd': 1,
    'bMean': 0,
    'bStd': 5,
    'cMean': 10,
    'cStd': 10,
    'noiseStd': 10
}

# Prior structures, TODO:should end up somewhere else
arcPrior = {
    # Gaussian priors on the parameters of ax^2 + bx + c
    # TODO: Put in reasonable values
    'aMean': -181,
    'aStd': 93,
    'bMean': 159,
    'bStd': 106,
    'cMean': 107,
    'cStd': 31,
    'noiseStd': 40
}

def genData(segments, arcPrior=arcPrior):
    data = []
    hidden = []
    for segmentSize in segments:
        a = rng.normal(arcPrior['aMean'],arcPrior['aStd'])
        b = rng.normal(arcPrior['bMean'],arcPrior['bStd'])
        #c = hidden[-1] if len(hidden)!=0 else rng.normal(arcPrior['cMean'],arcPrior['cStd'])
        c = rng.normal(arcPrior['cMean'],arcPrior['cStd'])
        x = np.linspace(0, 1, num=segmentSize)
        y = a * np.square(x) + b * x + c
        noise = rng.normal(0,arcPrior['noiseStd'], size=segmentSize)
        hidden.extend(y)
        data.extend(y+noise)
    return hidden,data

pmf = sp.stats.binom.pmf(list(range(31)),30,.5)
pmf[0] = 0
pmf[1] = 0
pmf = [float(m)/sum(pmf) for m in pmf]

cmf = np.cumsum(pmf)

def priorEvalCond(N,a,b):
    pmf = sp.stats.binom.pmf(list(range(31)),30,.5)
    pmf[0] = 0
    pmf[1] = 0
    pmf = [float(m)/sum(pmf) for m in pmf]
    cmf = np.cumsum(pmf)

    maxLength = 30
    cum = 1 if N-a > maxLength or cmf[N-a]==0 else cmf[N-a]
    prob = 1 if b-a > maxLength else pmf[b-a]
    return prob/cum

# Assume 10% probability of arc ending at any point
def priorEval(N,a,b):
    return .1*.1*pow(1-.1,b-a-1)


lengthPrior = {
    # Free prior on the length of arcs
    # needs to be scaled to the range of possible values
    'max': 30,  # Any longer will have 0 likelihood -- useful for bounding complexity
    'cmf':cmf,
    'pmf':pmf,
    # 'cmf': [0, 0, .05, .20, .35, .65, .80, .95, 1],  # Cumulative mass function
    # 'pmf': [0, 0, .05, .15, .15, .3, .15, .15, .05]  # Probability mass function
    'eval': priorEval,
    'evalCond': priorEvalCond 
}

rng = np.random.default_rng()
testSegment = rng.binomial(30,.5,size=10)


(truth, sampleData) = genData(testSegment)
posteriorMarginals = runAlphaBeta(sampleData,arcPrior,lengthPrior)

fig, ax1 = plt.subplots()
ax1.plot(sampleData)
ax1.plot(truth)
print(posteriorMarginals)

#TODO: second axis
ax2 = ax1.twinx()

ax2.plot(posteriorMarginals[1:], 'r')
guesses, convol = marginal2guess(posteriorMarginals[1:],3,.5)
ax2.plot(convol, 'g')
ax2.vlines(guesses,ymin=0,ymax=1)
plt.show()