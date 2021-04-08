arcPrior = {
    # Gaussian priors on the parameters of ax^2 + bx + c
    'aMean': 10,
    'aStd': 1,
    'bMean': 20,
    'bStd': 2,
    'cMean': 30,
    'cStd': 3,
    'noiseStd': 10
    }
arcPrior2 = {
    # Gaussian priors on the parameters of ax^2 + bx + c
    'aMean': -10,
    'aStd': 1,
    'bMean': -20,
    'bStd': 2,
    'cMean': -30,
    'cStd': 3,
    'noiseStd': 1
}
dataMultidim = [(0,[2,3]),(0.1,[2,3]),(0.3,[2,3]),(0.4,[2,3]),(0.7,[2,3])]
data1D = [(0,3),(0.1,3),(0.3,3),(0.4,3),(0.7,3)]