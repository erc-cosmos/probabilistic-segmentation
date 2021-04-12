# Algorithms for MAP estimation and PM computation
from singleArc import *
import math

alphaMatrix = []
betaMatrix = []


def computeMAPs(data, arcPrior, lengthPrior):
    """ Initializes MAPs matrix
        This matrix's elements consist of the maximally likely arc considering the data and
        the corresponding log-likelihood over all valid slices of data (valued None otherwise).
        A slice of data is indexed by start and end index and is valid if start<end and
        if its length is less than the specified maximum.
    """
    MAPs = [[{"LL": None, "Arc": None}
             for end in range(len(data))] for start in range(len(data))]

    # Fill up subdiagonals (rest is zeroes)
    # for end in range(len(data)):
    #     for start in range(max(0, end - maxLength), end):
    for start in range(len(data)):
        for end in range(start+1, lengthPrior.getMaxIndex(start)+1):
            MAPs[start][end] = arcMAP(arcPrior, normalizeX(data[start:end+1]))
    return MAPs


def computeDataLikelihood(data, arcPrior, lengthPrior):
    """ Initializes log-likelihood matrix
        This matrix lists the log-likelihood as an undivided arc
        of all valid slices of data (0 otherwise).
        A slice of data is indexed by start and end index and is valid
        if start<end and if its length is less than the specified maximum.
    """
    # Initialize log-Likelihood matrix
    DLs = [[0 for end in range(len(data))] for start in range(len(data))]

    # Fill up subdiagonals (rest is zeroes)
    for start in range(len(data)):
        for end in range(start+1, lengthPrior.getMaxIndex(start)+1):
            scaling = 0
            DLs[start][end] = arcLikelihood(
                arcPrior, normalizeX(data[start:end+1])) - scaling
    return DLs


def runViterbi(data, arcPrior, lengthPrior, MAPs=None):
    """ Runs a modified Viterbi algorithm to compute the MAP arc sequence
    """
    maxLength = lengthPrior['max']
    # Compute MAP arcs if not provided
    if MAPs is None:
        MAPs = computeMAPs(data, arcPrior, maxLength)

    # Forward pass: compute optimal predecessors
    predecessors = [(None, 0, None)]
    for arcEnd in range(1, len(data)):
        # Prior rescaling so it is a distribution (sums to 1)
        lengthNormalization = lengthPrior['cmf'][arcEnd] if arcEnd < maxLength else lengthPrior['cmf'][-1]

        def llMap(arcStart):
            # logp(data(0:arcStart), ZMAP(0:arcStart))
            llPrevious = predecessors[arcStart][1]
            # logp(data(arcStart+1:arcEnd) | [arcStart,arcEnd] in Z)
            llCurrent = MAPs[arcStart][arcEnd]["LL"]
            arcLength = arcEnd-arcStart
            # logp([arcStart,arcEnd] in Z | [~,arcEnd] in Z)
            llLength = math.log(
                lengthPrior['pmf'][arcLength]/lengthNormalization) if arcLength <= maxLength else -math.inf
            return llPrevious + llCurrent + llLength
        bestStart = max(range(max(0, arcEnd-maxLength), arcEnd), key=llMap)
        bestLL = llMap(bestStart)
        bestArc = MAPs[bestStart][arcEnd]
        predecessors.append((bestStart, bestLL, bestArc))

    # Backtrack pass: work out the optimal path
    # Use a reverse list rather than prepending
    reversePath = [predecessors[-1]]
    while reversePath[-1][0] is not None:  # As long as we haven't reached the start
        reversePath.append(predecessor[reversePath[-1][0]])
    path = list(reversed(reversePath))
    return path


def computeAlphas(arcPrior, lengthPrior, DLs):
    """ Alpha phase of modified alpha-beta algorithm
        Computes the joint likelihood of the data up to a point,
        and having an arc end at that point.
        This uses a recursive formulation with dynamic programming.
    """
    maxLength = lengthPrior.max
    # TODO: Insert reference to recursive formula

    # Indices are shifted by 1 compared to the doc !!!
    # alpha(0) = 0
    alphas = [0]  # Stored as log !
    N = len(DLs)
    alphaMatrix = np.NINF * np.ones((N+1, N+1))
    for n in range(0, N):
        # for i in range(max(0, n-maxLength), n):
        for i in range(lengthPrior.getMinIndex(n), n):
            arcLength = n-i
            # mu(D[arcStart+1, arcEnd]) in the doc
            llikData = DLs[i][n]
            # p([arcStart,arcEnd] in Z | [~,arcEnd] in Z)
            # lambda(arcStart,arcEnd) in the doc
            llikLength = np.log(lengthPrior.evalCond(i, n))
            # print(likData/scaling)
            alphaIncrementLog = alphas[i] + llikLength + llikData
            alphaMatrix[n][i] = alphaIncrementLog
        maxIncrementLog = max(alphaMatrix[n])
        alpha = np.log(np.sum(np.exp(alphaMatrix[n]-maxIncrementLog))) + \
            maxIncrementLog if maxIncrementLog != np.NINF else np.NINF
        alphas.append(alpha)
    return alphas


def computeBetas(arcPrior, lengthPrior, DLs):
    """ Beta phase of modified alpha-beta algorithm
        Computes the conditional likelihood of the data from a point to the end,
        assuming an arc begins at that point.
        This uses a recursive formulation with dynamic programming.
    """
    maxLength = lengthPrior.max
    # TODO: Insert reference to recursive formula

    N = len(DLs)
    betaMatrix = np.NINF * np.ones((N+1, N+1))
    betas = [0 for foo in range(N+1)]  # Stored as log
    betas[-1] = 0
    for n in reversed(range(-1, N-1)):
        beta = 0
        # for i in range(n+2,min(N,n+maxLength+1)):
        for i in range(n+2, lengthPrior.getMaxIndex(n)+1):
            arcLength = i-n
            # mu(D[arcStart+1, arcEnd]) in the doc
            llikData = DLs[n+1][i]
            # p([arcStart,arcEnd] in Z | [~,arcEnd] in Z)
            # lambda'(arcStart,arcEnd) in the doc
            llikLength = np.log(lengthPrior.evalCond(n+1, i))
            betaIncrementLog = betas[i+1] + llikData + llikLength
            betaMatrix[n][i] = betaIncrementLog

        maxIncrementLog = max(betaMatrix[n])
        beta = np.log(np.sum(np.exp(betaMatrix[n]-maxIncrementLog))) + \
            maxIncrementLog if maxIncrementLog != np.NINF else np.NINF
        betas[n+1] = beta

    return betas


def runAlphaBeta(data, arcPrior, lengthPrior, DLs=None, linearSampling=True):
    """ Modified alpha-beta algorithm to compute posterior marginals on arc boundaries
    """
    if DLs is None:
        DLs = computeDataLikelihood(data, arcPrior, lengthPrior)

    alphas = computeAlphas(arcPrior, lengthPrior, DLs)
    betas = computeBetas(arcPrior, lengthPrior, DLs)

    # print(np.round(alphas,3))
    # print(np.round(betas,3))

    return np.exp([alpha + beta - alphas[-1] for (alpha, beta) in zip(alphas, betas)])
