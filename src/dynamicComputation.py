"""Algorithms for MAP estimation and PM computation."""
import singleArc as sa
import math
import lengthPriors
import numpy as np
import itertools as itt


def computeMAPs(data, arcPrior, lengthPrior):
    """Construct MAPs matrix.

    This matrix's elements consist of the maximally likely arc considering the data and
    the corresponding log-likelihood over all valid slices of data (valued None otherwise).
    A slice of data is indexed by start and end index and is valid if start<end and
    if its length is less than the specified maximum.
    """
    MAPs = [[{"LL": None, "Arc": None} for end in range(len(data))] for start in range(len(data))]

    # Fill up subdiagonals (rest is zeroes)
    # for end in range(len(data)):
    #     for start in range(max(0, end - maxLength), end):
    for start in range(len(data)):
        for end in range(start+1, lengthPrior.getMaxIndex(start)+1):
            MAPs[start][end] = sa.arcMAP(arcPrior, sa.normalizeX(data[start:end+1]))
    return MAPs


def computeDataLikelihood(data, arcPrior, lengthPrior, linearSampling=True):
    """Construct log-likelihood matrix.

    This matrix lists the log-likelihood as an undivided arc
    of all valid slices of data (0 otherwise).
    A slice of data is indexed by start and end index and is valid
    if start<end and if its length is less than the specified maximum.
    """
    N = len(data)
    # Initialize log-Likelihood matrix
    DLs = np.full((N, N), np.nan)

    # Fill up subdiagonals (rest is zeroes)
    for start, end in itt.combinations_with_replacement(range(N), r=2):
        # (log-)Likelihood of the data assuming there is an arc
        llikData = sa.arcLikelihood(arcPrior, sa.normalizeX(data[start:end+1], linearSampling=linearSampling))
        try:
            # (log-)Likelihood of the arc assuming its start
            llikLength = np.log(lengthPrior.evalCond(start, end))
        except lengthPriors.ImpossibleCondition:  # The arc's start is impossible
            llikLength = np.nan
        DLs[start, end] = llikData + llikLength
    return DLs


def runViterbi(data, arcPrior, lengthPrior, MAPs=None):
    """Run a modified Viterbi algorithm to compute the MAP arc sequence."""
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
            llLength = math.log(lengthPrior['pmf'][arcLength] /
                                lengthNormalization) if arcLength <= maxLength else -math.inf
            return llPrevious + llCurrent + llLength
        bestStart = max(range(max(0, arcEnd-maxLength), arcEnd), key=llMap)
        bestLL = llMap(bestStart)
        bestArc = MAPs[bestStart][arcEnd]
        predecessors.append((bestStart, bestLL, bestArc))

    # Backtrack pass: work out the optimal path
    # Use a reverse list rather than prepending
    reversePath = [predecessors[-1]]
    while reversePath[-1][0] is not None:  # As long as we haven't reached the start
        reversePath.append(predecessors[reversePath[-1][0]])
    path = list(reversed(reversePath))
    return path


def computeAlphas(arcPrior, lengthPrior, DLs):
    """Perform the Alpha phase of modified alpha-beta algorithm.

    Computes the joint likelihood of the data up to a point,
    and having an arc end at that point.
    This uses a recursive formulation with dynamic programming.
    """
    alphaMatrix = []
    # TODO: Insert reference to recursive formula

    # Indices are shifted by 1 compared to the doc !!!
    # The data is assumed to start with an arc
    alphas = [0]  # Stored as log !
    N = len(DLs)
    alphaMatrix = np.full((N+1, N+1), np.nan)
    for n in range(0, N):
        minIndex = lengthPrior.getMinIndex(n)
        llArc = DLs[minIndex:n+1, n]
        alphaComponents = alphas[minIndex:n+1] + llArc
        alphaMatrix[minIndex:n+1, n] = alphaComponents
        maxIncrementLog = np.nanmax(alphaComponents)
        alpha = np.log(np.nansum(np.exp(alphaComponents - maxIncrementLog))) + \
            maxIncrementLog if maxIncrementLog != np.NINF else np.NINF
        # for i in range(max(0, n-maxLength), n):
        # for i in range(lengthPrior.getMinIndex(n), n+1):
        #     # mu(D[n, i]) x lambda(n,i) in the doc
        #     llArc = DLs[n, i]

        #     alphaIncrementLog = alphas[i] + llArc
        #     alphaMatrix[n, i] = alphaIncrementLog
        # maxIncrementLog = max(alphaMatrix[n])
        # alpha = np.log(np.nansum(np.exp(alphaMatrix[n, ] - maxIncrementLog))) + \
        #     maxIncrementLog if maxIncrementLog != np.NINF else np.NINF
        alphas.append(alpha)
    return alphas


def computeBetas(arcPrior, lengthPrior, DLs):
    """Perform the beta phase of modified alpha-beta algorithm.

    Computes the conditional likelihood of the data from a point to the end,
    assuming an arc begins at that point.
    This uses a recursive formulation with dynamic programming.
    """
    betaMatrix = []
    # TODO: Insert reference to recursive formula

    N = len(DLs)
    betaMatrix = np.full((N+1, N+1), np.nan)
    betas = np.full(N+1, np.nan)
    # betas = [0 for foo in range(N+1)]  # Stored as log
    betas[N] = 0  # There is no more data to be observed past the end
    for n in reversed(range(0, N)):  # This is the backward pass
        maxIndex = lengthPrior.getMaxIndex(n)
        # i = range(n, lengthPrior.getMaxIndex(n)+1)
        betaComponents = betas[(n+1):(maxIndex+2)] + DLs[n, n:maxIndex+1]
        betaMatrix[n, n:maxIndex+1] = betaComponents
        # for i in range(n, lengthPrior.getMaxIndex(n)+1):
        #     # mu(D[arcStart, arcEnd]) x lambda(arcStart,arcEnd) in the doc
        #     llArc = DLs[n, i]
        #     betaIncrementLog = betas[i+1] + llArc
        #     betaMatrix[n, i] = betaIncrementLog

        maxIncrementLog = np.nanmax(betaComponents)
        beta = np.log(np.nansum(np.exp(betaComponents - maxIncrementLog))) + maxIncrementLog \
            if maxIncrementLog != np.NINF else np.NINF

        betas[n] = beta

    return betas


def runAlphaBeta(data, arcPrior, lengthPrior, DLs=None, linearSampling=True):
    """Run the alpha-beta algorithm to compute posterior marginals on arc boundaries."""
    if DLs is None:
        DLs = computeDataLikelihood(data, arcPrior, lengthPrior)

    alphas = computeAlphas(arcPrior, lengthPrior, DLs)
    betas = computeBetas(arcPrior, lengthPrior, DLs)

    return np.exp([alpha + beta - alphas[-1] for (alpha, beta) in zip(alphas[1:], betas[1:])])
