"""Algorithms for MAP estimation and PM computation."""
import itertools as itt
import math
from typing import Any, List, Optional, Tuple

import numpy as np

import lengthPriors
import singleArc as sa


def computeMAPs(data, arcPrior, lengthPrior):
    """Construct MAPs matrix.

    This matrix's elements consist of the maximally likely arc considering the data and
    the corresponding log-likelihood over all valid slices of data (valued None otherwise).
    A slice of data is indexed by start and end index and is valid if start<end and
    if its length is less than the specified maximum.
    """
    MAPs = [[{"LL": None, "Arc": None} for end in range(len(data))] for start in range(len(data))]

    # Fill up subdiagonals (rest is zeroes)
    for start in range(len(data)):
        for end in range(start+1, lengthPrior.getMaxIndex(start)+1):
            MAPs[start][end] = sa.arcMAP(arcPrior, sa.normalizeX(data[start:end+1]))
    return MAPs


def computeDataLikelihood(data, arc_prior, length_prior, linear_sampling=True):
    """Construct log-likelihood transition matrix.

    This matrix lists the log-likelihood as an undivided arc conditionned on its start
    of all valid slices of data.
    A slice of data is indexed by start and end index (inclusive) and is valid
    if start<=end and if its length is less than the specified maximum.
    """
    return (_compute_data_transition_matrix(data, arc_prior, linear_sampling)
            + _compute_prior_transition_matrix(len(data), length_prior))


def _compute_transition_matrix(data_length, func):
    trans_matrix = np.full((data_length, data_length), np.NINF)
    for start, end in itt.combinations_with_replacement(range(data_length), r=2):
        trans_matrix[start, end] = func(start, end)
    return trans_matrix


def _compute_prior_transition_matrix(data_length, length_prior):
    def wrapper(start, end):
        try:
            lik_length = length_prior.evalCond(start, end)
        except lengthPriors.ImpossibleCondition:  # Impossible start
            lik_length = 0
        return np.log(lik_length)
    return _compute_transition_matrix(data_length, wrapper)


def _compute_data_transition_matrix(data, arc_prior, linear_sampling=True):
    def wrapper(start, end):
        return sa.arcLikelihood(arc_prior, sa.normalizeX(data[start:end+1], linearSampling=linear_sampling))
    return _compute_transition_matrix(len(data), wrapper)


def runViterbi(data, arcPrior, lengthPrior, MAPs=None):
    """Run a modified Viterbi algorithm to compute the MAP arc sequence."""
    maxLength = lengthPrior['max']
    # Compute MAP arcs if not provided
    if MAPs is None:
        MAPs = computeMAPs(data, arcPrior, maxLength)

    # Forward pass: compute optimal predecessors
    predecessors: List[Tuple[Optional[int], float, Any]] = [(None, 0, None)]
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


def computeAlphas(lengthPrior, DLs):
    """Perform the Alpha phase of modified alpha-beta algorithm.

    Computes the joint likelihood of the data up to a point,
    and having an arc end at that point.
    This uses a recursive formulation with dynamic programming.
    """
    # TODO: Insert reference to recursive formula

    # Indices are shifted by 1 compared to the doc !!!
    # The data is assumed to start with an arc
    alphas = [0]  # Stored as log !
    N = len(DLs)
    for n in range(0, N):
        minIndex = lengthPrior.getMinIndex(n)
        llArc = DLs[minIndex:n+1, n]
        alphaComponents = alphas[minIndex:n+1] + llArc
        maxIncrementLog = np.nanmax(alphaComponents)
        alpha = np.log(np.nansum(np.exp(alphaComponents - maxIncrementLog))) + \
            maxIncrementLog if maxIncrementLog != np.NINF else np.NINF
        alphas.append(alpha)
    return np.array(alphas)


def computeBetas(lengthPrior, DLs):
    """Perform the beta phase of modified alpha-beta algorithm.

    Computes the conditional likelihood of the data from a point to the end,
    assuming an arc begins at that point.
    This uses a recursive formulation with dynamic programming.
    """
    # TODO: Insert reference to recursive formula

    N = len(DLs)
    betas = np.full(N+1, np.nan)
    betas[N] = 0  # There is no more data to be observed past the end
    for n in reversed(range(0, N)):  # This is the backward pass
        maxIndex = lengthPrior.getMaxIndex(n)
        betaComponents = betas[(n+1):(maxIndex+2)] + DLs[n, n:maxIndex+1]
        maxIncrementLog = np.nanmax(betaComponents)
        beta = np.log(np.nansum(np.exp(betaComponents - maxIncrementLog))) + maxIncrementLog \
            if maxIncrementLog != np.NINF else np.NINF

        betas[n] = beta

    return betas


def runAlphaBeta(data, arcPrior, lengthPrior, DLs=None, linearSampling=True, return2D=False):
    """Run the alpha-beta algorithm to compute posterior marginals on arc boundaries."""
    if DLs is None:
        DLs = computeDataLikelihood(data, arcPrior, lengthPrior, linear_sampling=linearSampling)

    return _compute_marginals(length_prior=lengthPrior, DLs=DLs, return2D=return2D)


def _marginal_boundaries(alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Compute marginals for boundaries from alpha and beta probabilities.

    Args:
        alphas (1D Array): joint probability of the data up to index and a segment ending at index
        betas (1D Array): probability of the data past the index conditionned on a segment ending at index

    Returns:
        1D Array: probability of a segment ending at index
    """
    return np.exp([alpha + beta - alphas[-1] for (alpha, beta) in zip(alphas[1:], betas[1:])])


def _compute_marginals(length_prior, DLs, return2D=False):
    alphas = computeAlphas(length_prior, DLs)
    betas = computeBetas(length_prior, DLs)

    marginals = _marginal_boundaries(alphas, betas)
    if return2D:
        start_end_marginals = _marginal_segments(DLs, alphas, betas)
        return marginals, start_end_marginals
    else:
        return marginals


def _marginal_segments(DLs: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Compute marginals for segments based on alpha/beta probabilities and arc likelihoods.

    Args:
        DLs (2D np.ndarray): joint likelihood of the data between indices and the indices bounding a segment
        alphas (1D np.ndarray): joint probability of the data up to index and a segment ending at index
        betas (1D np.ndarray): probability of the data past the index conditionned on a segment ending at index

    Returns:
        2D np.ndarray: probability of a segment starting and ending at indices
    """
    start_end_marginals = np.zeros(np.shape(DLs))
    for (i, j) in np.ndindex(np.shape(start_end_marginals)):
        start_end_marginals[i, j] = np.exp(alphas[i]+betas[j+1]+DLs[i, j]-alphas[-1])
    return start_end_marginals


def prior_marginals(data_length, length_prior, return2D=False):
    """Compute the prior marginals with a length-based segment prior."""
    DLs = _compute_prior_transition_matrix(data_length, length_prior)
    return _compute_marginals(length_prior, DLs, return2D)


def data_marginals(data, length_prior, return2D=False):
    """Compute the posterior marginals with no prior."""
    DLs = _compute_prior_transition_matrix(len(data), length_prior)
    return _compute_marginals(length_prior, DLs, return2D)
