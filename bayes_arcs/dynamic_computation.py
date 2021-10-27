"""Algorithms for MAP estimation and PM computation."""
import itertools as itt
import math
from typing import Any, List, Optional, Tuple

import numpy as np

from . import length_priors
from . import single_arc as sa


def compute_maxima_a_posteriori(data, arc_prior, length_prior):
    """Construct MAPs matrix.

    This matrix's elements consist of the maximally likely arc considering the data and
    the corresponding log-likelihood over all valid slices of data (valued None otherwise).
    A slice of data is indexed by start and end index and is valid if start<end and
    if its length is less than the specified maximum.
    """
    maxima_a_posteriori = [[{"LL": None, "Arc": None} for end in range(len(data))] for start in range(len(data))]

    # Fill up subdiagonals (rest is zeroes)
    for start in range(len(data)):
        for end in range(start+1, length_prior.get_max_index(start)+1):
            maxima_a_posteriori[start][end] = sa.arc_max_a_posteriori(arc_prior, sa.normalize_x(data[start:end+1]))
    return maxima_a_posteriori


def compute_data_likelihood(data, arc_prior, length_prior, linear_sampling=True):
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
            lik_length = length_prior.eval_cond(start, end)
        except length_priors.ImpossibleCondition:  # Impossible start
            lik_length = 0
        return np.log(lik_length)
    return _compute_transition_matrix(data_length, wrapper)


def _compute_data_transition_matrix(data, arc_prior, linear_sampling=True):
    def wrapper(start, end):
        return sa.arc_likelihood(arc_prior, sa.normalize_x(data[start:end+1], linear_sampling=linear_sampling))
    return _compute_transition_matrix(len(data), wrapper)


def run_viterbi(data, arc_prior, length_prior, maxima_a_posteriori=None):
    """Run a modified Viterbi algorithm to compute the MAP arc sequence."""
    max_length = length_prior['max']
    # Compute MAP arcs if not provided
    if maxima_a_posteriori is None:
        maxima_a_posteriori = compute_maxima_a_posteriori(data, arc_prior, max_length)

    # Forward pass: compute optimal predecessors
    predecessors: List[Tuple[Optional[int], float, Any]] = [(None, 0, None)]
    for arc_end in range(1, len(data)):
        # Prior rescaling so it is a distribution (sums to 1)
        length_normalization = length_prior['cmf'][arc_end] if arc_end < max_length else length_prior['cmf'][-1]

        def ll_map(arc_start):
            # logp(data(0:arcStart), ZMAP(0:arcStart))
            ll_previous = predecessors[arc_start][1]
            # logp(data(arcStart+1:arcEnd) | [arcStart,arcEnd] in Z)
            ll_current = maxima_a_posteriori[arc_start][arc_end]["LL"]
            arc_length = arc_end-arc_start
            # logp([arcStart,arcEnd] in Z | [~,arcEnd] in Z)
            ll_length = math.log(length_prior['pmf'][arc_length] /
                                 length_normalization) if arc_length <= max_length else -math.inf
            return ll_previous + ll_current + ll_length
        best_start = max(range(max(0, arc_end-max_length), arc_end), key=ll_map)
        best_ll = ll_map(best_start)
        best_arc = maxima_a_posteriori[best_start][arc_end]
        predecessors.append((best_start, best_ll, best_arc))

    # Backtrack pass: work out the optimal path
    # Use a reverse list rather than prepending
    reverse_path = [predecessors[-1]]
    while reverse_path[-1][0] is not None:  # As long as we haven't reached the start
        reverse_path.append(predecessors[reverse_path[-1][0]])
    path = list(reversed(reverse_path))
    return path


def compute_alphas(length_prior, data_likelihoods):
    """Perform the Alpha phase of modified alpha-beta algorithm.

    Computes the joint likelihood of the data up to a point,
    and having an arc end at that point.
    This uses a recursive formulation with dynamic programming.
    """
    # TODO: Insert reference to recursive formula

    # Indices are shifted by 1 compared to the doc !!!
    # The data is assumed to start with an arc
    alphas = [0]  # Stored as log !
    data_len = len(data_likelihoods)
    for n in range(0, data_len):
        min_index = length_prior.get_min_index(n)
        ll_arc = data_likelihoods[min_index:n+1, n]
        alpha_components = alphas[min_index:n+1] + ll_arc
        max_increment_log = np.nanmax(alpha_components)
        alpha = np.log(np.nansum(np.exp(alpha_components - max_increment_log))) + \
            max_increment_log if max_increment_log != np.NINF else np.NINF
        alphas.append(alpha)
    return np.array(alphas)


def compute_betas(length_prior, data_likelihoods):
    """Perform the beta phase of modified alpha-beta algorithm.

    Computes the conditional likelihood of the data from a point to the end,
    assuming an arc begins at that point.
    This uses a recursive formulation with dynamic programming.
    """
    # TODO: Insert reference to recursive formula

    data_len = len(data_likelihoods)
    betas = np.full(data_len+1, np.nan)
    betas[data_len] = 0  # There is no more data to be observed past the end
    for n in reversed(range(0, data_len)):  # This is the backward pass
        max_index = length_prior.get_max_index(n)
        beta_components = betas[(n+1):(max_index+2)] + data_likelihoods[n, n:max_index+1]
        max_increment_log = np.nanmax(beta_components)
        beta = np.log(np.nansum(np.exp(beta_components - max_increment_log))) + max_increment_log \
            if max_increment_log != np.NINF else np.NINF

        betas[n] = beta

    return betas


def run_alpha_beta(data, arc_prior, length_prior, data_likelihoods=None, linear_sampling=True, return_2d=False):
    """Run the alpha-beta algorithm to compute posterior marginals on arc boundaries."""
    if data_likelihoods is None:
        data_likelihoods = compute_data_likelihood(data, arc_prior, length_prior, linear_sampling=linear_sampling)

    return _compute_marginals(length_prior=length_prior, data_likelihoods=data_likelihoods, return_2d=return_2d)


def _marginal_boundaries(alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Compute marginals for boundaries from alpha and beta probabilities.

    Args:
        alphas (1D Array): joint probability of the data up to index and a segment ending at index
        betas (1D Array): probability of the data past the index conditionned on a segment ending at index

    Returns:
        1D Array: probability of a segment ending at index
    """
    return np.exp([alpha + beta - alphas[-1] for (alpha, beta) in zip(alphas[1:], betas[1:])])


def _compute_marginals(length_prior, data_likelihoods, return_2d=False):
    alphas = compute_alphas(length_prior, data_likelihoods)
    betas = compute_betas(length_prior, data_likelihoods)

    marginals = _marginal_boundaries(alphas, betas)
    if return_2d:
        start_end_marginals = _marginal_segments(data_likelihoods, alphas, betas)
        return marginals, start_end_marginals
    else:
        return marginals


def _marginal_segments(data_likelihoods: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Compute marginals for segments based on alpha/beta probabilities and arc likelihoods.

    Args:
        DLs (2D np.ndarray): joint likelihood of the data between indices and the indices bounding a segment
        alphas (1D np.ndarray): joint probability of the data up to index and a segment ending at index
        betas (1D np.ndarray): probability of the data past the index conditionned on a segment ending at index

    Returns:
        2D np.ndarray: probability of a segment starting and ending at indices
    """
    start_end_marginals = np.zeros(np.shape(data_likelihoods))
    for (i, j) in np.ndindex(np.shape(start_end_marginals)):
        start_end_marginals[i, j] = np.exp(alphas[i]+betas[j+1]+data_likelihoods[i, j]-alphas[-1])
    return start_end_marginals


def prior_marginals(data_length, length_prior, return_2d=False):
    """Compute the prior marginals with a length-based segment prior."""
    data_likelihoods = _compute_prior_transition_matrix(data_length, length_prior)
    return _compute_marginals(length_prior, data_likelihoods, return_2d)


def data_marginals(data, length_prior, return_2d=False):
    """Compute the posterior marginals with no prior."""
    data_likelihoods = _compute_prior_transition_matrix(len(data), length_prior)
    return _compute_marginals(length_prior, data_likelihoods, return_2d)
