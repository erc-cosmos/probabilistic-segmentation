"""Functions for scoring a prediction against a reference."""

import itertools as itt
from typing import Collection, Sequence, Set, Tuple

import numpy as np


def score_prob_segmentation(reference: Collection[int], estimation: Collection[float]) -> float:
    """Return a probabilistic score between a marginal estimation and an empirical reference."""
    # Quadratic Bayesian scoring
    score = 0.0
    for i, p in enumerate(estimation):
        if i in reference:
            score += 1 - (1-p)**2
        else:
            score += 1 - p**2
    score /= len(estimation)  # Normalise by the number of guesses
    return score


def count_matches(reference: Collection[int], estimation: Collection[int], tolerance: int) -> int:
    """Count matches between a reference and an estimation, excluding surjective mappings."""
    used_guess = set()
    used_ref = set()
    count = 0
    for guess, ref in itt.product(set(estimation), set(reference)):
        if abs(guess-ref) <= tolerance and guess not in used_guess and ref not in used_ref:
            count += 1
            used_guess.add(guess)
            used_ref.add(ref)
    return count


def precision(reference: Collection[int], estimation: Collection[int], tolerance: int) -> float:
    """Compute the (tolerant) precision between estimation and reference."""
    if len(estimation) == 0:
        return 0
    return count_matches(reference, estimation, tolerance)/float(len(set(estimation)))


def recall(reference: Collection[int], estimation: Collection[int], tolerance: int) -> float:
    """Compute the (tolerant) recall between estimation and reference."""
    return precision(estimation, reference, tolerance)


def frp_measures(reference: Collection[int], estimation: Collection[int], tolerance: int, weight: float = 1) \
        -> Tuple[float, float, float]:
    """Return all 3 classification measures (F, recall and precision)."""
    p = precision(reference, estimation, tolerance)
    r = recall(reference, estimation, tolerance)
    f = (1+weight**2)*p*r/(weight**2*p+r) if p != 0 != r else 0
    return f, r, p


def f_measure(reference: Collection[int], estimation: Collection[int], tolerance: int, weight: float = 1) -> float:
    """Compute the (tolerant) F-measure between estimation and reference."""
    return frp_measures(reference, estimation, tolerance, weight)[0]


def marginal2guess(marginals: Sequence[float], tolerance: int, threshold: float)\
        -> Tuple[Collection[int], np.ndarray]:
    """Construct an estimation from a marginal function."""
    convol = np.convolve(marginals, np.ones(2*tolerance+1), mode='same')
    guesses: Set[int] = set()
    above = False
    best_index: int = 0
    best_value = threshold
    for it, value in enumerate(convol):
        if value >= best_value:
            best_index = it
            best_value = value
            above = True
        elif above and value < threshold:  # We've reached the end of this run
            guesses.add(best_index)
            best_value = threshold  # reset best
            above = False
    return guesses, convol
