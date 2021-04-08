""" Functions for scoring a prediction against a reference """

import numpy as np
import itertools as itt


def scoreProbSegmentation(reference, estimation):
    """ Returns a probabilistic score between a marginal estimation and an empirical reference """
    # Quadratic Bayesian scoring
    score = 0.0
    for i, p in enumerate(estimation):
        if i in reference:
            score += 1 - (1-p)**2
        else:
            score += 1 - p**2
    score /= len(estimation)  # Normalise by the number of guesses
    return score


def countMatches(reference, estimation, tolerance):
    """ Counts matches between a reference and an estimation, excluding surjective mappings """
    used_guess = set()
    used_ref = set()
    count = 0
    for guess, ref in itt.product(estimation, reference):
        if abs(guess-ref) <= tolerance and guess not in used_guess and ref not in used_ref:
            count += 1
            used_guess.add(guess)
            used_ref.add(ref)
    return count


def precision(reference, estimation, tolerance):
    """ Computes the (tolerant) precision between estimation and reference """
    if len(estimation) == 0:
        return 0
    return countMatches(reference, estimation, tolerance)/float(len(estimation))


def recall(reference, estimation, tolerance):
    """ Computes the (tolerant) recall between estimation and reference """
    return precision(estimation, reference, tolerance)


def frpMeasures(reference, estimation, tolerance, weight=1):
    """ Returns all 3 classification measures (F, recall and precision) """
    p = precision(reference, estimation, tolerance)
    r = recall(reference, estimation, tolerance)
    f = (1+weight**2)*p*r/(weight**2*p+r) if p != 0 != r else 0
    return f, r, p


def fMeasure(reference, estimation, tolerance, weight=1):
    """ Computes the (tolerant) F-measure between estimation and reference """
    return frpMeasures(reference, estimation, tolerance, weight)[0]


def marginal2guess(marginals, tolerance, threshold):
    """ Constructs an estimation from a marginal function """
    convol = np.convolve(marginals, np.ones(2*tolerance+1), mode='same')
    guesses = []
    above = False
    bestValue = threshold
    for it, value in enumerate(convol):
        if value >= bestValue:
            bestIndex = it
            bestValue = value
            above = True
        elif above and value < threshold:  # We've reached the end of this run
            guesses.append(bestIndex)
            bestValue = threshold  # reset best
            above = False
    return guesses, convol
