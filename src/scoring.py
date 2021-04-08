#!/usr/bin/python3
""" Functions for scoring a prediction against a reference
"""
import math

def scoreProbSegmentation(reference, estimation):
    # Quadratic Bayesian scoring
    score = 0.0
    for i,p in enumerate(estimation):
        if i in reference:
            score+= 1 - (1-p)**2
        else:
            score+= 1 - p**2
    score /= len(estimation) # Normalise by the number of guesses
    return score

def countMatches(reference, estimation, tolerance):
    used_guess = set()
    used_ref = set()
    count = 0
    for guess in estimation:
        for ref in reference:
            if abs(guess-ref) <= tolerance and guess not in used_guess and ref not in used_ref:
                count+= 1
                used_guess.add(guess)
                used_ref.add(ref)
    return count

def precision(reference, estimation, tolerance):
    return countMatches(reference, estimation, tolerance)/float(len(estimation))

def recall(reference, estimation, tolerance):
    return precision(estimation,reference,tolerance)

def frpMeasures(reference, estimation, tolerance, weight=1):
    if len(estimation) == 0 or len(reference) == 0:
        print("Warning: empty estimation or reference found")
        return 0,0,0
    p = precision(reference,estimation,tolerance)
    r = recall(reference,estimation,tolerance)
    f = (1+weight**2)*p*r/(weight**2*p+r)
    return f,r,p

def fMeasure(reference, estimation, tolerance, weight=1):
    return frpMeasures(reference, estimation, tolerance, weight)[0]


import numpy as np

def marginal2guess(marginals, tolerance, threshold):
    convol = np.convolve(marginals,np.ones(2*tolerance+1),mode='same')
    guesses = []
    above = False
    bestValue = threshold
    for it,value in enumerate(convol):
        if value>= bestValue:
            bestIndex = it
            bestValue = value
            above = True
        elif above and value<threshold: #We've reached the end of this run
            guesses.append(bestIndex)
            bestValue = threshold # reset best
            above = False
    return guesses,convol