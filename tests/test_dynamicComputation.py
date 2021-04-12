"""Tests for the dynamicComputation module."""
import hypothesis
import hypothesis.strategies as st

import dynamicComputation as dc
import syntheticData as sd
import lengthPriors
from defaultVars import arcPrior
import functools

means = st.floats(min_value=0, allow_infinity=False)
stddevs = st.floats(min_value=1, exclude_min=True, allow_infinity=False)
xinputs = st.lists(st.floats(min_value=0), min_size=2, unique=True).map(sorted)


def normalPriors(dataLength):
    """Generate normal priors bound to given data length."""
    return st.builds(functools.partial(lengthPriors.NormalLengthPrior, xinputs=list(range(dataLength))),
                     means, stddevs, st.integers(min_value=2, max_value=50))


segmentLengthSets = st.lists(st.integers(min_value=2, max_value=30), min_size=2, max_size=5)

features = st.builds(functools.partial(sd.genData, arcPrior=arcPrior), segmentLengthSets)


# alphabeta takes a little longer than most tests
@hypothesis.settings(deadline=2000, max_examples=20)
@hypothesis.given(features)
def test_marginals_are_probability(data):
    """Check that marginals are probabilities."""
    hidden, data = data
    lengthPrior = lengthPriors.NormalLengthPrior(15, 5, list(range(len(data))), maxLength=20)
    posteriorMarginals = dc.runAlphaBeta(data, arcPrior, lengthPrior)
    assert all(0 <= p <= 1 for p in posteriorMarginals)
