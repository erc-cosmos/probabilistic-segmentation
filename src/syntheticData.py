"""Functions to generate synthetic data according to priors."""
import numpy as np

rng = np.random.default_rng()

def genData(segments, arcPrior):
    """Generate arcs drawn from arcPrior with the specified segment lengths."""
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