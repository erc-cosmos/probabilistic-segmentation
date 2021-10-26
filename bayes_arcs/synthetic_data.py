"""Functions to generate synthetic data according to priors."""
import numpy as np

rng = np.random.default_rng()


def gen_data(segments, arc_prior):
    """Generate arcs drawn from arcPrior with the specified segment lengths."""
    data = []
    hidden = []
    for segment_size in segments:
        a = rng.normal(arc_prior['aMean'], arc_prior['aStd'])
        b = rng.normal(arc_prior['bMean'], arc_prior['bStd'])
        c = rng.normal(arc_prior['cMean'], arc_prior['cStd'])
        x = np.linspace(0, 1, num=segment_size)
        y = a * np.square(x) + b * x + c
        noise = rng.normal(0, arc_prior['noiseStd'], size=segment_size)
        hidden.extend(y)
        data.extend(y+noise)
    return hidden, data
