"""Notebook to visualise priors without data."""
# %%
from bayes_arcs import dynamic_computation as dc
from bayes_arcs import length_priors
from bayes_arcs import segment_viz
import numpy as np


# %%

data_length = 240
length_prior = length_priors.GeometricLengthPrior(data_length, .9, 31, 20)

# %%
# lengthPrior = lengthPriors.EmpiricalLengthPrior(list(range(1,20)), dataLength=data_length, maxLength=20)


# %%
# lengthPriorParamsLoud = {
#     'mean': 11.8,
#     'stddev': 5.53,
#     'maxLength': 30
# }
# lengthPrior=lengthPriors.NormalLengthPrior(20, 5, list(range(data_length)), 40)


# %%
prior, prior_2d = dc.prior_marginals(data_length, length_prior, return_2d=True)


# %%
_ = segment_viz.plot_segment_beams(prior_2d, 25)


# %%

_ = segment_viz.plot_segment_with_signal(prior, np.zeros((data_length, 1)), list(range(data_length)), smoothing=1)


# %%
_ = segment_viz.plot_segment_raindrop(prior_2d, 25)


# %%


# %%


# %%
