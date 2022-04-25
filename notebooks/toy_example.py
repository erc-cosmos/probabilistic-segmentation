# %%
import math

from bayes_arcs import dynamic_computation as dc
from bayes_arcs import length_priors
from bayes_arcs import segment_viz
from bayes_arcs import synthetic_data
from bayes_arcs.default_priors import arc_prior_tempo as arc_prior
import numpy as np

# %%


# %% [markdown]
#  # Simple application of Arcs to a toy example

# %% [markdown]
#  Let's get some default priors for the arcs.

# %%
# Generate some segments
rng = np.random.default_rng()
test_segment = [math.ceil(x) for x in rng.normal(15, 5, size=10)]
print(test_segment)


# %%
# Generate the actual data
(truth, sample_data) = synthetic_data.gen_data(test_segment, arc_prior)
print(len(sample_data), sample_data)


# %%
# Bind a length prior to our data
length_prior = length_priors.NormalLengthPrior(mean=15, stddev=5, x=list(range(len(sample_data))), max_length=30)


# %%
# Compute posterior marginals
posterior_marginals, post_segments = dc.compute_both_posteriors(sample_data, arc_prior, length_prior)


# %%
fig = segment_viz.plot_segment_with_signal(posterior_marginals, sample_data, data_time=range(
    len(sample_data)), input_label='Synthetic data', smoothing=1)
fig.get_axes()[0].plot(truth, "orange")
# fig.get_axes()[0].legend(['noisy', 'pure'], loc='upper left')

# %%
segment_viz.plot_segment_matrix(post_segments, smoothing=1)

# %%
segment_viz.plot_segment_raindrop(post_segments, max_length=30, ratio=2.5)

# %%
segment_viz.plot_segment_beams(post_segments, max_length=30, ratio=2.5)

# %%
