# %% [markdown]
#  # Simple test with real data

# %%

import itertools as itt

from bayes_arcs import default_priors
from bayes_arcs import dynamic_computation as dc
from bayes_arcs import length_priors
from bayes_arcs import readers
from bayes_arcs import segment_viz
from bayes_arcs.default_priors import \
    length_prior_params_tempo as length_prior_params
import numpy as np
import pandas as pd


# %% [markdown]
#  # Loading some data

# %%
timings_data = readers.read_all_mazurka_timings_and_seg(
    timing_path="data/beat_time", seg_path="data/deaf_structure_tempo")
dyn_data = readers.read_all_mazurka_data_and_seg(timing_path="data/beat_dyn", seg_path="data/deaf_structure_loudness")

full_data = [(piece, interpret, tempo, tempo_seg, dyn, dyn_seg)
             for ((piece, interpret, tempo, tempo_seg), (piece2, interpret2, dyn, dyn_seg))
             in itt.product(timings_data, dyn_data)
             if interpret == interpret2]
full_data = readers.load_mazurka_dataset_with_annot()


# %%
# arcPrior = [arcPriorTempo, arcPriorLoud]
arc_prior = default_priors.arc_prior_tempo


# %% [markdown]
#  # Get it Running

# %%
# Unpack the data
(piece, interpret, tempo, tempo_seg, dyn, dyn_seg) = full_data.loc[0, :]

piece_formatted = piece[16:20]
print(piece_formatted, interpret)

# sampleData = list(zip(tempo, dyn[1:]))
sample_data, times = tempo
sample_data = list(sample_data)
segs = (tempo_seg, dyn_seg)

tatums = list(range(len(sample_data)))

# Manually correct large anomalies
sample_data[106] = 200
sample_data[195] = 200


# %% [markdown]
#  ## Using a Gaussian Prior

# %%

length_prior = length_priors.NormalLengthPrior(length_prior_params['mean'], length_prior_params['stddev'], range(
    len(sample_data)), length_prior_params['maxLength'])

post_boundaries, post_segments = dc.run_alpha_beta(sample_data, arc_prior, length_prior, return_2d=True)


# %%
_ = segment_viz.plot_segment_with_signal(post_marginals=post_boundaries,
                                         data=sample_data, data_time=list(range(len(sample_data))), smoothing=1)
_ = segment_viz.plot_segment_beams(post2_bidim=post_segments, max_length=length_prior.max_length)


# %% [markdown]
#  ## Using a Geometric Prior

# %%

length_prior = length_priors.GeometricLengthPrior(len(sample_data), 0.95, min_length=5, max_length=30)

post_boundaries, post_segments = dc.run_alpha_beta(sample_data, arc_prior, length_prior, return_2d=True)
# _ = segment_viz.plot_segment_with_signal(post_marginals=post_boundaries, data=sampleData, data_time=sampleData)
# _ = segment_viz.plot_segment_beams(post2_bidim=post_segments, length_prior=length_prior)


# %%
_ = segment_viz.plot_segment_with_signal(post_marginals=post_boundaries,
                                         data=sample_data, data_time=list(range(len(sample_data))), smoothing=1)
_ = segment_viz.plot_segment_beams(post2_bidim=post_segments, max_length=length_prior.max_length)


# %% [markdown]
#  ## Using an Empirical Prior

# %%
length_observations = sum([[length for length in np.diff(t_seg)]
                           for (_piece, _interpret, _tempo, t_seg, _dyn, _dyn_seg) in full_data],
                          start=[])


# %%
length_prior = length_priors.EmpiricalLengthPrior(length_observations, len(sample_data), max_length=30)

post_boundaries, post_segments = dc.run_alpha_beta(sample_data, arc_prior, length_prior, return_2d=True)
# _ = segment_viz.plot_segment_with_signal(post_marginals=post_boundaries, data=sampleData, data_time=sampleData)
# _ = segment_viz.plot_segment_beams(post2_bidim=post_segments, length_prior=length_prior)


# %%
_ = segment_viz.plot_segment_with_signal(post_marginals=post_boundaries,
                                         data=sample_data, data_time=list(range(len(sample_data))), smoothing=1)
_ = segment_viz.plot_segment_beams(post2_bidim=post_segments, max_length=length_prior.max_length)


# %%
times = np.cumsum(1/(tempo/60))+2
df = pd.DataFrame.from_dict({'time': times, 'p_boundary': post_boundaries})
df.to_csv("boundaries_post.csv")


# %% [markdown]
#  ## Using a Uniform Prior

# %%

length_prior4 = length_priors.EmpiricalLengthPrior(range(5, 30), len(sample_data), max_length=30)

post_boundaries4, post_segments4 = dc.run_alpha_beta(sample_data, arc_prior, length_prior4, return_2d=True)
# _ = segment_viz.plot_segment_with_signal(post_marginals=post_boundaries, data=sampleData, data_time=sampleData)
# _ = segment_viz.plot_segment_beams(post2_bidim=post_segments, length_prior=length_prior)


# %%
_ = segment_viz.plot_segment_with_signal(post_marginals=post_boundaries4,
                                         data=sample_data, data_time=list(range(len(sample_data))), smoothing=1)
_ = segment_viz.plot_segment_beams(post2_bidim=post_segments4, max_length=length_prior4.max_length)
