# %%
import importlib
from bayes_arcs import default_priors
from bayes_arcs import dynamic_computation as dc
from bayes_arcs import segment_viz
from bayes_arcs import readers
from bayes_arcs import length_priors
import numpy as np

# %%
data_all_perf = readers.read_cosmo_piece(piece_folder="data/tempo_autocorrected/06-2", data_type='mixed')
data = [perf for pid, perf in data_all_perf if pid[:7] == "pid9150"][0]

# %%
(_, time), (_, tempo) = data.items()
tempo = np.array(tempo)

# %%
arc_prior = default_priors.arc_prior_tempo
length_prior_params = default_priors.length_prior_params_tempo
length_prior = length_priors.NormalLengthPrior(length_prior_params['mean'], length_prior_params['stddev'], range(
    len(tempo)), length_prior_params['maxLength'])


# %%
post_boundaries, post_segments = dc.run_alpha_beta(tempo, arc_prior, length_prior, return_2d=True)

# %%
fig = segment_viz.plot_segment_with_signal(post_marginals=post_boundaries,
                                           data=tempo, data_time=range(len(tempo)), smoothing=1, input_label="Tempo")

# %%
fig = segment_viz.plot_segment_with_signal(post_marginals=post_boundaries, data=tempo, data_time=time, smoothing=5)

# %%
fig = segment_viz.plot_segment_beams(post_segments, length_prior_params['maxLength'])

# %%
fig = segment_viz.plot_segment_raindrop(post_segments, length_prior_params['maxLength'])

# %%
importlib.reload(segment_viz)

# %%
fig = segment_viz.plot_segment_matrix(post_segments, smoothing=1)

# %%
