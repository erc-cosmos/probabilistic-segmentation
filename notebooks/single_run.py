# %%
import importlib

from bayes_arcs import default_priors
from bayes_arcs import dynamic_computation as dc
from bayes_arcs import length_priors
from bayes_arcs import readers
from bayes_arcs import segment_viz
import numpy as np

# %%
from_tempo = True  # Set processing to tempo (True) or loudness (False)

# %%
if from_tempo:
    data_all_perf = readers.read_cosmo_piece(piece_folder="data/tempo_autocorrected/06-2", data_type='mixed')
else:
    data_all_perf = readers.read_mazurka_data(filename="data/beat_dyn/M06-2beat_dynNORM.csv")
data = [perf for pid, perf in data_all_perf if pid[:7] == "pid1263"][0]
# 9150 = Schoonderwoerd
# 1263 = Csalog
# %%
if from_tempo:
    (_, time), (_, tempo) = data.items()
    tempo = np.array(tempo)
else:
    tempo = np.array(data)
time = list(range(len(tempo)))

# %%
if from_tempo:
    arc_prior = default_priors.arc_prior_tempo
else:
    arc_prior = default_priors.arc_prior_loud
length_prior_params = default_priors.length_prior_params_tempo
length_prior = length_priors.NormalLengthPrior(length_prior_params['mean'], length_prior_params['stddev'], range(
    len(tempo)), length_prior_params['maxLength'])

# %%
if from_tempo:
    input_label = "Tempo (bpm)"
else:
    input_label = "Loudness (normalised)"

# %%
post_boundaries, post_segments = dc.compute_both_posteriors(tempo, arc_prior, length_prior)

# %%
importlib.reload(segment_viz)

# %%
fig = segment_viz.plot_segment_with_signal(post_marginals=post_boundaries, data=tempo, data_time=range(
    len(tempo)), smoothing=[1, 5], input_label=input_label, legend=False)

# %%
fig = segment_viz.plot_segment_with_signal(post_marginals=post_boundaries,
                                           data=tempo, data_time=time, smoothing=[1, 5], legend=False)

# %%
fig = segment_viz.plot_segment_matrix(post_segments, smoothing=1)
# %%
fig = segment_viz.plot_segment_beams(post_segments, length_prior_params['maxLength'])

# %%
fig = segment_viz.plot_segment_raindrop(post_segments, length_prior_params['maxLength'])


# %%
fig = segment_viz.plot_segment_with_signal(post_marginals=post_boundaries, figsize=(10, 5),
                                           data=tempo, data_time=range(len(tempo)), smoothing=[1, 5],
                                           input_label=input_label, legend=False)

# Structure for M24-3
# segment_viz.add_structure(fig, len(tempo), structure=['A', 'A', 'B',
#                           'A', 'B', 'A', 'C'], segment_length_in_bars=12, subsegment_length=4)
# Structure for M6-02
segment_viz.add_structure(fig, len(tempo), structure=['I', 'A', 'A', 'B',
                          'A', 'B', 'A', 'C', "C'", 'I', 'A', 'A'], segment_length_in_bars=8, subsegment_length=4)

# %%
