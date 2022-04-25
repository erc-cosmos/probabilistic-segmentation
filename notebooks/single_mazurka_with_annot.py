# %%
from bayes_arcs import default_priors
from bayes_arcs import segment_viz, length_priors, readers, dynamic_computation as dc

# %%
cosmo = readers.read_all_cosmo_data()

# %%
cosmo_piece = cosmo[2]

# %%
in_loud = cosmo_piece.loudness
in_tempo = cosmo_piece.tempo
in_time = list(range(len(in_tempo)))
annot = cosmo_piece.annotations.audio[0]

# %%

# %%
in_data = list(zip(in_tempo[1:], in_loud[1:]))
arc_prior = [default_priors.arc_prior_tempo, default_priors.arc_prior_loud]

length_prior = length_priors.NormalLengthPrior(16, 8, range(
    len(in_data)), 30)

# %%
post_bound, post_segment = dc.compute_both_posteriors(in_data, arc_prior=arc_prior, length_prior=length_prior)

# %%
print(cosmo_piece.piece_id)
fig = segment_viz.plot_segment_with_signal(post_marginals=post_bound, boundaries=annot.boundaries,
                                           data_time=in_time[1:], smoothing=1, data=in_tempo[1:],
                                           input_label='Tempo', data_color='green')
fig.get_axes()[1].plot(in_loud[1:], 'red')
segment_viz.plot_segment_beams(post2_bidim=post_segment, max_length=length_prior.max_length, ratio=2.5)
