# %%
from bayes_arcs import segment_viz
import dynamic_computation as dc
from IPython import get_ipython
import length_priors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

get_ipython().run_line_magic('matplotlib', 'widget')


# %%
in_data = pd.read_csv(
    '/Users/guichaoua 1/Nextcloud/Cardiac_response_to_live_music/Updated features/full_ARIs/2018_05_22_P1.csv')


# %%
plt.plot(in_data.time, in_data.ARI)
plt.show()


# %% [markdown]
#  # First try

# %%
length_prior = length_priors.EmpiricalLengthPrior(range(5, 120), len(in_data.ARI))


# %%

length_prior = length_priors.GeometricLengthPrior(len(in_data.ARI), 0.8, 60, 5)


# %%
est_noise = np.std(np.diff(in_data.ARI))/2


# %%
arc_prior = {
    # Gaussian priors on the parameters of ax^2 + bx + c
    'aMean': 0,
    'aStd': 00,
    'bMean': 00,
    'bStd': 00,
    'cMean': 330,
    'cStd': 15,
    'noiseStd': est_noise
}


# %%
post, post_bidim = dc.run_alpha_beta(in_data.ARI, arc_prior, length_prior, return_2d=True)


# %%
f = segment_viz.plot_segment_with_signal(post, in_data.ARI, in_data.time,
                                         data_color='y', boundaries=in_data.time[post > .5])


# %% [markdown]
#  Let's plot only the strong boundaries

# %%
fig = plt.figure(figsize=(12, 8), dpi=100)

plt.plot(in_data.time[:len(post)], in_data.ARI[:len(post)], 'r')
# plt.xlim((0,500))
ax2 = plt.twinx()

high_post = post > .5
ax2.vlines(in_data.time[high_post], 0, 1)
plt.show()


# %%


# %% [markdown]
#  # Try 2D plots

# %%
smoothed = signal.convolve2d(post_bidim, np.ones((5, 5)), mode="same")
for i in range(np.shape(smoothed)[0]):
    smoothed[i, i] = 1


# %%
fig = plt.figure(figsize=(12, 8), dpi=100)

plt.imshow(smoothed)
plt.colorbar()

plt.show()


# %% [markdown]
#  # Do it again with other settings

# %% [markdown]
#  We use a Uniform Prior $U([10,90])$

# %%
length_prior = length_priors.EmpiricalLengthPrior(range(10, 90), len(in_data.ARI))


# %%
est_noise = np.std(np.diff(in_data.ARI)/2)


# %%
arc_prior = {
    # Gaussian priors on the parameters of ax^2 + bx + c
    'aMean': 0,
    'aStd': 00,
    'bMean': 00,
    'bStd': 00,
    'cMean': 330,
    'cStd': 15,
    'noiseStd': est_noise
}


# %%
post2, post2_bidim = dc.run_alpha_beta(in_data.ARI, arc_prior, length_prior, return_2d=True)


# %%
fig = plt.figure(figsize=(12, 8), dpi=100)

plt.plot(in_data.time[:len(post)], in_data.ARI[:len(post)], 'r')
# plt.xlim((500,1000))
ax2 = plt.twinx()

# ax2.plot(in_data.time[:len(post)],post)
post2_smoothed = np.convolve([1, 1, 1, 1, 1], post2, mode='same')

ax2.plot(in_data.time, post2_smoothed)
plt.show()


# %% [markdown]
#  Let's plot only the strong boundaries

# %%
fig = plt.figure(figsize=(12, 8), dpi=100)

plt.plot(in_data.time[:len(post)], in_data.ARI[:len(post)], 'r')
# plt.xlim((0,500))
ax2 = plt.twinx()

high_post = post2_smoothed > .5
ax2.vlines(in_data.time[high_post], 0, 1)
plt.show()


# %%
sum(post2)


# %%
sum(post)


# %% [markdown]
#  # Try 2D plots

# %%


# %%
segment_viz.plot_segment_matrix(post2_bidim, smoothing=5)


# %%
smoothed2 = signal.convolve2d(post2_bidim, np.ones((5, 5)), mode="same")
for i in range(np.shape(smoothed2)[0]):
    smoothed2[i, i] = 1


# %%
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

im = ax.imshow(smoothed2, cmap=plt.get_cmap('hot'))
cbar = fig.colorbar(im)
cbar.set_label("Likelihood")
ax.set_ylabel('Start')
ax.set_xlabel('End')
ax.set_title('Posterior likelihood of segment')


fig.show()


# %%
viz = segment_viz.plot_segment_raindrop(post2_bidim, length_prior.max_length)


# %%
viz2 = segment_viz.plot_segment_beams(post2_bidim, length_prior.max_length)


# %%
linear_viz = np.zeros((np.shape(post2_bidim)[0], length_prior.max_length))
for i in range(np.shape(post2_bidim)[0]):
    for j in range(length_prior.max_length):
        if i+j < np.shape(post2_bidim)[0]:
            for k in range(j):
                linear_viz[i+k, j] = linear_viz[i+k, j] + post2_bidim[i, i+j]


# %%
fig = plt.figure(figsize=(16, 4), dpi=100)

plt.imshow(np.transpose(linear_viz), cmap=plt.get_cmap('hot'), origin='lower', aspect=4)
plt.colorbar()

plt.show()


# %%


# %%
linear_viz_point = np.zeros((np.shape(smoothed2)[0], length_prior.max_length))
for i in range(np.shape(smoothed2)[0]):
    for j in range(length_prior.max_length):
        if i+j < np.shape(smoothed2)[0]:
            for k in range(1):
                linear_viz_point[i+k, j] = linear_viz_point[i+k, j] + smoothed2[i, i+j]


# %%
fig = plt.figure(figsize=(16, 4), dpi=100)

plt.imshow(np.transpose(linear_viz_point), cmap=plt.get_cmap('hot'), origin='lower', aspect=4)
plt.colorbar()

plt.show()


# %%
prior_marginals, prior_bidim = dc.run_alpha_beta([0 for _ in in_data.ARI], arc_prior, length_prior, return_2d=True)


# %%
_ = segment_viz.plot_segment_with_signal(prior_marginals, [0 for _ in in_data.ARI], in_data.time, smoothing=1)


# %%
geo_prior = length_priors.GeometricLengthPrior(len(in_data.ARI), .8, 60, 5)
geo_prior_marginals, geo_prior_bidim = dc.run_alpha_beta([0 for _ in in_data.ARI], arc_prior, geo_prior, return_2d=True)


# %%
_ = segment_viz.plot_segment_with_signal(geo_prior_marginals, [0 for _ in in_data.ARI], in_data.time, smoothing=1)


# %%
uni_prior = length_priors.EmpiricalLengthPrior(list(range(10, 15)), len(in_data.ARI))
uni_prior_marginals, uni_prior_bidim = dc.run_alpha_beta([0 for _ in in_data.ARI], arc_prior, uni_prior, return_2d=True)


# %%
_ = segment_viz.plot_segment_with_signal(uni_prior_marginals, [0 for _ in in_data.ARI], in_data.time, smoothing=5)


# %%
_ = segment_viz.plot_segment_raindrop(prior_bidim, max_length=120)
_ = segment_viz.plot_segment_raindrop(geo_prior_bidim, max_length=120)
_ = segment_viz.plot_segment_raindrop(uni_prior_bidim, max_length=120)


# %%
print(geo_prior.distrib.get(15)**2)
print(geo_prior.distrib.get(10)*geo_prior.distrib.get(20))


# %%
geo_prior = length_priors.GeometricLengthPrior(30, 0.8, 18, 11)
geo_prior_marginals, geo_prior_bidim = dc.run_alpha_beta([0 for _ in range(30)], arc_prior, geo_prior, return_2d=True)


# %%
_ = segment_viz.plot_segment_with_signal(geo_prior_marginals, [0 for _ in range(30)], list(range(30)), smoothing=1)


# %%
print(geo_prior_marginals)


# %%
