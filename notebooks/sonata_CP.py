# %% [markdown]
#  # Experiments on a modified moonlight sonata (mov3)

# %%
import csv

import dynamic_computation as dc
import length_priors
import matplotlib.pyplot as plt
import scipy.interpolate as inter


# %%
with open('data/sonataCP/Loudness_Changepoints_beats.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    beats = [float(t) for t, *_ in csv_reader]
interpolate_target = beats


# %% [markdown]
#  ## Piecewise constant

# %%
with open('data/sonataCP/Loudness_Changepoints_loudness.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    smooth_loudness = [(float(line[0]), float(line[3])) for line in csv_reader]


# %%
with open('data/sonataCP/Loudness_Changepoints_velocity.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    velocities = [(float(t), int(v)) for t, v in csv_reader]
t_scaling = 101.887 / velocities[-1][0]
velocities = [(t*t_scaling, v) for t, v in velocities]


# %%
t_v, v_v = zip(*velocities)
fig, ax1 = plt.subplots()
ax1.plot(t_v, v_v)
ax2 = ax1.twinx()
ax2.plot(*zip(*smooth_loudness), 'y')
fig.show()


# %%
spline = inter.UnivariateSpline(*zip(*smooth_loudness), s=0)
loudness = spline(interpolate_target)


# %%
t_v, v_v = zip(*velocities)
fig, ax1 = plt.subplots()
ax1.plot(t_v, v_v)
ax2 = ax1.twinx()
ax2.plot(interpolate_target, loudness, 'y')
fig.show()


# %%
length_prior = length_priors.EmpiricalLengthPrior(list(range(1, 20)), data_length=len(loudness), max_length=20)


# %%
arc_prior_loud = {
    # Gaussian priors on the parameters of ax^2 + bx + c
    'aMean': 0,
    'aStd': 00,
    'bMean': 00,
    'bStd': 00,
    'cMean': 0.41,
    'cStd': 0.4,
    'noiseStd': 0.1
}


# %%
posteriors = dc.compute_boundary_posteriors(loudness, arc_prior_loud, length_prior)


# %%
t_v, v_v = zip(*velocities)
fig, ax1 = plt.subplots()
ax1.plot(t_v, v_v)
ax2 = ax1.twinx()
ax2.plot(interpolate_target, loudness, 'y')
ax2.plot(interpolate_target, posteriors, 'r')
fig.show()


# %% [markdown]
#  ## Piecewise constant (no velocity changes)

# %%
with open('data/sonataCP/Loudness_Changepoints_flat_loudness.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    smooth_loudness_flat = [(float(line[0]), float(line[3])) for line in csv_reader]


# %%
spline = inter.UnivariateSpline(*zip(*smooth_loudness_flat), s=0)
loudness_flat = spline(interpolate_target)


# %%
arc_prior_loud_flat = {
    # Gaussian priors on the parameters of ax^2 + bx + c
    'aMean': 0,
    'aStd': 00,
    'bMean': 00,
    'bStd': 00,
    'cMean': 0.41,
    'cStd': 0.4,
    'noiseStd': 0.1
}


# %%
posteriors_flat = dc.compute_boundary_posteriors(loudness_flat, arc_prior_loud_flat, length_prior)


# %%
fig, ax1 = plt.subplots()
ax1.plot(interpolate_target, loudness_flat, 'y')
ax2 = ax1.twinx()
ax2.plot(interpolate_target, posteriors_flat, 'r')
fig.show()


# %% [markdown]
#  ## Piecewise linear

# %%
with open('data/sonataCP/Loudness_Changepoints_slopes_loudness.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    smooth_loudness_flat = [(float(line[0]), float(line[3])) for line in csv_reader]


# %%
with open('data/sonataCP/Loudness_Changepoints_slopes_velocity.csv') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader)
    velocities = [(float(t), int(v)) for t, v in csv_reader]
t_scaling = 101.887 / velocities[-1][0]
velocities = [(t*t_scaling, v) for t, v in velocities]


# %%
spline = inter.UnivariateSpline(*zip(*smooth_loudness_flat), s=0)
loudness_flat = spline(interpolate_target)


# %%
arc_prior_loud_flat = {
    # Gaussian priors on the parameters of ax^2 + bx + c
    'aMean': 0,
    'aStd': 00,
    'bMean': 00,
    'bStd': .8,
    'cMean': 0.41,
    'cStd': 0.4,
    'noiseStd': 0.2
}


# %%
length_prior = length_priors.EmpiricalLengthPrior(
    [*range(1, 21), *range(5, 16), *range(8, 13)], data_length=len(loudness_flat), max_length=20)


# %%
posteriors_flat = dc.compute_boundary_posteriors(loudness_flat, arc_prior_loud_flat, length_prior)


# %%
fig, ax1 = plt.subplots()
ax1.plot(*zip(*velocities))
ax2 = ax1.twinx()
ax2.plot(interpolate_target, loudness_flat, 'y')
ax2.plot(interpolate_target, posteriors_flat, 'r')
fig.show()


# %%


# %%


# %%
