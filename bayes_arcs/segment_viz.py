"""Visualisations for segmentations."""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.signal


def plot_segment_matrix(data, smoothing):
    """Display a matrix representation of the segment likelihoods."""
    # Sum the values in each region
    smoothed = sp.signal.convolve2d(data, np.ones((smoothing, smoothing)), mode="same")
    # Force the diagonal
    for i in range(np.shape(smoothed)[0]):
        smoothed[i, i] = 1
    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    im = ax.imshow(smoothed, cmap=plt.get_cmap('hot'))
    cbar = fig.colorbar(im)
    cbar.set_label("Likelihood")
    ax.set_ylabel('Start')
    ax.set_xlabel('End')
    ax.set_title('Posterior likelihood of segment')

    fig.show()


def plot_segment_raindrop(data, max_length):
    """Display a raindrop type graph for segmentation."""
    linear_viz_point = np.zeros((np.shape(data)[0], max_length))
    for i in range(np.shape(data)[0]):
        for j in range(max_length):
            if i+j < np.shape(data)[0]:
                linear_viz_point[i, j] = linear_viz_point[i, j] + data[i, i+j]
    fig, ax = plt.subplots(figsize=(16, 4), dpi=100)

    im = ax.imshow(np.transpose(linear_viz_point), cmap=plt.get_cmap('hot'), origin='lower', aspect=4)
    cbar = fig.colorbar(im)
    cbar.set_label("Likelihood")
    ax.set_ylabel('Duration')
    ax.set_xlabel('Start')
    ax.set_title('Posterior likelihood of segment')

    fig.show()
    return fig


def plot_segment_beams(post2_bidim, max_length):
    """Display a beams-like graph for segmentation."""
    linear_viz = np.zeros((np.shape(post2_bidim)[0], max_length))
    for i in range(np.shape(post2_bidim)[0]):
        for j in range(max_length):
            if i+j < np.shape(post2_bidim)[0]:
                for k in range(j):
                    linear_viz[i+k, j] = linear_viz[i+k, j] + post2_bidim[i, i+j]
    fig, ax = plt.subplots(figsize=(16, 4), dpi=100)

    im = ax.imshow(np.transpose(linear_viz), cmap=plt.get_cmap('hot'), origin='lower', aspect=4)
    cbar = fig.colorbar(im)
    cbar.set_label("Likelihood")
    ax.set_ylabel('Segment Duration')
    ax.set_xlabel('Point position')
    ax.set_title('Posterior likelihood of point within segment')

    fig.show()
    return fig


def plot_segment_with_signal(post_marginals, data, data_time, boundaries=None, *,
                             data_color='r', post_color='b', bound_color='k',
                             smoothing=5, show=True, input_label='Input data', time_label='Beat'):
    """Display boundary marginals with the input data."""
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=100)

    ax1.plot(data_time, data, color=data_color)
    ax2 = plt.twinx()

    if boundaries is not None:
        ax2.vlines(boundaries, ymin=0, ymax=1, color=bound_color)

    post2_smoothed = np.convolve(np.ones(smoothing), post_marginals, mode='same')

    ax2.plot(data_time, post2_smoothed, color=post_color)

    ax1.set_ylabel(input_label)
    ax2.set_xlabel(time_label)
    ax2.set_ylabel('Likelihood')
    ax2.set_ylim((0, 1))
    ax2.set_title('Posterior likelihood of boundaries')

    if show:
        fig.show()
    return fig
