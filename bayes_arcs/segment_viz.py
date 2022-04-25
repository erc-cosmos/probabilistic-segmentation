"""Visualisations for segmentations."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from typing import Iterable, Union

plt.rcParams["font.family"] = "Times"


def plot_segment_matrix(data, smoothing: int):
    """Display a matrix representation of the segment credence."""
    # Sum the values in each region
    smoothed = signal.convolve2d(data, np.ones((smoothing, smoothing)), mode="same")
    # Force the diagonal
    for i in range(np.shape(smoothed)[0]):
        smoothed[i, i] = .1

    fig, ax = plt.subplots(figsize=(12, 8), dpi=100)

    im = ax.imshow(smoothed, cmap=plt.get_cmap('binary'))
    cbar = fig.colorbar(im)
    cbar.set_label("Credence")
    ax.set_ylabel('Start')
    ax.set_xlabel('End')
    ax.set_title('Posterior credence of segment')

    fig.show()


def plot_segment_raindrop(data, max_length, ratio=4, show=True, xticks=12):
    """Display a raindrop type graph for segmentation."""
    linear_viz_point = np.zeros((np.shape(data)[0], max_length))
    for i in range(np.shape(data)[0]):
        for j in range(max_length):
            if i+j < np.shape(data)[0]:
                linear_viz_point[i, j] = linear_viz_point[i, j] + data[i, i+j]
    height = 4
    fig, ax = plt.subplots(figsize=(ratio*height, height), dpi=100)

    im = ax.imshow(np.transpose(linear_viz_point), cmap=plt.get_cmap('binary'), origin='lower', aspect=ratio)
    cbar = fig.colorbar(im)
    cbar.set_label("Credence")
    ax.set_ylabel('Duration')
    x_max = np.shape(data)[0]
    ax.set_xticks([xticks * x for x in range(int(x_max/xticks) + 1)])
    ax.set_xlabel('Start')
    ax.set_title('Posterior credence of segment')
    if show:
        fig.show()
    return fig


def plot_segment_beams(post2_bidim, max_length, ratio=4, show=True, xticks=12):
    """Display a beams-like graph for segmentation."""
    linear_viz = np.zeros((np.shape(post2_bidim)[0], max_length))
    for i in range(np.shape(post2_bidim)[0]):
        for j in range(max_length):
            if i+j < np.shape(post2_bidim)[0]:
                for k in range(j):
                    linear_viz[i+k, j] = linear_viz[i+k, j] + post2_bidim[i, i+j]
    height = 4
    fig, ax = plt.subplots(figsize=(ratio*height, height), dpi=100)

    im = ax.imshow(np.transpose(linear_viz), cmap=plt.get_cmap('binary'), origin='lower', aspect=ratio)
    cbar = fig.colorbar(im)
    cbar.set_label("Credence")
    ax.set_ylabel('Segment Duration')
    ax.set_xlabel('Point position')
    x_max = np.shape(post2_bidim)[0]
    ax.set_xticks([xticks * x for x in range(int(x_max/xticks) + 1)])
    ax.set_title('Posterior credence of point within segment')
    if show:
        fig.show()
    return fig


def plot_segment_with_signal(post_marginals, data, data_time, boundaries=None, *,
                             data_color='r', post_color='b', bound_color='k',
                             smoothing: Union[int, Iterable[int]] = 1, show: bool = True,
                             input_label: str = 'Input data', time_label: str = 'Beat',
                             figsize=(10, 4), xticks=12, legend=True):
    """Display boundary marginals with the input data."""
    fig, ax1 = plt.subplots(figsize=figsize, dpi=100)

    line_data = ax1.plot(data_time, data, color=data_color)
    lines = line_data
    ax2 = plt.twinx()

    if boundaries is not None:
        ax2.vlines(boundaries, ymin=0, ymax=1, color=bound_color)

    if isinstance(smoothing, int):
        smoothing = [smoothing]

    line_styles = ['solid', 'dashed', 'dotted']
    for smooth, style in zip(smoothing, line_styles):
        post2_smoothed = np.convolve(np.ones(smooth), post_marginals, mode='same')
        line_proba = ax2.plot(data_time, post2_smoothed, color=post_color, linestyle=style)
        lines = lines + line_proba

    x_max = max(data_time)
    ax1.set_ylabel(input_label)
    ax1.set_xlabel(time_label)
    ax1.set_xlim(0, x_max)
    ax1.set_xticks([xticks * x for x in range(int(x_max/xticks) + 1)])
    ax2.set_ylabel('Credence')
    ax2.set_ylim((0, 1))
    ax2.set_title('Posterior credence of boundaries')
    labels = [input_label, 'Credence of boundary']
    if legend:
        ax2.legend(lines, labels)
    if show:
        fig.show()
    return fig


def plot_segment_with_multiple_signals(post_marginals, data_a, data_b, data_time, boundaries=None, *,
                                       data_color_a='r', data_color_b='g', post_color='b', bound_color='k',
                                       smoothing: int = 5, show: bool = True,
                                       input_label_a: str = 'Tempo', input_label_b: str = 'Loudness', time_label: str = 'Beat',
                                       figsize=(10, 4), xticks=12, legend=True):
    """Display boundary marginals with dual input data."""
    fig, ax1 = plt.subplots(figsize=figsize, dpi=100)

    line_data_a = ax1.plot(data_time, data_a, color=data_color_a)
    ax2 = plt.twinx()

    if boundaries is not None:
        ax2.vlines(boundaries, ymin=0, ymax=1, color=bound_color)

    line_data_b = ax2.plot(data_time, data_b, color=data_color_b)

    post2_smoothed = np.convolve(np.ones(smoothing), post_marginals, mode='same')

    line_proba = ax2.plot(data_time, post2_smoothed, color=post_color)

    x_max = max(data_time)
    ax1.set_ylabel(input_label_a)
    ax1.set_xlabel(time_label)
    ax1.set_xlim(0, x_max)
    ax1.set_xticks([xticks * x for x in range(int(x_max/xticks) + 1)])
    ax2.set_ylabel('Credence')
    ax2.set_ylim((0, 1))
    ax2.set_title('Posterior credence of boundaries')
    lines = line_data_a + line_data_b + line_proba
    labels = [input_label_a, input_label_b, 'Credence of boundary']
    if legend:
        ax2.legend(lines, labels)
    if show:
        fig.show()
    return fig


def add_structure(fig, length, structure, text_height=0.85, bar_length=3,
                  segment_length_in_bars=4, subsegment_length=4):
    import matplotlib.patches as patches
    plt.figure(fig)

    segment_length = bar_length*segment_length_in_bars
    for i in range(0, int(length/segment_length)+1):
        plt.axvline(x=segment_length*i, linewidth=2, zorder=-50)
        [plt.axvline(x=segment_length*i+bar_length*j, linewidth=0.1 if j % subsegment_length else .8, zorder=-50)
            for j in range(1, segment_length_in_bars)]

    plt.axvline(x=segment_length*(int(length/segment_length)+1), linewidth=0.8, zorder=-50)

    colormap = {'I': 'red', 'A': 'orange', 'B': 'green', 'C': 'purple'}
    color = [colormap[seg[0]] for seg in structure]

    bottom, top = fig.axes[0].get_ylim()
    for i in range(0, len(color)):
        left, width = (i*segment_length, segment_length)
        rect = patches.Rectangle((left, bottom), width, top-bottom, alpha=.2, facecolor=color[i], zorder=-100)
        fig.axes[0].add_patch(rect)
        plt.text(segment_length*(i+0.5), text_height, structure[i], size=25, horizontalalignment='center', zorder=-80)

    plt.show()
