"""Batch run on a whole dataset."""
import os.path
import os
import pandas as pd
import numpy as np

from pip import main

from . import dynamic_computation as dc
from . import writers
from .default_priors import arc_prior_loud, arc_prior_tempo
from .default_priors import length_prior_params_loud, length_prior_params_tempo
from .length_priors import NormalLengthPrior
from . import readers


def trim_pid(pid):
    """Trim a performer ID to the relevant part."""
    return pid.split('-')[0]


def batch_run(full_data, arc_prior, length_prior_params, output_dir, piece_name_finder):
    """Run a Bayesian arc estimation on many series.

    full_data -- collection of collection of performances to analyze
    arc_prior -- prior on arc shapes
    length_prior_params -- parameters of the prior on arc lengths
    output_dir -- destination directory for the posterior marginals
    piece_name_finder -- callable to extract a piece ID from a file name
    """
    for (piece, performances) in full_data:
        for (pid, data) in performances:
            if isinstance(data, tuple):
                _times, data = data
            pid = trim_pid(pid)
            piece_formatted = piece_name_finder(piece)
            print(piece_formatted, pid)

            length_prior = NormalLengthPrior(length_prior_params['mean'], length_prior_params['stddev'],
                                             range(len(data)), length_prior_params['maxLength'])

            posterior_marginals = dc.run_alpha_beta(data, arc_prior, length_prior)

            writers.write_marginals(os.path.join(output_dir, f"{piece_formatted}_{pid}_pm.csv"), posterior_marginals)


def run_mazurka_tempo_correction():
    import matplotlib.pyplot as plt

    def compare_before_after(before, after, dest_path):
        fig = plt.figure()
        plt.plot(60/readers.np.diff(before))
        plt.plot(after)
        plt.legend(["Before correction", "After correction"])
        plt.ylabel("Tempo (bpm)")
        plt.xlabel("Score time (beat)")
        plt.savefig(dest_path, dpi=300)
        plt.close(fig)

    full_data = list(readers.read_all_mazurka_data("data/beat_time"))

    def piece_name_finder(f):
        return f[16:20]

    output_dir = "data/tempo_autocorrected"

    for (piece, performances) in full_data:
        piece_dir = os.path.join(output_dir, piece_name_finder(piece))
        os.makedirs(piece_dir, exist_ok=True)
        for (pid, data) in performances:
            perf_path = os.path.join(piece_dir, f"{pid}.csv")
            new_tempo = readers.preprocess_tempo_outlier_correction(data)
            df = readers.pd.DataFrame(new_tempo, columns=["tempo"])
            df.to_csv(perf_path, index_label="count")
            if new_tempo != data:
                fig_path = os.path.join(piece_dir, f"{pid}.pdf")
                compare_before_after(data, new_tempo, fig_path)


def run_tempo():
    """Pre-filled batch run for mazurka Tempo."""
    full_data = list(readers.read_all_mazurka_timings("data/beat_time"))
    arc_prior = arc_prior_tempo
    length_prior_params = length_prior_params_tempo
    output_dir = os.path.join('output', 'tempoBased')

    batch_run(full_data, arc_prior, length_prior_params, output_dir, piece_name_finder=lambda f: f[16:20])


def run_loud():
    """Pre-filled batch run for mazurka Loudness."""
    full_data = list(readers.read_all_mazurka_data("data/beat_dyn"))
    arc_prior = arc_prior_loud
    length_prior_params = length_prior_params_loud
    output_dir = os.path.join('output', 'loudnessBased')
    batch_run(full_data, arc_prior, length_prior_params, output_dir, piece_name_finder=lambda f: f[15:19])


def run_m68_3_tempo():
    """Pre-filled batch run for Mazurka 68-3."""
    files = [(os.path.splitext(f)[0], os.path.join('data/M68-3', f)) for f in os.listdir('data/M68-3')]
    full_data = [("M68-3", [(name, (60/readers.np.diff(readers.read_cosmo_beats(f))
                             if name != 'Average'
                             else readers.np.array(readers.pd.read_csv(f, header=None)[0])))
                            for name, f in files])]
    arc_prior = arc_prior_tempo
    length_prior_params = length_prior_params_tempo
    output_dir = os.path.join('output', 'M68-3_tempoBased')

    batch_run(full_data, arc_prior, length_prior_params, output_dir, piece_name_finder=lambda f: f)


def run_tempo_autocorrected():
    """Pre-filled batch run for all Mazurka with the corrected tempo."""
    main_folder = "data/tempo_autocorrected"
    full_data = readers.read_cosmo_collection(main_folder, include_average=True)
    arc_prior = arc_prior_tempo
    length_prior_params = length_prior_params_tempo
    output_dir = os.path.join('output', 'tempo_autocorrected')

    batch_run(full_data, arc_prior, length_prior_params, output_dir, piece_name_finder=lambda f: f"Mazurka{f}")


def run_tempo_autocorrected_average_only():
    """Pre-filled batch run for all Mazurka with the corrected tempo."""
    main_folder = "data/tempo_autocorrected"
    full_data = readers.read_cosmo_collection(main_folder, include_average=True)
    full_data = [(piece, [perfs[-1]]) for piece, perfs in full_data]
    arc_prior = arc_prior_tempo
    length_prior_params = length_prior_params_tempo
    output_dir = os.path.join('output', 'tempo_autocorrected')

    batch_run(full_data, arc_prior, length_prior_params, output_dir, piece_name_finder=lambda f: f"Mazurka{f}")


def read_full_one_per_perf(main_folder):
    pieces = sorted([f for f in os.listdir(main_folder)
                     if os.path.isdir(os.path.join(main_folder, f))])
    print(pieces)
    full_data = []
    for piece in pieces:
        piece_folder = os.path.join(main_folder, piece)
        piece_data = []
        perfs = [f for f in os.listdir(piece_folder) if f.endswith('.csv')]
        for perf in perfs:
            data_path = os.path.join(piece_folder, perf)
            data = pd.read_csv(data_path)
            perf_id, _ = os.path.splitext(perf)
            piece_data.append((perf_id, np.array(data['tempo'])))
        full_data.append((piece, piece_data))
    return full_data


# if __name__ == "__main__":
#     fullData = list(readers.read_all_mazurka_timings("data/beat_time"))
#     arcPrior = arc_prior_tempo
#     lengthPriorParams = length_prior_params_tempo
#     outputDir = os.path.join('output', 'tempoBased')

#     batch_run(fullData, arcPrior, lengthPriorParams, outputDir, piece_name_finder=lambda f: f[16:20])

#     full_data = list(readers.read_all_mazurka_data("data/beat_dyn"))
#     arc_prior = arc_prior_loud
#     length_prior_params = length_prior_params_loud
#     output_dir = os.path.join('output', 'loudnessBased')

#     batch_run(full_data, arc_prior, length_prior_params, output_dir, piece_name_finder=lambda f: f[15:19])
