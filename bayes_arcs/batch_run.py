"""Batch run on a whole dataset."""
import os.path

from . import dynamic_computation as dc
from . import writers
from .default_priors import arc_prior_loud
from .default_priors import length_prior_params_loud
from .length_priors import NormalLengthPrior
from .readers import read_all_mazurka_data


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
            pid = trim_pid(pid)
            piece_formatted = piece_name_finder(piece)
            print(piece_formatted, pid)

            length_prior = NormalLengthPrior(length_prior_params['mean'], length_prior_params['stddev'],
                                             range(len(data)), length_prior_params['maxLength'])

            posterior_marginals = dc.run_alpha_beta(data, arc_prior, length_prior)

            writers.write_marginals(os.path.join(output_dir, f"{piece_formatted}_{pid}_pm.csv"), posterior_marginals)


if __name__ == "__main__":
    # fullData = readAllMazurkaTimings("data/beat_time")
    # arcPrior = arcPriorTempo
    # lengthPriorParams = lengthPriorParamsTempo
    # outputDir = os.path.join('output','tempoBased')

    # batchRun(fullData,arcPrior,lengthPriorParams,outputDir,pieceNameFinder=lambda f:f[16:20])

    full_data = list(read_all_mazurka_data("data/beat_dyn"))
    arc_prior = arc_prior_loud
    length_prior_params = length_prior_params_loud
    output_dir = os.path.join('output', 'loudnessBased')

    batch_run(full_data, arc_prior, length_prior_params, output_dir, piece_name_finder=lambda f: f[15:19])
