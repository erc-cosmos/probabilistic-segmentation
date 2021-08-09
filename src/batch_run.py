"""Batch run on a whole dataset."""
import os.path

import dynamicComputation as dc
from lengthPriors import NormalLengthPrior
from readers import readAllMazurkaData
from defaultPriors import arcPriorLoud, lengthPriorParamsLoud
import writers


def trim_PID(pid):
    """Trim a performer ID to the relevant part."""
    return pid.split('-')[0]


def batchRun(full_data, arc_prior, length_prior_params, output_dir, piece_name_finder):
    """Run a Bayesian arc estimation on many series.

    full_data -- collection of collection of performances to analyze
    arc_prior -- prior on arc shapes
    length_prior_params -- parameters of the prior on arc lengths
    output_dir -- destination directory for the posterior marginals
    piece_name_finder -- callable to extract a piece ID from a file name
    """
    for (piece, performances) in full_data:
        for (pid, data) in performances:
            pid = trim_PID(pid)
            piece_formatted = piece_name_finder(piece)
            print(piece_formatted, pid)

            lengthPrior = NormalLengthPrior(length_prior_params['mean'], length_prior_params['stddev'],
                                            range(len(data)), length_prior_params['maxLength'])

            posteriorMarginals = dc.runAlphaBeta(data, arc_prior, lengthPrior)

            writers.writeMarginals(os.path.join(output_dir, f"{piece_formatted}_{pid}_pm.csv"), posteriorMarginals)


if __name__ == "__main__":
    # fullData = readAllMazurkaTimings("data/beat_time")
    # arcPrior = arcPriorTempo
    # lengthPriorParams = lengthPriorParamsTempo
    # outputDir = os.path.join('output','tempoBased')

    # batchRun(fullData,arcPrior,lengthPriorParams,outputDir,pieceNameFinder=lambda f:f[16:20])

    fullData = list(readAllMazurkaData("data/beat_dyn"))
    arcPrior = arcPriorLoud
    lengthPriorParams = lengthPriorParamsLoud
    outputDir = os.path.join('output', 'loudnessBased')

    batchRun(fullData, arcPrior, lengthPriorParams, outputDir, piece_name_finder=lambda f: f[15:19])
