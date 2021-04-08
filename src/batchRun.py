#######################
# Batch run on whole dataset
import numpy as np
import scipy as sp
import csv
import os.path

from dynamicComputation import *
from lengthPrior import *
from readers import *
from defaultPriors import *
import writers

def trimPID(pid):
    return pid.split('-')[0]

def batchRun(fullData,arcPrior,lengthPriorParams,outputDir,pieceNameFinder):
    # os.makedirs(outputDir, exist_ok=True)
    for (piece, performances) in fullData:
        for (pid,data) in performances:
            pid = trimPID(pid)
            piece_formatted = pieceNameFinder(piece)
            print(piece_formatted, pid)

            lengthPrior = NormalLengthPrior(lengthPriorParams['mean'],lengthPriorParams['stddev'],range(len(data)),lengthPriorParams['maxLength'])

            posteriorMarginals = runAlphaBeta(data,arcPrior,lengthPrior)

            writers.writeMarginals(os.path.join(outputDir,f"{piece_formatted}_{pid}_pm.csv"), posteriorMarginals)
            # with open(os.path.join(outputDir,f"{piece_formatted}_{pid}_pm.csv"),'w') as outfile:
            #     csvWriter = csv.writer(outfile)
            #     csvWriter.writerow(["Beat count", "Posterior Marginal"])
            #     csvWriter.writerows(enumerate(posteriorMarginals))

if __name__=="__main__":
    # fullData = readAllMazurkaTimings("data/beat_time")
    # arcPrior = arcPriorTempo
    # lengthPriorParams = lengthPriorParamsTempo
    # outputDir = os.path.join('output','tempoBased')

    # batchRun(fullData,arcPrior,lengthPriorParams,outputDir,pieceNameFinder=lambda f:f[16:20])

    fullData = list(readAllMazurkaData("data/beat_dyn"))
    arcPrior = arcPriorLoud
    lengthPriorParams = lengthPriorParamsLoud
    outputDir = os.path.join('output','loudnessBased')

    batchRun(fullData,arcPrior,lengthPriorParams,outputDir,pieceNameFinder=lambda f:f[15:19])