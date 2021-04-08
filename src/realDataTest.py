#######################@
# Simlpe test with real data
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import csv

from dynamicComputationMultidim import *
from lengthPrior import *
from readers import *

# Set for tempo
arcPriorTempo = { # Set after first 10 each of M06-1 and M06-2
    # Gaussian priors on the parameters of ax^2 + bx + c
    # TODO: Put in reasonable values
    'aMean': -181,
    'aStd': 93,
    'bMean': 159,
    'bStd': 106,
    'cMean': 107,
    'cStd': 31,
    'noiseStd': 18.1
}
# lengthPriorParams = {
#     'mean':14.7,
#     'stddev':5.95,
#     'maxLength':30
# }

# Set for loudness
arcPriorLoud = { # Set after first 10 each of M06-1 and M06-2
    # Gaussian priors on the parameters of ax^2 + bx + c
    'aMean': -0.73,
    'aStd': 0.55,
    'bMean': 0.68,
    'bStd': 0.60,
    'cMean': 0.41,
    'cStd': 0.19,
    'noiseStd': 0.039
}
lengthPriorParams = {
    'mean':11.8,
    'stddev':5.53,
    'maxLength':30
}

# Higher noise parameter for note-level
# arcPrior["noiseStd"] = 40


def legacyLoad():
    with open("2020-03-12_EC_Chopin_Ballade_N2_Take_2_tempo_man.csv") as csvFile:
    #with open("Mazurka-test.csv") as csvFile:
    #with open("iTempoBackward.csv") as csvFile:
        csvReader = csv.reader(csvFile)
        next(csvReader)#Skip header
        # tatums,tempos = zip(*[(float(row[0])/1000,80*float(row[2])) for row in csvReader])
        tatums,tempos = zip(*((i,row[0]) for i,row in enumerate(csvReader)))
        y = np.array(tempos,'float128')
        y = np.divide(60,np.diff(y,axis=0))
    return tatums, y

# tatums, y = legacyLoad()

timingsData = readAllMazurkaTimingsAndSeg(timingPath="data/beat_time",segPath="data/deaf_structure_tempo")
dynData = readAllMazurkaDataAndSeg(timingPath="data/beat_dyn", segPath="data/deaf_structure_loudness")


arcPrior = [arcPriorTempo,arcPriorLoud]
sampleData = None

for (piece, interpret, tempo, tempoSeg) in timingsData:
    for (piece2, interpret2, dyn, dynSeg) in dynData:
        if interpret == interpret2:
            piece_formatted = piece[16:20]
            print(piece_formatted, interpret)

            sampleData = list(zip(tempo,dyn[1:]))
            segs = (tempoSeg, dynSeg)

            tatums = list(range(len(sampleData)))
            # tatums,idx = np.unique(tatums[1:],return_index=True)
            # sampleData = y[idx]

            # idx = sampleData<300
            # sampleData = sampleData[idx]
            # tatums = tatums[idx]

            # sampleData = list(zip(sampleData,sampleData))
            # tatums=tatums[:300]
            # sampleData=sampleData[:300]

            lengthPrior = NormalLengthPrior(lengthPriorParams['mean'],lengthPriorParams['stddev'],range(len(sampleData)),lengthPriorParams['maxLength'])

            posteriorMarginals = runAlphaBeta(sampleData,arcPrior,lengthPrior)


            fig, ax1 = plt.subplots()
            tempo, dyn = zip(*sampleData)
            ax1.plot(tatums,tempo, color = "r") # Tempo input data
            plt.ylim(0,300)
            #TODO: second axis
            ax2 = ax1.twinx()

            ax2.plot(tatums,posteriorMarginals[1:], 'k') # Posterior Marginals
            plt.ylim(0,1)
            ax2.plot(tatums,dyn, color = "b") # Dyn input data
            plt.vlines(segs[0], ymin=0, ymax=1, colors="r", linestyle='dotted') # Tempo seg
            plt.vlines(segs[1], ymin=0, ymax=1, colors="b", linestyle='dotted') # Dyn seg
            # plt.show()

            with open(f"output/{piece_formatted}_{interpret}_pm.csv",'w') as outfile:
                csvWriter = csv.writer(outfile)
                csvWriter.writerow(["Beat count", "Posterior Marginal"])
                csvWriter.writerows(enumerate(posteriorMarginals))