#!/usr/bin/python3

from readers import *
import matplotlib.pyplot as plt
import numpy as np

from singleArc import *

timingsData = readAllMazurkaTimingsAndSeg(timingPath="data/beat_time",segPath="data/deaf_structure_tempo")
dynData = readAllMazurkaDataAndSeg(timingPath="data/beat_dyn", segPath="data/deaf_structure_loudness")

scores = []
for (piece, interpret, tempo, tempoSeg) in timingsData:
    for (piece2, interpret2, dyn, dynSeg) in dynData:
        if interpret == interpret2 :
            print(interpret)
            tempoY = []
            dynY = []
            tempoML = []
            dynML = []
            tempoLengths = []
            dynLengths = []
            for (sscurr,ssnext) in zip([0,*tempoSeg],tempoSeg[:]):
                data = normalizeX(tempo[(sscurr+1):ssnext+1])
                tempoML.append(arcML(data))
                tempoLengths.append(ssnext-sscurr)
                tempoY.extend([tempoML[-1][0]*x*x + tempoML[-1][1]*x + tempoML[-1][2] for (x,_) in data])
            for (sscurr,ssnext) in zip([0, *dynSeg],dynSeg[:]):
                data = normalizeX(dyn[(sscurr+1):ssnext+1])
                dynML.append(arcML(data))
                dynLengths.append(ssnext-sscurr)
                dynY.extend([dynML[-1][0]*x*x + dynML[-1][1]*x + dynML[-1][2] for (x,_) in data])
            fig = plt.figure()
            plt.plot(tempo, color = "r")
            plt.plot(tempoY, color = "r", linestyle='dashed')
            plt.vlines(tempoSeg, ymin=np.min(tempo), ymax=np.max(tempo), colors="r", linestyle='dotted')
            ax = plt.twinx()
            ax.plot(dyn, color = "b")
            ax.plot(dynY, color = "b", linestyle='dashed')
            ax.vlines(dynSeg, ymin=np.min(dyn), ymax=np.max(dyn), colors="b", linestyle='dotted')
            plt.show()
            plt.close(fig)
        