#!/usr/bin/python3

from readers import *
from scoring import *
import numpy as np

timingsData = readAllMazurkaTimingsAndSeg()
dynData = readAllMazurkaDataAndSeg()

scores = []
for (piece, interpret, tempo, tempoSeg) in timingsData:
    for (piece2, interpret2, dyn, dynSeg) in dynData:
        if interpret == interpret2 :
            dynSeg = [x-1 for x in dynSeg]
            scores.append( (piece, interpret, frpMeasures(dynSeg, tempoSeg, tolerance=3)) )

# print(scores)
print(np.mean([f for (piece,intepret,(f,r,p)) in scores]))
print(np.mean([r for (piece,intepret,(f,r,p)) in scores]))
print(np.mean([p for (piece,intepret,(f,r,p)) in scores]))

scores_interPerf = []
for (piece, interpret, tempo, tempoSeg) in dynData:
    for (piece2, interpret2, tempo2, tempoSeg2) in timingsData:
        if piece[10:14] == piece2[11:15] and interpret != interpret2:
            scores_interPerf.append( (piece,interpret,interpret2, frpMeasures(tempoSeg,tempoSeg2, tolerance=3)))

print(np.mean([f for (piece,intepret,_,(f,r,p)) in scores_interPerf]))
print(np.mean([r for (piece,intepret,_,(f,r,p)) in scores_interPerf]))
print(np.mean([p for (piece,intepret,_,(f,r,p)) in scores_interPerf]))