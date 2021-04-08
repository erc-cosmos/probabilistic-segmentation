
import csv
import matplotlib.pyplot as plt
import numpy as np

from singleArc import *
from readers import *

fullData2 = readAllMazurkaTimingsAndSeg(timingPath='data/beat_time',segPath='data/deaf_structure_tempo')
fullData = readAllMazurkaDataAndSeg(timingPath="data/beat_dyn",segPath="data/deaf_structure_loudness")

#Minirun for debugging
fullData = fullData[-20:]
# print(fullData)

#keyList = [piece+"//"+pid for piece,pid,tim,seg in fullData]
_,_,dataList,segList = zip(*fullData)

target = dataList
seg = segList
ML = []
lengths = []
err = []

for t,s in zip(target,seg):
    y = []
    s = [-1,*s]
    for (sscurr,ssnext) in zip(s,s[1:]):
        data = normalizeX(t[(sscurr+1):ssnext+1])
        ML.append(arcML(data))
        lengths.append(ssnext-sscurr)
        y.extend([ML[-1][0]*x*x + ML[-1][1]*x + ML[-1][2] for (x,_) in data])
    err.extend(t-y)
    # plt.figure()
    # plt.plot(t)
    # plt.plot(y)
    # plt.vlines(s, ymin=np.min(t), ymax=np.max(t), colors="r")
    # plt.show()

a,b,c = zip(*[(MLi[0],MLi[1],MLi[2]) for MLi in ML])
print(np.mean(a))
print(np.std(a))
print(np.mean(b))
print(np.std(b))
print(np.mean(c))
print(np.std(c))
print(np.mean(lengths))
print(np.std(lengths))
print(np.mean(err))
print(np.std(err))
plt.figure()
plt.hist(lengths)
plt.show()