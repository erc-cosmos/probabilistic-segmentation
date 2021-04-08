#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from readers import *

#mazurka, tempos = list(readAllMazurkaTimings())[1]
#with open("2020-03-12_EC_Chopin_Ballade_N2_Take_2_tempo_man.csv") as csvFile:

files = ['beat_dyn/M06-1beat_dynNORM.csv','beat_dyn/M06-2beat_dynNORM.csv','beat_dyn/M33-4beat_dynNORM.csv','beat_dyn/M59-2beat_dynNORM.csv']

for f in files:
    data = readMazurkaData(f)
    print(f)
    for (i,(interpret,series)) in enumerate(data):
        if i<=10:
            print(str(i)+": "+interpret)
            plt.plot(series)
            plt.show()

print("Done")
