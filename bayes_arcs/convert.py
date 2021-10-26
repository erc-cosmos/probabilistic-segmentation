"""Script to convert data from the MazurkaBL format to the CosmoNote format."""

import readers
import os
import csv

tempoData = readers.readAllMazurkaTimings()
dynData = readers.readAllMazurkaData()
timingsData = readers.readAllMazurkaData(dirpath="data/beat_time")


data = []
for (piece, it) in tempoData:
    piece = piece[11:15]
    for (interpret, tempo) in it:
        for (piece2, it2) in dynData:
            for (interpret2, dyn) in it2:
                if interpret == interpret2:
                    for (piece3, it3) in timingsData:
                        for (interpret3, tim) in it3:
                            if interpret == interpret3:
                                data.append((piece, interpret, tim, [float('NaN'), *tempo], dyn))


for (piece, performer, timing, tempo, dyn) in data:
    fileBase = f"Chopin_Mazurka-{piece}_{performer}"
    os.makedirs(os.path.join("CosmoNoteData", fileBase), exist_ok=True)
    with open(os.path.join("CosmoNoteData", fileBase, fileBase+"_data.csv"), 'w') as metaFile:
        csvWriter = csv.writer(metaFile)
        csvWriter.writerow(["File name", "Composer", "Piece", "Performer"])
        csvWriter.writerow([fileBase, "Frederic Chopin", f"Mazurka {piece}", performer])
    with open(os.path.join("CosmoNoteData", fileBase, fileBase+"_tempo.csv"), 'w') as tempoFile:
        csvWriter = csv.writer(tempoFile)
        csvWriter.writerow(["Time", "Tempo"])
        csvWriter.writerows(zip(timing, tempo))
    with open(os.path.join("CosmoNoteData", fileBase, fileBase+"_loudness.csv"), 'w') as dynFile:
        csvWriter = csv.writer(dynFile)
        csvWriter.writerow(["Time", "Loudness"])
        csvWriter.writerows(zip(timing, dyn))
