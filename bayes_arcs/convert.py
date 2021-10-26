"""Script to convert data from the MazurkaBL format to the CosmoNote format."""

import csv
import os

import readers

tempo_data = readers.read_all_mazurka_timings()
dyn_data = readers.read_all_mazurka_data()
timings_data = readers.read_all_mazurka_data(dirpath="data/beat_time")


data = []
for (piece, it) in tempo_data:
    piece = piece[11:15]
    for (interpret, tempo) in it:
        for (piece2, it2) in dyn_data:
            for (interpret2, dyn) in it2:
                if interpret == interpret2:
                    for (piece3, it3) in timings_data:
                        for (interpret3, tim) in it3:
                            if interpret == interpret3:
                                data.append((piece, interpret, tim, [float('NaN'), *tempo], dyn))


for (piece, performer, timing, tempo, dyn) in data:
    file_base = f"Chopin_Mazurka-{piece}_{performer}"
    os.makedirs(os.path.join("CosmoNoteData", file_base), exist_ok=True)
    with open(os.path.join("CosmoNoteData", file_base, file_base+"_data.csv"), 'w') as meta_file:
        csv_writer = csv.writer(meta_file)
        csv_writer.writerow(["File name", "Composer", "Piece", "Performer"])
        csv_writer.writerow([file_base, "Frederic Chopin", f"Mazurka {piece}", performer])
    with open(os.path.join("CosmoNoteData", file_base, file_base+"_tempo.csv"), 'w') as tempo_file:
        csv_writer = csv.writer(tempo_file)
        csv_writer.writerow(["Time", "Tempo"])
        csv_writer.writerows(zip(timing, tempo))
    with open(os.path.join("CosmoNoteData", file_base, file_base+"_loudness.csv"), 'w') as dyn_file:
        csv_writer = csv.writer(dyn_file)
        csv_writer.writerow(["Time", "Loudness"])
        csv_writer.writerows(zip(timing, dyn))
