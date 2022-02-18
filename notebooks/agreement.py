# %% [markdown]
#  # Find the agreement between dynamics- and tempo-based segmentations

# %%
from bayes_arcs import readers
from bayes_arcs import scoring
import numpy as np


# %%
timings_data = readers.read_all_mazurka_timings_and_seg()
dyn_data = readers.read_all_mazurka_data_and_seg()

scores = []
for _, (piece, interpret, _tempo, tempo_seg, _dyn, dyn_seg) in readers.load_mazurka_dataset_with_annot().iterrows():
    dyn_seg = [x-1 for x in dyn_seg]
    scores.append((piece, interpret, scoring.frp_measures(dyn_seg, tempo_seg, tolerance=3)))


# %%
# print(scores)
print(np.mean([f for (piece, intepret, (f, r, p)) in scores]))
print(np.mean([r for (piece, intepret, (f, r, p)) in scores]))
print(np.mean([p for (piece, intepret, (f, r, p)) in scores]))


# %%
scores_inter_perf = []
for (piece, interpret, tempo, tempo_seg) in dyn_data:
    for (piece2, interpret2, tempo2, tempo_seg2) in timings_data:
        if piece[15:19] == piece2[16:20] and interpret != interpret2:
            scores_inter_perf.append((piece, interpret, interpret2,
                                      scoring.frp_measures(tempo_seg, tempo_seg2, tolerance=3)))


# %%
print(np.mean([f for (piece, intepret, _, (f, r, p)) in scores_inter_perf]))
print(np.mean([r for (piece, intepret, _, (f, r, p)) in scores_inter_perf]))
print(np.mean([p for (piece, intepret, _, (f, r, p)) in scores_inter_perf]))
