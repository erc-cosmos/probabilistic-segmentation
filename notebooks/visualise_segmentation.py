# %%

from bayes_arcs import readers
from bayes_arcs import single_arc as sa
import matplotlib.pyplot as plt
import numpy as np


# %%
full_data2 = readers.read_all_mazurka_timings_and_seg(
    timing_path='data/beat_time', seg_path='data/deaf_structure_tempo')
full_data = readers.read_all_mazurka_data_and_seg(timing_path="data/beat_dyn", seg_path="data/deaf_structure_loudness")


# %%
# Minirun for debugging
full_data = full_data[-20:]


# %%
_, _, data_list, seg_list = zip(*full_data)

target = data_list
seg = seg_list
ML = []
lengths = []
err = []

for t, s in zip(target, seg):
    y = []
    s = [-1, *s]
    for (sscurr, ssnext) in zip(s, s[1:]):
        data = sa.normalize_x(t[(sscurr+1):ssnext+1])
        ML.append(sa.arc_max_likelihood(data))
        lengths.append(ssnext-sscurr)
        y.extend([ML[-1][0]*x*x + ML[-1][1]*x + ML[-1][2] for (x, _) in data])
    err.extend(t-y)
    plt.figure()
    plt.plot(t)
    plt.plot(y)
    plt.vlines(s, ymin=np.min(t), ymax=np.max(t), colors="r")
    plt.show()

a, b, c = zip(*[(MLi[0], MLi[1], MLi[2]) for MLi in ML])


# %%
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


# %%
plt.figure()
plt.hist(lengths, bins=20)
plt.show()


# %%
