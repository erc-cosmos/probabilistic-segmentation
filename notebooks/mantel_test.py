# %%
import matplotlib.pyplot as plt
import os
import mantel
import numpy as np
import itertools as itt


# %%
path1 = "/Users/guichaoua 1/Nextcloud/Workspace/ArcV2/other/Distance Matrix Rubinstein/Distance_Matrix_06-2.npy"
path2 = "/Users/guichaoua 1/Nextcloud/Workspace/ArcV2/other/Distance Matrix Rubinstein/Distance_Matrix_07-3.npy"


# %%

folder = "other/Distance Matrix Rubinstein"
paths = [os.path.join(folder, f) for f in sorted(os.listdir(folder))]

# %%
corr = np.ones((len(paths), len(paths)))
ps = np.zeros((len(paths), len(paths)))
for (i, a), (j, b) in itt.permutations(enumerate(paths), 2):
    mat1 = np.load(a)
    mat2 = np.load(b)

    r, p, z = mantel.test(mat1, mat2, perms=1000)
    corr[i, j] = r
    corr[j, i] = r

    ps[i, j] = p
    ps[j, i] = p
print(corr)
print(ps)
# %%
plt.imshow(corr)
plt.colorbar()
# %%
plt.imshow(np.log(ps))
plt.colorbar()


# %%
