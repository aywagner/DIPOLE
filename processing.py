# This script generates the four datasets found in the manuscript.

# Brain
import numpy as np
import helper
data = np.load('data/brain1.npy')
data = helper.subsample(data, voxel_size = 5)
np.savetxt("data/brain.txt", data)

# Swiss roll with holes
from sklearn.datasets import make_swiss_roll
def remove_cylinder(X):
    Z = []
    for pt in X:
        if pt[0]**2 + (pt[1]-10)**2 >= 25:
            Z.append(pt)
    return np.asarray(Z)
data = make_swiss_roll(n_samples=3000, noise=0.0, random_state=None)[0]
data = remove_cylinder(data)
np.savetxt("data/swisshole.txt", data)

# Mammoth
import requests
url = 'https://raw.githubusercontent.com/PAIR-code/understanding-umap/master/raw_data/mammoth_3d.json'
r = requests.get(url)
data = np.asarray(r.json())
data = helper.subsample(data, voxel_size = 10)
np.savetxt("data/mammoth.txt", data)

# Stanford faces
# No pre-processing is performed on this dataset.