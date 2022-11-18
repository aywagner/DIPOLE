# This script generates the visualizations found in the manuscript.

import matplotlib.pyplot as plt
import numpy as np
import drmethods

# Mammoth
data = np.loadtxt('data/mammoth.txt')
col = data[:, 1]
embedding_dipole = drmethods.DIPOLE_mask_method(high_ptcloud=data,
                                                target_dim=2,
                                                lmr_edges=5,
                                                alpha=0.001, # TODO: 5 lmr and alpha 0.001?
                                                k=64,
                                                lr=1.0)
embedding_lmr = drmethods.LMR_method(high_ptcloud=data,
                                     target_dim=2,
                                     lmr_edges=10,
                                     lr=1.0)
embedding_tsne = drmethods.TSNE_method(high_ptcloud=data,
                                       target_dim=2,
                                       perplexity=30)
fig = plt.figure(figsize=(20,20))
ax_pts = fig.add_subplot(2,2,1, projection='3d')
ax_pts.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.5, c=col)
ax_pts.view_init(elev=100., azim=260)
ax_pts.set_title("Initial Point Cloud", fontsize = 30)
ax_dipole = fig.add_subplot(2,2,2)
ax_dipole.scatter(embedding_dipole[:, 0], embedding_dipole[:, 1], alpha=0.5, c=col)
ax_dipole.set_title("DIPOLE", fontsize = 30)
ax_LMR = fig.add_subplot(2,2,3)
ax_LMR.scatter(embedding_lmr[:, 0], embedding_lmr[:, 1], alpha=0.5, c=col)
ax_LMR.set_title("Local Metric Regularizer", fontsize = 30)
ax_TSNE = fig.add_subplot(2,2,4)
ax_TSNE.scatter(embedding_tsne[:, 0], embedding_tsne[:, 1], alpha=0.5, c=col)
ax_TSNE.set_title("t-SNE", fontsize = 30)
plt.savefig('figures/mammoth.png', format='png')
exit()

# Brain
data = np.loadtxt('data/brain.txt')
col = data[:, 2]
embedding_dipole = drmethods.DIPOLE_mask_method(high_ptcloud=data,
                                                target_dim=2,
                                                lmr_edges=5,
                                                alpha=0.001,
                                                k=64,
                                                lr=1.0)
embedding_lmr = drmethods.LMR_method(high_ptcloud=data,
                                     target_dim=2,  
                                     lmr_edges=10,
                                     lr=1.0)
embedding_umap = drmethods.umap_method(high_ptcloud=data,
                                       target_dim=2,
                                       n_neighbors=15,
                                       min_dist=0.1)
fig = plt.figure(figsize=(20,20))
ax_pts = fig.add_subplot(2,2,1, projection='3d')
ax_pts.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.5, c=col)
ax_pts.set_title("Initial Data", fontsize = 30)
ax_dipole = fig.add_subplot(2,2,2)
ax_dipole.scatter(embedding_dipole[:, 0], embedding_dipole[:, 1], alpha=0.5, c=col)
ax_dipole.set_title("DIPOLE", fontsize = 30)
ax_LMR = fig.add_subplot(2,2,3)
ax_LMR.scatter(embedding_lmr[:, 0], embedding_lmr[:, 1], alpha=0.5, c=col)
ax_LMR.set_title("Local Metric Regularizer", fontsize = 30)
ax_UMAP = fig.add_subplot(2,2,4)
ax_UMAP.scatter(embedding_umap[:, 0], embedding_umap[:, 1], alpha=0.5, c=col)
ax_UMAP.set_title("UMAP", fontsize = 30)
plt.savefig('figures/brain.png', format='png')


# Swiss roll with holes
data = np.loadtxt('data/swisshole.txt')
col = data[:, 2]
embedding_dipole = drmethods.DIPOLE_mask_method(high_ptcloud=data,
                                                target_dim=2,
                                                lmr_edges=3,
                                                alpha=0.1,
                                                k=64,
                                                lr=0.1,
                                                m=10)
embedding_lmr = drmethods.LMR_method(high_ptcloud=data,
                                     target_dim=2,
                                     lmr_edges=10,
                                     lr=1.0,
                                     m=10)
embedding_umap = drmethods.umap_method(high_ptcloud=data,
                                       target_dim=2,
                                       n_neighbors=15,
                                       min_dist=0.1)
fig = plt.figure(figsize=(20,20))
ax_pts = fig.add_subplot(2,2,1, projection='3d')
ax_pts.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.5, c=col)
ax_pts.set_title("Initial Point Cloud", fontsize = 30)
ax_dipole = fig.add_subplot(2,2,2)
ax_dipole.scatter(embedding_dipole[:, 0], embedding_dipole[:, 1], alpha=0.5, c=col)
ax_dipole.set_title("DIPOLE", fontsize = 30)
ax_LMR = fig.add_subplot(2,2,3)
ax_LMR.scatter(embedding_lmr[:, 0], embedding_lmr[:, 1], alpha=0.5, c=col)
ax_LMR.set_title("Local Metric Regularizer", fontsize = 30)
ax_UMAP = fig.add_subplot(2,2,4)
ax_UMAP.scatter(embedding_umap[:, 0], embedding_umap[:, 1], alpha=0.5, c=col)
ax_UMAP.set_title("UMAP", fontsize = 30)
plt.savefig('figures/swisshole.png', format='png')

# Stanford Faces dataset
from scipy.io import loadmat
mat = loadmat('data/face_data.mat')
data = mat['images'].T
def minmaxnormalize(x):
    x -= np.min(x)
    return x/np.max(x)
colors = np.ones((data.shape[0], 3)) * 0.5
colors[:, 0] = minmaxnormalize(mat['poses'][0, :])
colors[:, 2] = minmaxnormalize(mat['poses'][1, :])
embedding_isomap = drmethods.Isomap_method(high_ptcloud=data,
                                           target_dim=3)
embedding_dipole = drmethods.DIPOLE_mask_method(high_ptcloud=data,
                                                target_dim=3,
                                                lmr_edges=5,
                                                alpha=0.1,
                                                k=32,
                                                lr=1.0)
embedding_lmr = drmethods.LMR_method(high_ptcloud=data,
                                     target_dim=3,
                                     lmr_edges=10,
                                     lr=1.0)
embedding_umap = drmethods.umap_method(high_ptcloud=data,
                                       target_dim=3,
                                       n_neighbors=15,
                                       min_dist=0.1)
fig = plt.figure(figsize=(20,20))
ax_pts = fig.add_subplot(2,2,1, projection='3d')
ax_pts.scatter(embedding_isomap[:, 0], embedding_isomap[:, 1], embedding_isomap[:, 2], alpha=0.5, c=colors)
ax_pts.set_title("Isomap", fontsize = 30)
ax_dipole = fig.add_subplot(2,2,2,projection='3d')
ax_dipole.scatter(embedding_dipole[:, 0], embedding_dipole[:, 1], embedding_dipole[:, 2], alpha=0.5, c=colors)
ax_dipole.set_title("DIPOLE", fontsize = 30)
ax_LMR = fig.add_subplot(2,2,3,projection='3d')
ax_LMR.scatter(embedding_lmr[:, 0], embedding_lmr[:, 1], embedding_lmr[:, 2], alpha=0.5, c=colors)
ax_LMR.set_title("Local Metric Regularizer", fontsize = 30)
ax_UMAP = fig.add_subplot(2,2,4,projection='3d')
ax_UMAP.scatter(embedding_umap[:, 0], embedding_umap[:, 1], embedding_umap[:, 2], alpha=0.5, c=colors)
ax_UMAP.set_title("UMAP", fontsize = 30)
plt.savefig('figures/faces.png', format='png')