# Collection of helper functions

import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path


def plot_3d_points(pts, col=None):
    fig = plt.figure(figsize=(10, 10))
    ax_pts = fig.add_subplot(111, projection='3d')
    ax_pts.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.5, c=col)
    plt.show()


def plot_2d_points(pts, col=None):
    fig = plt.figure(figsize=(10, 10))
    ax_pts = fig.add_subplot(111)
    ax_pts.scatter(pts[:, 0], pts[:, 1], alpha=0.5, c=col)
    plt.show()


def subsample(X, voxel_size):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(X)
    downpcd = pcd.voxel_down_sample(voxel_size)
    return np.asarray(downpcd.points)


def knn_dist_mat(data, k):
    graph = kneighbors_graph(data, k, mode='distance')
    return shortest_path(graph, directed=False, unweighted=False)