# This script performs the quantitative analysis found in the manuscript.

import numpy as np
import pandas as pd
from drmethods import Isomap_method, TSNE_method, umap_method, DIPOLE_mask_method, LMR_method
from drtests import quality_gauntlet, compare_metrics
from helper import knn_dist_mat


def make_hyperparam_df(hyperparams, names, types):
    """
    creates hyperparameter grid out of choices for each parameter
    hyperparams: list of numpy arrays of parameter choices
    names: list of names of hyperparams
    types: list of types
    """
    # Create mesh
    params = np.array(np.meshgrid(*hyperparams)).T.reshape(-1, len(hyperparams))
    params = pd.DataFrame(params, columns=names)
    # Fix types
    params = params.astype(dict(zip(names, types)))
    return params
    

def isomap_metrics(data, high_distmat, target_dim, name):
    target_dims = np.array([target_dim])
    params = make_hyperparam_df([target_dims], ['target_dim'], [int])
    results = quality_gauntlet(high_distmat, Isomap_method, data, params)
    results.to_csv('tables/isomap_' + name + '.csv', index=False)


def tsne_metrics(data, high_distmat, target_dim, name):
    # Define individual params
    target_dims = np.array([target_dim])
    perplexity = np.linspace(start=2, stop=75, num=32)
    names = ['target_dim', 'perplexity']
    types = [int, float]
    params = make_hyperparam_df([target_dims, perplexity], names, types)
    results = quality_gauntlet(high_distmat, TSNE_method, data, params)
    results.to_csv('tables/tsne_' + name + '.csv', index=False)


def umap_metrics(data, high_distmat, target_dim, name):
    # Define individual params
    target_dims = np.array([target_dim])
    n_neighbors = np.array([4, 8, 16, 32, 64, 128])
    min_dists = np.array([0, 0.1, 0.25, 0.5, 0.8])
    names = ['target_dim', 'n_neighbors', 'min_dist']
    types = [int, int, float]
    params = make_hyperparam_df([target_dims, n_neighbors, min_dists], names, types)
    results = quality_gauntlet(high_distmat, umap_method, data, params)
    results.to_csv('tables/umap_' + name + '.csv', index=False)


def dipole_metrics(data, high_distmat, target_dim, name):
    # Define individual params
    target_dims = np.array([target_dim])
    lmr_edges = np.array([3, 5])
    alphas = np.array([0.01, 0.1])
    ks = np.array([32, 64])
    lrs = np.array([0.1, 1, 2])
    names = ['target_dim', 'lmr_edges', 'alpha', 'k', 'lr']
    types = [int, int, float, int, float]
    params = make_hyperparam_df([target_dims, lmr_edges, alphas, ks, lrs], names, types)
    results = quality_gauntlet(high_distmat, DIPOLE_mask_method, data, params)
    results.to_csv('tables/dipole_' + name + '.csv', index=False)


def LMR_metrics(data, high_distmat, target_dim, name):
    # Define individual params
    target_dims = np.array([target_dim])
    lmr_edges = np.array([2, 3, 5, 10])
    lrs = np.array([0.1, 1, 2])
    names = ['target_dim', 'lmr_edges', 'lr']
    types = [int, int, float]
    params = make_hyperparam_df([target_dims, lmr_edges, lrs], names, types)
    results = quality_gauntlet(high_distmat, LMR_method, data, params)
    results.to_csv('tables/lmr_' + name + '.csv', index=False)
    
    
def all_metrics(data, high_distmat, target_dim, name):
    """
    Helper function to run all DR methods on same inputs
    """
    isomap_metrics(data=data, high_distmat=high_distmat, target_dim=target_dim, name=name)
    tsne_metrics(data=data, high_distmat=high_distmat, target_dim=target_dim, name=name)
    umap_metrics(data=data, high_distmat=high_distmat, target_dim=target_dim, name=name)
    dipole_metrics(data=data, high_distmat=high_distmat, target_dim=target_dim, name=name)
    LMR_metrics(data=data, high_distmat=high_distmat, target_dim=target_dim, name=name)


# Mammoth
name = 'mammoth'
data = np.loadtxt('data/mammoth.txt')
high_distmat = knn_dist_mat(data, 5)
target_dim = 2
all_metrics(data, high_distmat, target_dim=target_dim, name=name)
compare_metrics(name)

# Brain
name = 'brain'
data = np.loadtxt('data/brain.txt')
high_distmat = knn_dist_mat(data, 5)
target_dim = 2
all_metrics(data, high_distmat, target_dim=target_dim, name=name)
compare_metrics(name)

# Swiss roll with holes
name = 'swisshole'
data = np.loadtxt('data/swisshole.txt')
high_distmat = knn_dist_mat(data, 5)
target_dim = 2
all_metrics(data, high_distmat, target_dim=target_dim, name=name)
compare_metrics(name)

# Stanford Faces dataset
from scipy.io import loadmat
name = 'faces'
mat = loadmat('data/face_data.mat')
data = mat['images'].T
high_distmat = knn_dist_mat(data, 5)
target_dim = 3
all_metrics(data, high_distmat, target_dim=target_dim, name=name)
compare_metrics(name)