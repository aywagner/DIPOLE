# Collection of dimensionality reduction tests

import numpy as np
import pandas as pd
import gudhi as gd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, kendalltau
from gudhi.wasserstein import wasserstein_distance
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

def ijk_test(high_distmat, low_distmat, num_pairs):
    assert high_distmat.shape == low_distmat.shape
    n_pts = high_distmat.shape[0]
    while True:
        ijks = np.random.choice(n_pts, size=(2*num_pairs, 3))
        is_distinct = (ijks[:, 0] != ijks[:, 1]) & (ijks[:, 0] != ijks[:, 2]) & (ijks[:, 1] != ijks[:, 2])
        ijks = ijks[is_distinct, :]
        if ijks.shape[0] >= num_pairs:
            ijks = ijks[:num_pairs, :]
            break
    high_order = high_distmat[ijks[:, 0], ijks[:, 1]] > high_distmat[ijks[:, 0], ijks[:, 2]]
    low_order = low_distmat[ijks[:, 0], ijks[:, 1]] > low_distmat[ijks[:, 0], ijks[:, 2]]
    prob_agreement = np.sum(high_order == low_order) / num_pairs
    return 1 - prob_agreement


def kendall_test(high_distmat, low_distmat):
    tau = kendalltau(high_distmat, low_distmat)[0]
    return -tau


def resvar_test(high_distmat, low_distmat):
    r = pearsonr(high_distmat.ravel(), low_distmat.ravel())[0]
    return 1 - r**2


def distmat_furthest_point(distmat, n_pts):
    n = distmat.shape[0]
    assert n >= n_pts
    # perm = indices of furthest point sample
    perm = np.zeros(n_pts, dtype=int)
    # Choose starting point of furthest point sample randomly
    idx = np.random.choice(n, 1)
    perm[0] = idx
    dists = distmat[idx, :]
    for i in range(1, n_pts):
        idx = np.argmax(dists)
        perm[i] = idx
        dists = np.minimum(dists, distmat[idx, :])
    return perm


def get_dgms(distmat):
    rips = gd.RipsComplex(distance_matrix=distmat)
    st = rips.create_simplex_tree(max_dimension=2)
    st.persistence()
    dgm0, dgm1 = st.persistence_intervals_in_dimension(0), st.persistence_intervals_in_dimension(1)
    # Remove infinite bars
    dgm0, dgm1 = dgm0[dgm0[:, 1] < np.inf, :], dgm1[dgm1[:, 1] < np.inf, :]
    return dgm0, dgm1


def globaltop_test(high_distmat, low_distmat, n_pts):
    high_perm, low_perm = distmat_furthest_point(high_distmat, n_pts), distmat_furthest_point(low_distmat, n_pts)
    high_distmat, low_distmat = high_distmat[np.ix_(high_perm, high_perm)], low_distmat[np.ix_(low_perm, low_perm)]
    high_dgm0, high_dgm1 = get_dgms(high_distmat)
    low_dgm0, low_dgm1 = get_dgms(low_distmat)
    return wasserstein_distance(high_dgm0, low_dgm0, order=2), wasserstein_distance(high_dgm1, low_dgm1, order=2)


def quality_gauntlet(high_distmat, dr_method, dataset, hyperparameters):
    """
    high_distmat: "target" distance matrix
    dataset: high dimensional input
    dr_method: DR function with input (dataset, **hyperparam.to_dict('records'))
    hyperparameters: pandas.DataFrame of hyperparameters
    """
    assert np.max(high_distmat) < np.inf, "High/target distance matrix has infinite values."
    hyperparameters_dicts = hyperparameters.to_dict('records')
    quality_metrics = np.zeros((len(hyperparameters_dicts), 5))
    for i in tqdm(range(len(hyperparameters_dicts))):
        embedding = dr_method(dataset, **hyperparameters_dicts[i])
        low_distmat = squareform(pdist(embedding))
        quality_metrics[i, 0] = ijk_test(high_distmat, low_distmat, num_pairs=10000)
        quality_metrics[i, 1] = kendall_test(high_distmat, low_distmat)
        quality_metrics[i, 2] = resvar_test(high_distmat, low_distmat)
        quality_metrics[i, 3:5] = globaltop_test(high_distmat, low_distmat, n_pts=256)
    quality_metrics = pd.DataFrame(quality_metrics, columns=['ijk', 'kendall', 'residual variance', 'globalh0','globalh1'])
    return pd.concat([hyperparameters, quality_metrics], axis=1)


def compare_metrics(name):
    """
    Saves figure comparing dimensionality reduction techniques across metrics
    for a given dataset
    """
    # Load in results
    dr_methods = ['isomap', 'tsne', 'umap', 'dipole', 'lmr']
    results = []
    for dr_method in dr_methods:
         results.append(pd.read_csv('tables/' + dr_method + '_' + name + '.csv'))
    metrics = ['ijk', 'residual variance', 'globalh0','globalh1']
    n_bins = 32
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    for i in range(len(metrics)):    
        row, col = np.unravel_index(i, (2, 2))
        # Grab the results for the metric in question
        results_data = [result[metrics[i]] for result in results]
        # Plot shared histogram of values
        axs[row, col].hist(results_data, n_bins, density=False, histtype='barstacked', label=dr_methods)
        if i == 0:
            axs[row, col].legend(prop={'size' : 10})
        axs[row, col].set_title(metrics[i])
    fig.tight_layout()
    plt.savefig('figures/' + name + '_metrics.png', format='png')