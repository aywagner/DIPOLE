# Collection of models

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed

from losses import LocalMetricRegularizer, DistTopLoss, \
    LocalMetricRegularizerMask
from pers import RipsPersistenceDistance, WassersteinDistance


class onlyLMRNet(nn.Module):
    def __init__(self, dist_mat, edge_mask):
        super(onlyLMRNet, self).__init__()
        self.metric_loss = LocalMetricRegularizerMask(dist_mat, edge_mask)
  
    def forward(self, embedding):
        # Compute the metric loss
        return self.metric_loss(embedding)


class DimensionReductionNet(nn.Module):
    def __init__(self, dist_mat, dist_thresh, alpha,
                 hom_dims, hom_weights, k, num_subsets, p):
        super(DimensionReductionNet, self).__init__()
        self.metric_loss = LocalMetricRegularizer(dist_mat, dist_thresh)
        self.top_loss = DistTopLoss(dist_mat, hom_dims, hom_weights, k, p)
        self.num_subsets = num_subsets
        self.alpha = alpha
  
    def forward(self, embedding):
        # Compute the metric loss
        metric_loss = self.metric_loss(embedding)
        # Compute the average topological loss
        top_loss = torch.tensor(0., requires_grad=True)
        for _ in range(self.num_subsets):
            top_loss = top_loss + self.top_loss(embedding)
        return self.alpha * metric_loss + (1-self.alpha) * (top_loss / self.num_subsets)


def dist_top_loss(subsampled_distmats, hom_dims, hom_weights, p):
    dom_distmat, codom_distmat = subsampled_distmats
    rips = RipsPersistenceDistance(hom_dims)
    wasserstein = WassersteinDistance(p)
    dom_dgms, codom_dgms = rips(dom_distmat), rips(codom_distmat)

    top_loss = torch.tensor(0., requires_grad=True)
    for i in np.arange(len(hom_weights)):
        top_loss = \
            top_loss + hom_weights[i] * wasserstein(dom_dgms[i], codom_dgms[i])

    return top_loss


class DimensionReductionNetMask(nn.Module):
    def __init__(self, dist_mat, edge_mask, alpha,
                 hom_dims, hom_weights, k, num_subsets, p):
        super(DimensionReductionNetMask, self).__init__()
        self.dist_mat = dist_mat
        self.metric_loss = LocalMetricRegularizerMask(dist_mat, edge_mask)
        self.alpha = alpha
        self.hom_dims = hom_dims
        self.hom_weights = hom_weights
        self.k = k
        self.num_subsets = num_subsets
        self.p = p

    def _create_submatrices(self, embedding):
        # Randomly sample k points.
        indices = torch.tensor(np.random.choice(embedding.shape[0],
                                                size=self.k,
                                                replace=False)).type(
            torch.LongTensor)
        # Compute k x k submatrix of domain distance matrix.
        dom_distmat = self.dist_mat[np.ix_(indices, indices)]
        # Compute k x k submatrix of codomain distance matrix.
        codom_distmat = torch.cdist(embedding[indices, :],
                                    embedding[indices, :])

        return dom_distmat, codom_distmat

    def forward(self, embedding):
        # Compute the metric loss
        metric_loss = self.metric_loss(embedding)
        # Compute the average topological loss
        top_loss = torch.tensor(0., requires_grad=True)
        top_losses = Parallel(n_jobs=-1)(
            delayed(dist_top_loss)(self._create_submatrices(embedding),
                                   self.hom_dims, self.hom_weights, self.p)
            for _ in range(self.num_subsets)
        )
        for i in range(self.num_subsets):
            top_loss = top_loss + top_losses[i]

        return self.alpha * metric_loss + \
            (1-self.alpha) * (top_loss / self.num_subsets)
