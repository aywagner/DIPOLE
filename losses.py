# Collection of geometric and topological losses

import torch
import torch.nn as nn
import numpy as np
from pers import RipsPersistenceDistance, WassersteinDistance


class LocalMetricRegularizer(nn.Module):
    """
    pytorch module that computes L_2^2 deviation from edges below a threshold
    """
    def __init__(self, dist_mat, dist_thresh):
        super(LocalMetricRegularizer, self).__init__()
        mask = dist_mat < dist_thresh
        self.indices = torch.nonzero(mask)
        self.small_dists = dist_mat[mask]
         
    def forward(self, input):
        input_diffs = input[self.indices[:, 0], :] - input[self.indices[:, 1], :]
        input_small_dists = torch.linalg.norm(input_diffs, dim=1)
        return ((self.small_dists - input_small_dists)**2).sum()


class LocalMetricRegularizerMask(nn.Module):
    """
    pytorch module that computes L_2^2 deviation from edges specified by a mask
    """
    def __init__(self, dist_mat, edge_mask):
        super(LocalMetricRegularizerMask, self).__init__()
        self.indices = torch.nonzero(edge_mask)
        self.small_dists = dist_mat[edge_mask]
         
    def forward(self, input):
        input_diffs = input[self.indices[:, 0], :] - input[self.indices[:, 1], :]
        input_small_dists = torch.linalg.norm(input_diffs, dim=1)
        return ((self.small_dists - input_small_dists)**2).sum()


class DistTopLoss(nn.Module):
    """
    pytorch module that computes weighted topological loss for random sample
    """
    def __init__(self, dist_mat, hom_dims, hom_weights, k, p):
        """
        dist_mat: target distance matrix
        hom_dims: tuple of persistent homology degrees
        hom_weights: tuple of weights for persistent homology degrees
        k: subset size
        p: choice of Wasserstein distance raised to p
        """
        super(DistTopLoss, self).__init__()
        self.dist_mat = dist_mat
        self.rips = RipsPersistenceDistance(hom_dims)
        self.hom_weights = hom_weights
        self.k = k
        self.wasserstein = WassersteinDistance(p)
  
    def forward(self, embedding):
        # Randomly sample k points.
        indices = torch.tensor(np.random.choice(embedding.shape[0], 
                                                size=self.k, 
                                                replace=False)).type(torch.LongTensor)
        # Compute k x k submatrix of domain distance matrix.
        dom_distmat = self.dist_mat[np.ix_(indices, indices)]
        # Compute k x k submatrix of codomain distance matrix.
        codom_distmat = torch.cdist(embedding[indices, :], embedding[indices, :])
        dom_dgms, codom_dgms = self.rips(dom_distmat), self.rips(codom_distmat)
        top_loss = torch.tensor(0., requires_grad=True)
        for i in np.arange(len(self.hom_weights)):
            top_loss = top_loss + self.hom_weights[i] * \
                self.wasserstein(dom_dgms[i], codom_dgms[i])
        return top_loss
    