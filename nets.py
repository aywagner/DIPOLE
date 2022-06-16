# Collection of models

import numpy as np
import torch
import torch.nn as nn
from gph import ripser_parallel
from gudhi.wasserstein import wasserstein_distance
from joblib import Parallel, delayed

from losses import LocalMetricRegularizer, LocalMetricRegularizerMask


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


def compute_pers_gens_and_matchings(dom_distmat_np, codom_distmat_np,
                                    hom_dims, p):
    ripser_dom = ripser_parallel(dom_distmat_np,
                                 metric="precomputed",
                                 maxdim=max(hom_dims),
                                 return_generators=True)
    ripser_codom = ripser_parallel(codom_distmat_np,
                                   metric="precomputed",
                                   maxdim=max(hom_dims),
                                   return_generators=True)
    pers_gens_dom, pers_gens_codom = ripser_dom["gens"], ripser_codom["gens"]
    dgms_dom, dgms_codom = ripser_dom["dgms"], ripser_codom["dgms"]
    dgms_dom[0] = dgms_dom[0][:-1, :]
    dgms_codom[0] = dgms_codom[0][:-1, :]
    matchings_by_dim = [
        (wasserstein_distance(
            dgms_dom[d], dgms_codom[d], matching=True, order=p, internal_p=1
            )[1]
         if (len(dgms_dom[d]) and len(dgms_codom[d])) else None)
        for d in hom_dims
        ]

    return pers_gens_dom, pers_gens_codom, matchings_by_dim


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
        indices = torch.tensor(
            np.random.choice(embedding.shape[0], size=self.k, replace=False)
            ).type(torch.LongTensor)
        # Compute k x k submatrix of domain distance matrix.
        dom_distmat = self.dist_mat[np.ix_(indices, indices)]
        # Compute k x k submatrix of codomain distance matrix.
        codom_distmat = torch.cdist(embedding[indices, :],
                                    embedding[indices, :])

        return dom_distmat, codom_distmat

    def _dgms_from_gens(self, pers_gens, distmat):
        dgms = []
        for d in self.hom_dims:
            if not d:
                if not len(pers_gens[0]):
                    dgms.append(torch.zeros((0, 2), requires_grad=True))
                else:
                    verts = torch.from_numpy(
                        pers_gens[0]
                        ).type(torch.LongTensor)
                    dgm = torch.stack((distmat[verts[:, 0], verts[:, 0]],
                                       distmat[verts[:, 1], verts[:, 2]]), 1)
                    dgms.append(dgm)
            elif not len(pers_gens[1]):
                dgms.append(torch.zeros((0, 2), requires_grad=True))
            else:
                verts = torch.from_numpy(
                    pers_gens[1][d - 1]
                    ).type(torch.LongTensor)
                dgm = torch.stack((distmat[verts[:, 0], verts[:, 1]],
                                   distmat[verts[:, 2], verts[:, 3]]), 1)
                dgms.append(dgm)

        return dgms

    def _wasserstein(self, dgm1, dgm2, matching):
        # First check if either diagram is empty
        if dgm1.shape[0] == 0:
            return torch.sum(torch.pow(dgm2[:, 1] - dgm2[:, 0], self.p))
        elif dgm2.shape[0] == 0:
            return torch.sum(torch.pow(dgm1[:, 1] - dgm1[:, 0], self.p))
        # Initialize cost
        cost = torch.tensor(0., requires_grad=True)
        # Note these calculations are using L1 ground metric on upper half-plane
        is_unpaired_1 = (matching[:, 1] == -1)
        if np.any(is_unpaired_1):
            unpaired_1_idx = matching[is_unpaired_1, 0]
            cost = cost + torch.sum(
                torch.pow(dgm1[unpaired_1_idx, 1] - dgm1[unpaired_1_idx, 0],
                          self.p)
                )
        is_unpaired_2 = (matching[:, 0] == -1)
        if np.any(is_unpaired_2):
            unpaired_2_idx = matching[is_unpaired_2, 1]
            cost = cost + torch.sum(
                torch.pow(dgm2[unpaired_2_idx, 1] - dgm2[unpaired_2_idx, 0],
                          self.p)
                )
        is_paired = (~is_unpaired_1 & ~is_unpaired_2)
        if np.any(is_paired):
            paired_1_idx, paired_2_idx = matching[is_paired, 0], matching[is_paired, 1]
            paired_dists = torch.sum(
                torch.abs(dgm1[paired_1_idx, :] - dgm2[paired_2_idx, :]),
                dim=1
                )
            paired_costs = torch.sum(torch.pow(paired_dists, self.p))
            cost = cost + paired_costs

        return cost

    def forward(self, embedding):
        # Compute the metric loss
        metric_loss = self.metric_loss(embedding)

        submats = [self._create_submatrices(embedding)
                   for _ in range(self.num_subsets)]
        pers_gens_and_matchings = Parallel(n_jobs=-1)(
            delayed(compute_pers_gens_and_matchings)(
                submats_in_subset[0].detach().numpy(),
                submats_in_subset[1].detach().numpy(),
                self.hom_dims,
                self.p
                )
            for submats_in_subset in submats
        )

        # Compute the average topological loss
        top_loss = torch.tensor(0., requires_grad=True)
        for i in range(self.num_subsets):
            dom_dgms = self._dgms_from_gens(pers_gens_and_matchings[i][0],
                                            submats[i][0])
            codom_dgms = self._dgms_from_gens(pers_gens_and_matchings[i][1],
                                              submats[i][1])
            matchings = pers_gens_and_matchings[i][2]
            top_loss_in_subset = torch.tensor(0., requires_grad=True)
            for j in np.arange(len(self.hom_weights)):
                top_loss_in_subset = \
                    (top_loss_in_subset +
                     self.hom_weights[j] * self._wasserstein(dom_dgms[j],
                                                             codom_dgms[j],
                                                             matchings[j]))

            top_loss = top_loss + top_loss_in_subset

        return self.alpha * metric_loss + \
            (1-self.alpha) * (top_loss / self.num_subsets)
