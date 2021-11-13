# Collection of geometric and topological losses

import torch
import torch.nn as nn


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
