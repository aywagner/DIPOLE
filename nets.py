# Collection of models

import torch
import torch.nn as nn
from losses import LocalMetricRegularizer, DistTopLoss, LocalMetricRegularizerMask


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


class DimensionReductionNetMask(nn.Module):
    def __init__(self, dist_mat, edge_mask, alpha,
                 hom_dims, hom_weights, k, num_subsets, p):
        super(DimensionReductionNetMask, self).__init__()
        self.metric_loss = LocalMetricRegularizerMask(dist_mat, edge_mask)
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
