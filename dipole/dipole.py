import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from gph import ripser_parallel
from gudhi.wasserstein import wasserstein_distance
from joblib import Parallel, delayed
from sklearn.manifold import Isomap
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path


class LocalMetricRegularizer(nn.Module):
    """
    pytorch module that computes L_2^2 deviation from special edges
    
    Args:
        dist_func: function accepting n x 2 arrays and 
        returning distances as tensors
        edge_indices: n x 2 tensor with edge indices
        weights: length n tensor with edge weights. If none, all weights are 1. 
        Defaults to None.
    """
    def __init__(self, dist_func, edge_indices, weights=None):
        super(LocalMetricRegularizer, self).__init__()
        assert edge_indices.shape[0] > 0
        self.edge_indices = edge_indices
        self.small_dists = dist_func(self.edge_indices)
        # TODO: Refactor to make weights adjustable during training?
        if weights is not None:
            self.weights = weights
        else:
            self.weights = torch.ones(edge_indices.shape[0])
         
    def forward(self, input):
        input_diffs = input[self.edge_indices[:, 0], :] - \
            input[self.edge_indices[:, 1], :]
        input_small_dists = torch.linalg.norm(input_diffs, dim=1)
        return torch.dot(self.weights, (self.small_dists - input_small_dists)**2)


def _compute_pers_gens_and_matchings(dom_distmat_np, codom_distmat_np, hom_dims, p):
    """
    Computes persistence generators and optimal matching achieving Wasserstein distance

    Args:
        dom_distmat_np (numpy array): domain distance matrix
        codom_distmat_np (numpy array): codomain distance matrix
        hom_dims (tuple): persistence homology dimensions
        p (scalar): order of Wasserstein distance
    """
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


class DIPOLELoss(nn.Module):
    """
    pytorch module computing the DIPOLE loss

    Args:
        dist_func: function accepting n x 2 arrays and 
        returning distances as tensors
        edge_indices: n x 2 tensor of edges indices
        alpha: tradeoff parameter between topological loss and
        local metric regularizer
        hom_dims: tuple of homology dimensions
        hom_weights: tuple of weights for each dimension
        k: size of subsets
        num_subsets: number of subsets to compute, i.e. batch size
        p: order of Wasserstein distance
    """
    def __init__(self, dist_func, edge_indices, weights, alpha,
                 hom_dims, hom_weights, k, num_subsets, p):
        super(DIPOLELoss, self).__init__()
        self.dist_func = dist_func
        self.metric_loss = LocalMetricRegularizer(dist_func, edge_indices, weights)
        self.alpha = alpha
        self.hom_dims = hom_dims
        self.hom_weights = hom_weights
        self.k = k
        self.num_subsets = num_subsets
        self.p = p
        # TODO: Overwrite init to include sphere diagrams

    def _create_submatrices(self, embedding):
        # Randomly sample k points.
        indices = torch.tensor(
            np.random.choice(embedding.shape[0], size=self.k, replace=False)
            ).type(torch.LongTensor)
        # Compute k x k submatrix of domain distance matrix.
        dom_distmat = self.dist_func(torch.cartesian_prod(indices, indices))
        dom_distmat = dom_distmat.reshape(self.k, self.k)
        # dom_distmat = self.dist_mat[np.ix_(indices, indices)]
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
        if self.alpha == 1:
            return metric_loss
        # Compute submatrices corresponding to subsets
        submats = [self._create_submatrices(embedding)
                   for _ in range(self.num_subsets)]
        pers_gens_and_matchings = Parallel(n_jobs=-1)(
            delayed(_compute_pers_gens_and_matchings)(
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


class EarlyStopping():
    """
    early stopping for pytorch
    
    Args:
        patience: number of consecutive iterations to trigger early stop
        min_rel_change: smallest relative change not triggering patience counter
    """
    def __init__(self, patience, min_rel_change):
        self.patience = patience
        self.min_rel_change = min_rel_change
        self.counter = 0
        self.prev_loss = float('Inf')
        self.early_stop = False
    
    def __call__(self, loss):
        if 1 - (loss/self.prev_loss) < self.min_rel_change:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
        self.prev_loss = loss


def DIPOLE(dist_func, 
           embedding, 
           edge_indices,
           weights,
           num_subsets,
           alpha=0.1, 
           k=32, 
           lr=0.1, 
           n_iterations=2500, 
           patience=10, 
           min_rel_change=0.01):
    """
    General implementation of DIPOLE

    Args:
        dist_func (function): function accepting n x 2 arrays and 
        returning distances as tensors
        embedding (torch.tensor): initial embedding with one row per point
        edge_indices (torch.tensor): n x 2 tensor of edge indices
        num_subsets(int): number of subsets to compute, i.e. batch size
        alpha (float, optional): tradeoff parameter between topological loss
        and local metric regularizer. Defaults to 0.1.
        k (int, optional): Size of subsets. Defaults to 32.
        lr (float, optional): Learning rate. Defaults to 0.1.
        n_iterations (int, optional): Max number of iterations. Defaults to 2500.
        patience (int, optional): number of consecutive iterations to trigger 
        early stop. Defaults to 10.
        min_rel_change (float, optional): smallest relative change not 
        triggering patience counter. Defaults to 0.01.

    Returns:
        np.ndarray: embedding
    """
    net = DIPOLELoss(dist_func=dist_func,
                     edge_indices=edge_indices,
                     weights=weights,
                     alpha=alpha, 
                     hom_dims=(0,1), 
                     hom_weights=(0.5,0.5), 
                     k=k, 
                     num_subsets=num_subsets,
                     p=1)
    opt = torch.optim.Adam([embedding], lr=lr)
    scheduler = LambdaLR(opt,[lambda epoch: 1000./(1000+epoch)])
    early_stopping = EarlyStopping(patience, min_rel_change) 
    for i in range(n_iterations):
        opt.zero_grad()
        loss = net(embedding)
        print(loss.item()) # TODO: Remove
        early_stopping(loss.item())
        if early_stopping.early_stop:
            print('At epoch:', i)
            break
        net(embedding).backward()
        opt.step()
        scheduler.step()
    return embedding.detach().numpy()


def knn_dist_mat(data, k):
    """Computes k-nearest neighbors distance matrix

    Args:
        data (np.ndarray): point cloud
        k (int): number of nearest neighbors

    Returns:
        np.ndarray : path distance on knn graph
    """
    graph = kneighbors_graph(data, k, mode='distance')
    return shortest_path(graph, directed=False, unweighted=False)


def DietDIPOLE(high_ptcloud, 
               target_dim, 
               num_subsets,
               lmr_edges=3, 
               alpha=0.1, 
               k=32, 
               m=5, 
               lr=0.1, 
               n_iterations=2500, 
               patience=10, 
               min_rel_change=0.01):
    """
    General implementation of DIPOLE

    Args:
        high_ptcloud (np.ndarray): original, high-dimensional pointcloud
        target_dim (int): dimension in which to embed
        num_subsets(int): number of subsets to compute, i.e. batch size
        lmr_edges (int): number of edges to use in the local metric regularizer
        alpha (float, optional): tradeoff parameter between topological loss
        and local metric regularizer. Defaults to 0.1.
        k (int, optional): Size of subsets. Defaults to 32.
        m (int, optional): Number of nearest neighbors for intrinsic metric
        lr (float, optional): Learning rate. Defaults to 0.1.
        n_iterations (int, optional): Max number of iterations. Defaults to 2500.
        patience (int, optional): number of consecutive iterations to trigger 
        early stop. Defaults to 10.
        min_rel_change (float, optional): smallest relative change not 
        triggering patience counter. Defaults to 0.01.

    Returns:
        np.ndarray: embedding
    """
    embedding = Isomap(n_components=target_dim)
    embedding = torch.tensor(embedding.fit_transform(high_ptcloud),
                             requires_grad=True,
                             dtype=torch.float32)
    dist_mat = torch.tensor(knn_dist_mat(data=high_ptcloud, k=m), dtype=torch.float32)
    edge_indices = torch.tensor(kneighbors_graph(high_ptcloud, 
                                                 lmr_edges, 
                                                 mode='connectivity').nonzero()).T.type(torch.LongTensor)
    return DIPOLE(dist_func=lambda x: dist_mat[x[:, 0], x[:, 1]],
                  embedding=embedding,
                  edge_indices=edge_indices,
                  weights=torch.ones(edge_indices.shape[0]),
                  num_subsets=num_subsets,
                  alpha=alpha,
                  k=k,
                  lr=lr,
                  n_iterations=n_iterations,
                  patience=patience,
                  min_rel_change=min_rel_change)
    
    
def DietDIPOLE2(high_ptcloud, 
               target_dim, 
               num_subsets,
               tau = 100.,
               alpha=1, 
               k=32, 
               m=5, 
               lr=0.1, 
               n_iterations=2500, 
               patience=10, 
               min_rel_change=0.01):
    
    embedding = Isomap(n_components=target_dim)
    
    # from sklearn.random_projection import GaussianRandomProjection
    # embedding = GaussianRandomProjection(n_components=target_dim)
    
    embedding = torch.tensor(embedding.fit_transform(high_ptcloud),
                             requires_grad=True,
                             dtype=torch.float32)
    dist_mat = torch.tensor(knn_dist_mat(data=high_ptcloud, k=m), dtype=torch.float32)
    
    # non-uniform weights
    weights = torch.exp(-(dist_mat ** 2) / tau)
    # weights = dist_mat
    top_n = torch.argsort(-weights, dim=1) < 10
    edge_indices = torch.nonzero(top_n)
    sparse_weights = weights[edge_indices[:, 0], edge_indices[:, 1]]
    return DIPOLE(dist_func=lambda x: dist_mat[x[:, 0], x[:, 1]],
                  embedding=embedding,
                  edge_indices=edge_indices,
                  weights=sparse_weights,
                  num_subsets=num_subsets,
                  alpha=alpha,
                  k=k,
                  lr=lr,
                  n_iterations=n_iterations,
                  patience=patience,
                  min_rel_change=min_rel_change)


if __name__=='__main__':
    import matplotlib.pyplot as plt
    data = np.loadtxt('data/mammoth.txt')
    col = data[:, 1]
    embedding = DietDIPOLE2(high_ptcloud=data,
                           target_dim=2,
                           num_subsets=2,
                           tau=1.,
                           alpha=0.001,
                           k=64,
                           m=5,
                           lr=0.5)
    fig = plt.figure(figsize=(10,10))
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5, c=col)
    plt.show()
    