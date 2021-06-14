# Collection of dimensionality reduction methods

from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import umap

def TSNE_method(high_ptcloud, target_dim=3, perplexity=30.):
    embedding = TSNE(n_components=target_dim, perplexity=perplexity)
    return embedding.fit_transform(high_ptcloud)


def Isomap_method(high_ptcloud, target_dim=3):
    embedding = Isomap(n_components=target_dim)
    return embedding.fit_transform(high_ptcloud)


def umap_method(high_ptcloud, target_dim, n_neighbors, min_dist):
    fit = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=target_dim)
    return fit.fit_transform(high_ptcloud)


from nets import DimensionReductionNet, DimensionReductionNetMask
from helper import knn_dist_mat
import torch
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

def DIPOLE_quantiles_method(high_ptcloud, target_dim=3, q=0.01, alpha=0.1, k=32, lr=0.1, m=5):
    """
    q = quantile used for local distances
    alpha = tradeoff parameter for geo/top loss
    k = size of subsets used for top loss
    m = number of nearest neighbors for intrinsic metric
    """
    embedding = torch.tensor(Isomap_method(high_ptcloud=high_ptcloud, target_dim=target_dim), requires_grad=True)
    intrinsic_distmat = torch.tensor(knn_dist_mat(data=high_ptcloud, k=m))
    dist_thresh = np.quantile(intrinsic_distmat.numpy(), q)
    net = DimensionReductionNet(dist_mat=intrinsic_distmat,
                                dist_thresh=dist_thresh, 
                                alpha=alpha, 
                                hom_dims=(0,1), 
                                hom_weights=(0.5,0.5), 
                                k=k, 
                                num_subsets=1,
                                p=2)
    opt = torch.optim.Adam([embedding], lr=lr)
    scheduler = LambdaLR(opt,[lambda epoch: 1000./(1000+epoch)]) 
    for _ in range(2500):
        opt.zero_grad()
        net(embedding).backward()
        opt.step()
        scheduler.step()
    return embedding.detach().numpy()


from sklearn.neighbors import kneighbors_graph

def DIPOLE_mask_method(high_ptcloud, target_dim=3, lmr_edges=3, alpha=0.1, k=32, lr=0.1, m=5):
    """
    lmr_edges = number of edges for nearest neighbors edge mask
    alpha = tradeoff parameter for geo/top loss
    k = size of subsets used for top loss
    m = number of nearest neighbors for intrinsic metric
    """
    embedding = torch.tensor(Isomap_method(high_ptcloud=high_ptcloud, target_dim=target_dim), requires_grad=True)
    intrinsic_distmat = torch.tensor(knn_dist_mat(data=high_ptcloud, k=m))
    edge_mask = torch.tensor(kneighbors_graph(high_ptcloud, lmr_edges, mode='connectivity').A > 0)
    net = DimensionReductionNetMask(dist_mat=intrinsic_distmat,
                                    edge_mask=edge_mask,
                                    alpha=alpha, 
                                    hom_dims=(0,1), 
                                    hom_weights=(0.5,0.5), 
                                    k=k, 
                                    num_subsets=1,
                                    p=2)
    opt = torch.optim.Adam([embedding], lr=lr)
    scheduler = LambdaLR(opt,[lambda epoch: 1000./(1000+epoch)]) 
    for _ in range(2500):
        opt.zero_grad()
        net(embedding).backward()
        opt.step()
        scheduler.step()
    return embedding.detach().numpy()

from nets import onlyLMRNet

def LMR_method(high_ptcloud, target_dim, lmr_edges=3, lr=0.1, m=5):
    """
    lmr_edges = number of edges for nearest neighbors edge mask
    m = number of nearest neighbors for intrinsic metric
    """
    embedding = torch.tensor(Isomap_method(high_ptcloud=high_ptcloud, target_dim=target_dim), requires_grad=True)
    intrinsic_distmat = torch.tensor(knn_dist_mat(data=high_ptcloud, k=m))
    edge_mask = torch.tensor(kneighbors_graph(high_ptcloud, lmr_edges, mode='connectivity').A > 0)
    net = onlyLMRNet(dist_mat=intrinsic_distmat, edge_mask=edge_mask)
    opt = torch.optim.Adam([embedding], lr=lr)
    scheduler = LambdaLR(opt,[lambda epoch: 1000./(1000+epoch)]) 
    for _ in range(2500):
        opt.zero_grad()
        net(embedding).backward()
        opt.step()
        scheduler.step()
    return embedding.detach().numpy()
