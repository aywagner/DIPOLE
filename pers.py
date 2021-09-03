# Collection of pytorch modules for persistence related calculations

import gudhi as gd
from gudhi.wasserstein import wasserstein_distance
import torch
import torch.nn as nn
import numpy as np


class RipsPersistenceDistance(nn.Module):
    """
    pytorch module that takes a distance matrix and outputs a list of persistence diagrams
    of the associated Rips filtration
    """
    def __init__(self, hom_dims):
        """
        hom_dims: a tuple of degrees in which to compute persistent homology
        """
        super(RipsPersistenceDistance, self).__init__()
        self.hom_dims = hom_dims

    def forward(self, input):
        # Compute persistence from distance matrix.
        rips = gd.RipsComplex(distance_matrix=input.detach().numpy())
        st = rips.create_simplex_tree(max_dimension=max(self.hom_dims) + 1)
        st.persistence()
        idx = st.flag_persistence_generators()
        dgms = []
        for hom_dim in self.hom_dims:
            if hom_dim == 0:
                if idx[0].shape[0] == 0:
                    dgms.append(torch.zeros((0, 2), requires_grad=True))
                else:
                    verts = torch.from_numpy(idx[0]).type(torch.LongTensor)
                    dgm = torch.stack((input[verts[:, 0], verts[:, 0]], input[verts[:, 1], verts[:, 2]]), 1)
                    dgms.append(dgm)
            if hom_dim != 0:
                if len(idx[1]) == 0:
                    dgms.append(torch.zeros((0, 2), requires_grad=True))
                else:
                    verts = torch.from_numpy(idx[1][hom_dim - 1]).type(torch.LongTensor)
                    dgm = torch.stack((input[verts[:, 0], verts[:, 1]], input[verts[:, 2], verts[:, 3]]), 1)
                    dgms.append(dgm)
        return dgms
     
     
class WassersteinDistance(nn.Module):
    """
    pytorch module that computes p-Wasserstein distance to the p power between two persistence diagrams.
    """
    def __init__(self, p):
        super(WassersteinDistance, self).__init__()
        self.p = p
    
    def forward(self, dgm1, dgm2):
        # First check if either diagram is empty
        if dgm1.shape[0] == 0:
            return torch.sum(torch.pow(dgm2[:, 1]- dgm2[:, 0], self.p))
        if dgm2.shape[0] == 0:
            return torch.sum(torch.pow(dgm1[:, 1]- dgm1[:, 0], self.p))
        dgm1_np, dgm2_np = dgm1.detach().numpy(), dgm2.detach().numpy()
        # Compute optimal matching using GUDHI
        matching = wasserstein_distance(dgm1_np, dgm2_np, matching=True, order=self.p, internal_p=1)[1]
        # Initialize cost
        cost = torch.tensor(0., requires_grad=True)
        # Note these calculations are using L1 ground metric on upper half-plane
        is_unpaired_1 = (matching[:, 1] == -1)
        if np.any(is_unpaired_1):
            unpaired_1_idx = matching[is_unpaired_1, 0]
            cost = cost + torch.sum(torch.pow(dgm1[unpaired_1_idx, 1] - dgm1[unpaired_1_idx, 0], self.p))
        is_unpaired_2 = (matching[:, 0] == -1)
        if np.any(is_unpaired_2):
            unpaired_2_idx = matching[is_unpaired_2, 1]
            cost = cost + torch.sum(torch.pow(dgm2[unpaired_2_idx, 1] - dgm2[unpaired_2_idx, 0], self.p))
        is_paired = (~is_unpaired_1 & ~is_unpaired_2)
        if np.any(is_paired):
            paired_1_idx, paired_2_idx = matching[is_paired, 0], matching[is_paired, 1]
            paired_dists = torch.sum(torch.abs(dgm1[paired_1_idx, :] - dgm2[paired_2_idx, :]), dim=1)
            paired_costs = torch.sum(torch.pow(paired_dists, self.p))
            cost = cost + paired_costs
        return cost
    

if __name__ == '__main__':
    import numpy as np
    # Testing RipsPersistenceDistance
    ptcloud = np.random.uniform(size=(100, 2))
    x = torch.tensor(ptcloud, requires_grad=True)
    rips = RipsPersistenceDistance((0, 1))
    dgms = rips(torch.cdist(x, x))
    for dim in np.arange(len(dgms)):
        print(dgms[dim])
    ripscomplex = gd.RipsComplex(points=ptcloud)
    st = ripscomplex.create_simplex_tree(max_dimension=2)
    st.persistence()
    dgm1_gd = st.persistence_intervals_in_dimension(1)
    print(wasserstein_distance(dgms[1].detach().numpy(), dgm1_gd))
    # Testing WassersteinDistance
    dgmx = dgms[0]
    ptcloud = np.random.uniform(size=ptcloud.shape)
    x = torch.tensor(ptcloud, requires_grad=True)
    dgms = rips(torch.cdist(x, x))
    dgmy = dgms[0]
    wass = WassersteinDistance(2)
    print('custom Wasserstein: ' + str(wass(dgmx, dgmy)))
    dgmx_np, dgmy_np = dgmx.detach().numpy(), dgmy.detach().numpy()
    print('GUDHI Wasserstein: ' + str(wasserstein_distance(dgmx_np, dgmy_np, matching=False, order=2, internal_p=1) ** 2))
