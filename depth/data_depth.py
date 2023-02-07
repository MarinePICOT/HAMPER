from tqdm import tqdm
import logging
import numpy as np
import torch
import os

logger = logging.getLogger(__name__)

def sampled_sphere(K, d):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    mean = torch.zeros(d)
    identity = torch.eye(d)
    dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, scale_tril=identity)
    U = dist.rsample(sample_shape=(K,))
    return F.normalize(U)


class DataDepth:
    def __init__(self, K, dataset):
        self.K = K  # as a starter you can set it to 10 times the dimension
        self.dataset = dataset

    def halfspace_mass(self, X, layer, num_class, psi=32, lamb=0.5, X_test=None, U=None):
        n, d = X.shape
        Score = np.zeros(n)

        if U is None:
            U = sampled_sphere(self.K, d)
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X).cuda()
        if not isinstance(U.T, torch.Tensor):
            M = torch.matmul(X, torch.tensor(U.T).cuda()).cpu().numpy()
        else:
            M = torch.matmul(X.cuda(), torch.tensor(torch.transpose(U)).cuda()).cpu().numpy()

        mass_left = np.zeros(self.K)
        mass_right = np.zeros(self.K)
        s = np.zeros(self.K)
        for i in range(self.K):
            try:
                subsample = np.random.choice(np.arange(n), size=psi, replace=False)
            except:
                subsample = np.random.choice(np.arange(n), size=n - 1, replace=False)
            SP = M[subsample, i]
            max_i = np.max(SP)
            min_i = np.min(SP)
            mid_i = (max_i + min_i) / 2
            s[i] = (
                    lamb * (max_i - min_i) * np.random.uniform()
                    + mid_i
                    - lamb * (max_i - min_i) / 2
            )
            mass_left[i] = (SP < s[i]).sum() / psi
            mass_right[i] = (SP > s[i]).sum() / psi
            Score += mass_left[i] * (M[:, i] < s[i]) + mass_right[i] * (M[:, i] > s[i])

        if X_test is None:
            return Score / self.K
        else:
            Score_test = np.zeros(len(X_test))
            if not isinstance(X_test, torch.Tensor):
                X_test = torch.tensor(X_test).cuda()
            if not isinstance(U.T, torch.Tensor):
                M_test = torch.matmul(X_test, torch.tensor(U.T).cuda()).cpu().numpy()
            else:
                M_test = torch.matmul(X_test.cuda(), torch.transpose(U).cuda()).cpu().numpy()
            for i in range(self.K):
                Score_test += mass_left[i] * (M_test[:, i] < s[i]) + mass_right[i] * (
                        M_test[:, i] > s[i]
                )
            return Score_test / self.K
