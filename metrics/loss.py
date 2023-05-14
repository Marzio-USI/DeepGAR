import torch.nn as nn
import torch
from distributions.distributions import Distribution, GaussianDistribution, StudentTDistribution


class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, pred, target):
        return torch.sqrt(torch.mean((pred - target) ** 2))

class RMSE_paper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred, target, n_nodes):
        # pred and target have shape (8, 1)

        nom = torch.sqrt(torch.sum((target - pred)**2)/n_nodes)
        denom = torch.sum(torch.abs(target))/ n_nodes 
        return nom / denom


class ND_paper(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, pred, target):
        a = torch.sum(torch.abs(target - pred))
        b = torch.sum(torch.abs(target))
        return a / b


class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, pred, target):
        return torch.mean(torch.abs(pred - target))


class NLL(nn.Module):
    """
    This is the negative log-likelihood loss function for a distribution passed in as the first argument.
    """

    def __init__(self, distribution: Distribution):
        super(NLL, self).__init__()
        self.distribution = distribution

    def forward(self, mu, sigma, target):
        dist = self.distribution.generate_dist(mu, sigma)
        likelihood = dist.log_prob(target)
        return -torch.mean(likelihood)
