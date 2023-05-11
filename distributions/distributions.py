import warnings
from typing import Any

import torch


class Distribution:
    def __init__(self):
        pass

    def generate_dist(self, mu, sigma) -> Any:
        pass

    def simple_sample(self, mu, sigma):
        dist = self.generate_dist(mu, sigma)
        return dist.sample()

    def monte_carlo_sampling(self, mu, sigma, n_samples):
        dist = self.generate_dist(mu, sigma)
        samples = dist.sample((n_samples,))
        median = torch.median(samples)[0]
        return median

    def run(self, x):
        print(x)


class GaussianDistribution(Distribution):
    def __init__(self):
        super(GaussianDistribution, self).__init__()

    def generate_dist(self, mu, sigma):
        return torch.distributions.normal.Normal(mu, sigma)

    def monte_carlo_sampling(self, mu, sigma, n_samples):
        warnings.warn('Monte Carlo sampling for Gaussian distribution just returns the mean (mu)')
        return mu


class StudentTDistribution(Distribution):
    def __init__(self, df):
        super(StudentTDistribution, self).__init__()
        self.df = df

    def generate_dist(self, mu, sigma):
        return torch.distributions.studentT.StudentT(self.df, mu, sigma)

    def monte_carlo_sampling(self, mu, sigma, n_samples):
        return super().monte_carlo_sampling(mu, sigma, n_samples)
