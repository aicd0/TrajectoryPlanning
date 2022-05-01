import numpy as np
from framework.noise import Noise

# [reference] https://github.com/ghliu/pytorch-ddpg/blob/master/random_process.py
# [reference] https://github.com/matthiasplappert/keras-rl/blob/master/rl/random.py

class AnnealedGaussianProcess(Noise):
    def __init__(self, dim, mu, sigma, sigma_min, n_steps_annealing):
        super().__init__(dim)
        self.mu = mu
        self.sigma = sigma
        self.n_steps = 0

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.
            self.c = sigma
            self.sigma_min = sigma

    @property
    def current_sigma(self):
        sigma = max(self.sigma_min, self.m * float(self.n_steps) + self.c)
        return sigma

# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckProcess(AnnealedGaussianProcess):
    def __init__(self, dim, theta, mu=0., sigma=1., dt=1e-2, x0=None, sigma_min=None, n_steps_annealing=1000):
        super().__init__(dim, mu=mu, sigma=sigma, sigma_min=sigma_min, n_steps_annealing=n_steps_annealing)
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.reset()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.current_sigma * np.sqrt(self.dt) * np.random.normal(size=self.dim)
        self.x_prev = x
        self.n_steps += 1
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.dim)