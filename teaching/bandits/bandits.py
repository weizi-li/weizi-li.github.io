from __future__ import division

import time
import numpy as np


class Bandit(object):

    def generate_reward(self, i):
        raise NotImplementedError


class BernoulliBandit(Bandit):

    def __init__(self, n, prob=None):
        assert prob is None or len(prob) == n
        self.n = n
        if prob is None:
            np.random.seed(int(time.time()))
            self.prob = [np.random.random() for _ in range(self.n)]
        else:
            self.prob = prob

        self.best_prob = max(self.prob)

    def generate_reward(self, i):
        # The player selected the i-th machine.
        if np.random.random() < self.prob[i]:
            return 1
        else:
            return 0
