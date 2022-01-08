import matplotlib

import matplotlib.pyplot as plt
import numpy as np

from bandits import BernoulliBandit
from solvers import Solver, EpsilonGreedy


def plot_results(solvers, solver_names):
    """
    Plot the results by multi-armed bandit solvers.
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    b = solvers[0].bandit

    fig = plt.figure()
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot()

    # Sub.fig. 1: Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    plt.show()


def experiment(K, N):
    """
    Solving a Bernoulli bandit problem with K slot machines
    Args:
        K (int): number of slot machines.
        N (int): number of steps to try.
    """

    b = BernoulliBandit(K)
    print("True reward probabilities:\n", b.prob)
    print("The best machine is {} with reward probability: {}".format(
        max(range(K), key=lambda i: b.prob[i]), max(b.prob)))

    test_solvers = [
        EpsilonGreedy(b, 0.01),
    ]
    names = [
        r'$\epsilon$' + '-Greedy'
    ]

    for s in test_solvers:
        s.run(N)

    plot_results(test_solvers, names)


if __name__ == '__main__':
    experiment(10, 5000)
