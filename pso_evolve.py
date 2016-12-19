"""
Particle Swarm strategy training code.

Original version by Georgios Koutsovoulos @GDKO :
  https://gist.github.com/GDKO/60c3d0fd423598f3c4e4
Based on Martin Jones @mojones original LookerUp code
"""

from itertools import product

from pyswarm import pso

import axelrod as axl
from axelrod import Gambler, init_args
from axelrod_utils import score_single


class TestGambler(Gambler):
    """
    A LookerUp strategy that uses pattern supplied when initialised.
    """

    name = "TestGambler"

    @init_args
    def __init__(self, pattern):
        plays = 2
        opponent_start_plays = 2

        # Generate the list of possible tuples, i.e. all possible combinations
        # of m actions for me, m actions for opponent, and n starting actions
        # for opponent.
        self_histories = [''.join(x) for x in product('CD', repeat=plays)]
        other_histories = [''.join(x) for x in product('CD', repeat=plays)]
        opponent_starts = [''.join(x) for x in
                           product('CD', repeat=opponent_start_plays)]
        lookup_table_keys = list(product(opponent_starts, self_histories,
                                         other_histories))

        # Zip together the keys and the action pattern to get the lookup table.
        lookup_table = dict(zip(lookup_table_keys, pattern))
        Gambler.__init__(self, lookup_table=lookup_table)


def score_for_pattern(my_strategy_factory, pattern, iterations=200):
    """
    Given a function that will return a strategy,
    calculate the average score per turn
    against all ordinary strategies. If the
    opponent is classified as stochastic, then
    run 100 repetitions and take the average to get
    a good estimate.
    """
    scores_for_all_opponents = []
    strategies = [s for s in axl.strategies
                  if not s().classifier['long_run_time']]
    for opponent in strategies:

        # Decide whether we need to sample or not
        if opponent().classifier['stochastic']:
            repetitions = 100
        else:
            repetitions = 1
        scores_for_this_opponent = []

        # Calculate an average for this opponent
        for _ in range(repetitions):
            me = my_strategy_factory(pattern)
            other = opponent()
            # Make sure that both players know what length the match will be
            me.set_match_attributes(length=iterations)
            other.set_match_attributes(length=iterations)

            scores_for_this_opponent.append(score_single(me, other, iterations))

        mean_vs_opponent = sum(scores_for_this_opponent) / len(scores_for_this_opponent)
        scores_for_all_opponents.append(mean_vs_opponent)

    # Calculate the average for all opponents
    overall_average_score = sum(scores_for_all_opponents) / len(scores_for_all_opponents)
    return overall_average_score


def optimizepso(x):
    return -score_for_pattern(TestGambler, x)

if __name__ == "__main__":
    lb = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ub = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    # There is a multiprocessing version (0.7) of pyswarm available at
    # https://github.com/tisimst/pyswarm
    # Pip installs version 0.6
    xopt, fopt = pso(optimizepso, lb, ub, swarmsize=100, maxiter=20,
                     debug=True, phip=0.8, phig=0.8, omega=0.8)
    print(xopt)
    print(fopt)
