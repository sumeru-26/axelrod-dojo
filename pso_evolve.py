"""
Particle Swarm strategy training code.

Original version by Georgios Koutsovoulos @GDKO :
  https://gist.github.com/GDKO/60c3d0fd423598f3c4e4
Based on Martin Jones @mojones original LookerUp code
"""

import axelrod

from pyswarm import pso


class Gambler(Player):

    name = 'Gambler'
    classifier = {
        'memory_depth': float('inf'),
        'stochastic': True,
        'makes_use_of': set(),
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    @init_args
    def __init__(self, lookup_table=None):
        """
        If no lookup table is provided to the constructor, then use the TFT one.
        """
        Player.__init__(self)

        if not lookup_table:
            lookup_table = {
            ('', 'C', 'D') : 0,
            ('', 'D', 'D') : 0,
            ('', 'C', 'C') : 1,
            ('', 'D', 'C') : 1,
        }

        self.lookup_table = lookup_table
        # Rather than pass the number of previous turns (m) to consider in as a
        # separate variable, figure it out. The number of turns is the length
        # of the second element of any given key in the dict.
        self.plays = len(list(self.lookup_table.keys())[0][1])
        # The number of opponent starting actions is the length of the first
        # element of any given key in the dict.
        self.opponent_start_plays = len(list(self.lookup_table.keys())[0][0])
        # If the table dictates to ignore the opening actions of the opponent
        # then the memory classification is adjusted
        if self.opponent_start_plays == 0:
            self.classifier['memory_depth'] = self.plays

        # Ensure that table is well-formed
        for k, v in lookup_table.items():
            if (len(k[1]) != self.plays) or (len(k[0]) != self.opponent_start_plays):
                raise ValueError("All table elements must have the same size")


    def strategy(self, opponent):
        # If there isn't enough history to lookup an action, cooperate.
        if len(self.history) < max(self.plays, self.opponent_start_plays):
            return C
        # Count backward m turns to get my own recent history.
        history_start = -1 * self.plays
        my_history = ''.join(self.history[history_start:])
        # Do the same for the opponent.
        opponent_history = ''.join(opponent.history[history_start:])
        # Get the opponents first n actions.
        opponent_start = ''.join(opponent.history[:self.opponent_start_plays])
        # Put these three strings together in a tuple.
        key = (opponent_start, my_history, opponent_history)
        # Look up the action associated with that tuple in the lookup table.
        action = float(self.lookup_table[key])
        return random_choice(action)



class TestGambler(Gambler):
    """
    A LookerUp strategy that uses pattern supplied when initialised.
    """

    name = "TestGambler"

    def __init__(self,pattern):
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


def score_for_pattern(my_strategy_factory,pattern, iterations=200):
    """
    Given a function that will return a strategy,
    calculate the average score per turn
    against all ordinary strategies. If the
    opponent is classified as stochastic, then
    run 100 repetitions and take the average to get
    a good estimate.
    """
    scores_for_all_opponents = []
    for opponent in axelrod.ordinary_strategies:

        # decide whether we need to sample or not
        if opponent.classifier['stochastic']:
            repetitions = 100
        else:
            repetitions = 1
        scores_for_this_opponent = []

        # calculate an average for this opponent
        for _ in range(repetitions):
            me = my_strategy_factory(pattern)
            other = opponent()
            # make sure that both players know what length the match will be
            me.set_tournament_attributes(length=iterations)
            other.set_tournament_attributes(length=iterations)

            scores_for_this_opponent.append(score_single(me, other, iterations))

        mean_vs_opponent = sum(scores_for_this_opponent) / len(scores_for_this_opponent)
        scores_for_all_opponents.append(mean_vs_opponent)

    # calculate the average for all opponents
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

    # The parameters of phip, phig and omega will lead to slower conversion
    xopt, fopt = pso(optimizepso, lb, ub, swarmsize=100, maxiter=20, processes=60,
                     debug=True,
                     phip=0.8, phig=0.8, omega=0.8)