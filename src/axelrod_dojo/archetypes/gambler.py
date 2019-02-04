import random
import numpy as np

from axelrod import Action, Gambler
from axelrod.strategies.lookerup import create_lookup_table_keys
from axelrod.strategies.lookerup import Plays

from .params import Params

C, D = Action.C, Action.D


class GamblerParams(Params):
    """
    A class for a Gambler archetype used to train Gambler strategies using 
    various optimization algorithms. 
    
    More details for the Gambler player can be found in the axelrod 
    documentation: http://axelrod.readthedocs.io/ 
    
    Parameters
    ----------
    plays : integer
        The number of player's last moves to remember
    op_plays : integer
        The number of opponent's last moves to remember
    op_start_plays : integer
        The number of opponent's initial moves to remember   
    """
    def __init__(self, plays, op_plays, op_start_plays, pattern=None,
                 mutation_probability=0):
        self.PlayerClass = Gambler
        self.plays = plays
        self.op_plays = op_plays
        self.op_start_plays = op_start_plays
        self.mutation_probability = mutation_probability

        if pattern is None:
            self.randomize()
        else:
            # Make sure to copy the lists
            self.pattern = pattern

    def player(self):
        """Turns the attribute vector in to a Gambler player instance."""
        parameters = Plays(self_plays=self.plays, op_plays=self.op_plays,
                           op_openings=self.op_start_plays)
        player = self.PlayerClass(pattern=self.pattern, parameters=parameters)
        return player

    def copy(self):
        return GamblerParams(plays=self.plays, op_plays=self.op_plays,
                             op_start_plays=self.op_start_plays, pattern=self.pattern)

    def receive_vector(self, vector):
        """Receives a vector and updates the player's pattern.
          Ignores extra parameters."""
        self.pattern = vector

    def create_vector_bounds(self):
        """Creates the bounds for the decision variables.  Ignores extra
        parameters."""
        size = len(self.pattern)
        lb = [0.0] * size
        ub = [1.0] * size

        return lb, ub

    @staticmethod
    def mutate_pattern(pattern, mutation_probability):
        randoms = np.random.random(len(pattern))

        for i, _ in enumerate(pattern):
            if randoms[i] < mutation_probability:
                ep = random.uniform(-1, 1) / 4
                pattern[i] += ep
                if pattern[i] < 0:
                    pattern[i] = 0
                if pattern[i] > 1:
                    pattern[i] = 1
        return pattern

    def mutate(self):
        self.pattern = self.mutate_pattern(self.pattern, self.mutation_probability)

    def crossover(self, other):
        pattern1 = self.pattern
        pattern2 = other.pattern

        cross_point = int(random.randint(0, len(pattern1)))
        offspring_pattern = pattern1[:cross_point] + pattern2[cross_point:]
        return GamblerParams(plays=self.plays, op_plays=self.op_plays,
                             op_start_plays=self.op_start_plays,
                             pattern=offspring_pattern,
                             mutation_probability=self.mutation_probability)

    @staticmethod
    def random_params(plays, op_plays, op_start_plays):
        keys = create_lookup_table_keys(plays, op_plays, op_start_plays)

        pattern = [random.random() for _ in keys]
        return pattern

    def randomize(self):
        pattern = self.random_params(
            self.plays, self.op_plays, self.op_start_plays)
        self.pattern = pattern

    def __repr__(self):
        return "{}:{}:{}:{}".format(
            self.plays,
            self.op_plays,
            self.op_start_plays,
            '|'.join(str(v) for v in self.pattern)
        )
