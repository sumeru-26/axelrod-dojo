import random
from random import randrange, uniform

from axelrod.ultimatum import DoubleThresholdsPlayer

from .params import Params


def copy_lists(rows):
    new_rows = list(map(list, rows))
    return new_rows


class DoubleThresholdsPlayerParams(Params):
    def __init__(self, lower_offer=None, upper_offer=None, lower_accept=None,
                 upper_accept=None, mutation_probability=0.1):
        self.PlayerClass = DoubleThresholdsPlayer
        if not lower_offer:
            self.randomize()
        else:
            self.lower_offer = lower_offer
            self.upper_offer = upper_offer
            self.lower_accept = lower_accept
            self.upper_accept = upper_accept
        self.mutation_probability = mutation_probability
        self.order_intervals()

    def player(self):
        player = self.PlayerClass(self.lower_offer, self.upper_offer,
                                  self.lower_accept, self.upper_accept)
        return player

    def copy(self):
        return DoubleThresholdsPlayerParams(
            self.lower_offer, self.upper_offer, self.lower_accept,
            self.upper_accept, self.mutation_probability)

    @staticmethod
    def random_params():
        lower_offer = random.random()
        upper_offer = uniform(lower_offer, 1)
        lower_accept = random.random()
        upper_accept = uniform(lower_accept, 1)

        return lower_offer, upper_offer, lower_accept, upper_accept

    def randomize(self):
        lower_offer, upper_offer, lower_accept, upper_accept = self.random_params()
        self.lower_offer = lower_offer
        self.upper_offer = upper_offer
        self.lower_accept = lower_accept
        self.upper_accept = upper_accept

    @staticmethod
    def normalize_range(x):
        if x < 0:
            return 0
        if x > 1:
            return 1
        return x

    def order_intervals(self):
        if self.upper_offer < self.lower_offer:
            self.lower_offer, self.upper_offer = self.upper_offer, self.lower_offer
        if self.upper_accept < self.lower_accept:
            self.lower_accept, self.upper_accept = self.upper_accept, self.lower_accept

    def mutate(self, step=0.2):
        if 4 * random.random() <= self.mutation_probability:
            p = uniform(-step, step)
            self.lower_offer = self.normalize_range((1 + p) * self.lower_offer)
        if 4 * random.random() <= self.mutation_probability:
            p = uniform(-step, step)
            self.upper_offer = self.normalize_range((1 + p) * self.upper_offer)

        if 4 * random.random() <= self.mutation_probability:
            p = uniform(-step, step)
            self.lower_accept = self.normalize_range((1 + p) * self.lower_accept)
        if 4 * random.random() <= self.mutation_probability:
            p = uniform(-step, step)
            self.upper_accept = self.normalize_range((1 + p) * self.upper_accept)

        self.order_intervals()

    @staticmethod
    def crossover_rows(row1, row2):
        crosspoint = randrange(1, len(row1)-1)
        new_rows = list(row1[:crosspoint])
        new_rows += list(row2[crosspoint:])
        return new_rows

    def crossover(self, other):
        row1 = [self.lower_offer, self.upper_offer, self.lower_accept,
                self.upper_accept]
        row2 = [other.lower_offer, other.upper_offer, other.lower_accept,
                other.upper_accept]
        new_rows = self.crossover_rows(row1, row2)
        return DoubleThresholdsPlayerParams(*new_rows)

    @staticmethod
    def repr_rows(rows):
        ss = []
        for row in rows:
            ss.append("_".join(list(map(str, row))))
        return ":".join(ss)

    def __repr__(self):
        return "{}:{}:{}:{}".format(
            self.lower_offer, self.upper_offer, self.lower_accept,
            self.upper_accept
        )

    @classmethod
    def parse_repr(cls, s):
        lower_offer, upper_offer, lower_accept, upper_accept = list(map(float, s.split(':')))
        return cls(lower_offer, upper_offer, lower_accept, upper_accept)

    # def receive_vector(self, vector):
    #     """
    #     Read a serialized vector into the set of FSM parameters (less initial
    #     state).  Then assign those FSM parameters to this class instance.
    #
    #     The vector has three parts. The first is used to define the next state
    #     (for each of the player's states - for each opponents action).
    #
    #     The second part is the player's next moves (for each state - for
    #     each opponent's actions).
    #
    #     Finally, a probability to determine the player's first move.
    #     """
    #     state_scale = vector[:self.num_states * 2]
    #     next_states = [int(s * (self.num_states - 1)) for s in state_scale]
    #     actions = vector[self.num_states * 2: -1]
    #
    #     self.initial_action = C if round(vector[-1]) == 0 else D
    #     self.initial_state = 1
    #
    #     self.rows = []
    #     for i, (initial_state, action) in enumerate(
    #             itertools.product(range(self.num_states), [C, D])):
    #         next_action = C if round(actions[i]) == 0 else D
    #         self.rows.append(
    #             [initial_state, action, next_states[i], next_action])
    #
    # def create_vector_bounds(self):
    #     """Creates the bounds for the decision variables."""
    #     size = len(self.rows) * 2 + 1
    #
    #     lb = [0] * size
    #     ub = [1] * size
    #
    #     return lb, ub
