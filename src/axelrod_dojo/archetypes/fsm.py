import random
import itertools
from random import randrange, choice

import numpy as np
from axelrod import Action, FSMPlayer
from axelrod.action import UnknownActionError

from axelrod_dojo.utils import Params

C, D = Action.C, Action.D


def copy_lists(rows):
    new_rows = list(map(list, rows))
    return new_rows


class FSMParams(Params):

    def __init__(self, num_states, rows=None,
                 initial_state=0, initial_action=C,
                 mutation_probability=0):
        self.PlayerClass = FSMPlayer
        self.num_states = num_states
        self.mutation_probability = mutation_probability
        if rows is None:
            self.randomize()
        else:
            # Make sure to copy the lists
            self.rows = copy_lists(rows)
        self.initial_state = initial_state
        self.initial_action = initial_action

    @staticmethod
    def repr_rows(rows):
        ss = []
        for row in rows:
            ss.append("_".join(list(map(str, row))))
        return ":".join(ss)

    def __repr__(self):
        return "{}:{}:{}".format(
            self.initial_state,
            self.initial_action,
            self.repr_rows(self.rows)
        )

    @classmethod
    def parse_repr(cls, s):
        rows = []
        lines = s.split(':')
        initial_state = int(lines[0])
        initial_action = Action.from_char(lines[1])

        for line in lines[2:]:
            row = []

            for element in line.split('_'):
                try:
                    row.append(Action.from_char(element))
                except UnknownActionError:
                    row.append(int(element))

            rows.append(row)
        num_states = len(rows) // 2
        return cls(num_states, rows, initial_state, initial_action)

    def receive_vector(self, vector):
        """
        Read a serialized vector into the set of FSM parameters (less initial
        state).  Then assign those FSM parameters to this class instance.

        The vector has three parts. The first is used to define the next state 
        (for each of the player's states - for each opponents action).

        The second part is the player's next moves (for each state - for 
        each opponent's actions).

        Finally, a probability to determine the player's first move.
        """
        state_scale = vector[:self.num_states * 2]
        next_states = [int(s * (self.num_states - 1)) for s in state_scale]
        actions = vector[self.num_states * 2: -1]
        
        self.initial_action = C if round(vector[-1]) == 0 else D
        self.initial_state = 1

        self.rows = []
        for i, (initial_state, action) in enumerate(
                itertools.product(range(self.num_states), [C, D])):
            next_action = C if round(actions[i]) == 0 else D
            self.rows.append([initial_state, action, next_states[i], next_action])

    def create_vector_bounds(self):
        """Creates the bounds for the decision variables."""
        size = len(self.rows) * 2 + 1

        lb = [0] * size
        ub = [1] * size

        return lb, ub
