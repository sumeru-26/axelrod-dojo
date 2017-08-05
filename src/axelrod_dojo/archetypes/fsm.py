import random
from random import randrange, choice

import numpy as np
from axelrod import Action, FSMPlayer
from axelrod.action import UnknownActionError

from axelrod_dojo.utils import Params, Population, prepare_objective

C, D = Action.C, Action.D


def copy_lists(rows):
    new_rows = list(map(list, rows))
    return new_rows


class FSMParams(Params):

    def __init__(self, num_states, mutation_rate=None, rows=None,
                 initial_state=0, initial_action=C):
        self.PlayerClass = FSMPlayer
        self.num_states = num_states
        if mutation_rate is None:
            self.mutation_rate = 1 / (2 * num_states)
        else:
            self.mutation_rate = mutation_rate
        if rows is None:
            self.randomize()
        else:
            # Make sure to copy the lists
            self.rows = copy_lists(rows)
        self.initial_state = initial_state
        self.initial_action = initial_action


    def player(self):
        player = self.PlayerClass(self.rows, self.initial_state,
                                  self.initial_action)
        return player

    def copy(self):
        return FSMParams(self.num_states, self.mutation_rate,
                         self.rows, self.initial_state, self.initial_action)

    @staticmethod
    def random_params(num_states):
        rows = []
        actions = (C, D)
        for j in range(num_states):
            for action in actions:
                next_state = randrange(num_states)
                next_action = choice(actions)
                row = [j, action, next_state, next_action]
                rows.append(row)
        initial_state = randrange(num_states)
        initial_action = choice([C, D])
        return rows, initial_state, initial_action

    def randomize(self):
        rows, initial_state, initial_action = self.random_params(self.num_states)
        self.rows = rows
        self.initial_state = initial_state
        self.initial_action = initial_action

    @staticmethod
    def mutate_rows(rows, mutation_rate):
        randoms = np.random.random(len(rows))
        # Flip each value with a probability proportional to the mutation rate
        for i, row in enumerate(rows):
            if randoms[i] < mutation_rate:
                row[3] = row[3].flip()
        # Swap Two Nodes?
        if random.random() < 0.5:
            nodes = len(rows) // 2
            n1 = randrange(nodes)
            n2 = randrange(nodes)
            for j, row in enumerate(rows):
                if row[1] == n1:
                    row[1] = n2
                elif row[1] == n2:
                    row[1] = n1
        return rows

    def mutate(self):
        self.rows = self.mutate_rows(self.rows, self.mutation_rate)
        if random.random() < self.mutation_rate / 10:
            self.initial_action = self.initial_action.flip()
        if random.random() < self.mutation_rate / (10 * self.num_states):
            self.initial_state = randrange(self.num_states)
        # Change node size?

    @staticmethod
    def crossover_rows(rows1, rows2):
        num_states = len(rows1) // 2
        crosspoint = 2 * randrange(num_states)
        new_rows = copy_lists(rows1[:crosspoint])
        new_rows += copy_lists(rows2[crosspoint:])
        return new_rows

    def crossover(self, other):
        # Assuming that the number of states is the same
        new_rows = self.crossover_rows(self.rows, other.rows)
        return FSMParams(self.num_states, self.mutation_rate, new_rows,
                         self.initial_state, self.initial_action)

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
        return cls(num_states, 0.1, rows, initial_state, initial_action)
