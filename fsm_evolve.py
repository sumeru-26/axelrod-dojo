"""
Finite State Machine Evolver

Usage:
    fsm_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]
    [--states NUM_STATES]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Population size  [default: 40]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 10]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: fsm_tables.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --states NUM_STATES         Number of FSM states [default: 8]
"""

import random
from random import randrange, choice

from docopt import docopt
import numpy as np

from axelrod import Actions, flip_action
from axelrod.strategies.finite_state_machines import FSMPlayer

from evolve_utils import Params, Population, prepare_objective

C, D = Actions.C, Actions.D


def copy_lists(rows):
    new_rows = list(map(list, rows))
    return new_rows


class FSMParams(Params):

    def __init__(self, num_states, mutation_rate=None, rows=None,
                 initial_state=0, initial_action=C):
        self.PlayerClass = FSMPlayer
        # Initialize to "zero" state?
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
                row[3] = flip_action(row[3])
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
            self.initial_action = flip_action(self.initial_action)
        if random.random() < self.mutation_rate / (10 * self.num_states):
            self.initial_state = randrange(self.num_states)
        # return self
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
        lines = rep.split(':')
        initial_state = int(lines[0])
        initial_action = lines[1]

        for line in lines[2:]:
            row = line.split('_')
            rows.append(row)
        num_states = len(rows) // 2
        return cls(num_states, rows, initial_state, initial_action)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='FSM Evolver 0.3')
    print(arguments)
    processes = int(arguments['--processes'])

    # Vars for the genetic algorithm
    population = int(arguments['--population'])
    mutation_rate = float(arguments['--mu'])
    generations = int(arguments['--generations'])
    bottleneck = int(arguments['--bottleneck'])
    output_filename = arguments['--output']

    # Objective
    name = str(arguments['--objective'])
    repetitions = int(arguments['--repetitions'])
    turns = int(arguments['--turns'])
    noise = float(arguments['--noise'])
    nmoran = int(arguments['--nmoran'])

    # FSM
    num_states = int(arguments['--states'])
    param_args = [num_states, mutation_rate]

    objective = prepare_objective(name, turns, noise, repetitions, nmoran)
    population = Population(FSMParams, param_args, population, objective,
                            output_filename, bottleneck, processes=processes)
    population.run(generations)
