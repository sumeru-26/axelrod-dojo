"""Lookup Table Evolver

Usage:
    lookup_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]
    [--plays PLAYS] [--op_plays OP_PLAYS] [--op_start_plays OP_START_PLAYS]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Starting population size  [default: 10]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 5]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: lookup_tables.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --plays PLAYS               Number of recent plays in the lookup table [default: 2]
    --op_plays OP_PLAYS         Number of recent plays in the lookup table [default: 2]
    --op_start_plays OP_START_PLAYS   Number of opponent starting plays in the lookup table [default: 2]
"""

import random
from random import choice

from docopt import docopt
import numpy as np

from axelrod import Actions, flip_action
from axelrod.strategies.lookerup import LookerUp, create_lookup_table_keys

from evolve_utils import Params, Population, prepare_objective

C, D = Actions.C, Actions.D


class LookerUpParams(Params):

    def __init__(self, plays, op_plays, op_start_plays, initial_actions=None,
                 mutation_rate=None, table=None):
        self.PlayerClass = LookerUp
        self.plays = plays
        self.op_plays = op_plays
        self.op_start_plays = op_start_plays

        if not initial_actions:
            table_depth = max(self.plays, self.op_plays, self.op_start_plays)
            initial_actions = [choice(C, D) for _ in range(table_depth)]
        self.initial_actions = initial_actions

        if mutation_rate is None:
            keys = create_lookup_table_keys(plays, op_plays, op_start_plays)
            self.mutation_rate = 2 / len(keys)
        else:
            self.mutation_rate = mutation_rate
        if table is None:
            self.randomize()
        else:
            # Make sure to copy the lists
            self.table = dict(table)

    def player(self):
        player = self.PlayerClass(self.table, self.initial_actions)
        return player

    def copy(self):
        return LookerUpParams(self.plays, self.op_plays, self.op_start_plays,
                              list(self.initial_actions), self.mutation_rate,
                              dict(self.table))

    @staticmethod
    def random_params(plays, op_plays, op_start_plays):
        keys = create_lookup_table_keys(plays, op_plays, op_start_plays)
        # To get a pattern, we just randomly pick between C and D for each key
        pattern = ''.join([random.choice([C, D]) for _ in keys])
        table = dict(zip(keys, pattern))
        return table

    def randomize(self):
        table = self.random_params(
            self.plays, self.op_plays, self.op_start_plays)
        self.table = table

    @staticmethod
    def mutate_table(table, mutation_rate):
        randoms = np.random.random(len(table.keys()))
        # Flip each value with a probability proportional to the mutation rate
        for i, (history, move) in enumerate(table.items()):
            if randoms[i] < mutation_rate:
                table[history] = flip_action(move)
        return table

    def mutate(self):
        self.table = self.mutate_table(self.table, self.mutation_rate)
        # Add in starting moves
        for i in range(len(self.initial_actions)):
            r = random.random()
            if r < 0.05:
                self.initial_actions[i] = flip_action(self.initial_actions[i])

    @staticmethod
    def crossover_tables(table1, table2):
        keys = list(sorted(table1.keys()))
        crosspoint = random.randrange(len(keys))
        new_items = [(k, table1[k]) for k in keys[:crosspoint]]
        new_items += [(k, table2[k]) for k in keys[crosspoint:]]
        new_table = dict(new_items)
        return new_table

    def crossover(self, other):
        # Assuming that the number of states is the same
        new_table = self.crossover_tables(self.table, other.table)
        return LookerUpParams(self.plays, self.op_plays, self.op_start_plays,
                              list(self.initial_actions), self.mutation_rate,
                              new_table)

    def __repr__(self):
        return "{}:{}:{}:{}:{}".format(
            self.plays,
            self.op_plays,
            self.op_start_plays,
            ''.join(self.initial_actions),
            ''.join([v for k, v in sorted(self.table.items())])
        )

    @classmethod
    def parse_repr(cls, s):
        elements = s.split(':')
        plays, op_plays, op_start_plays, initial_actions, pattern = elements
        keys = create_lookup_table_keys(plays, op_plays, op_start_plays)
        table = dict(zip(keys, pattern))
        return cls(plays, op_plays, op_start_plays,
                   initial_actions=initial_actions, table=table)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Lookup Evolver 0.3')
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

    # Lookup Tables
    plays = int(arguments['--plays'])
    op_plays = int(arguments['--op_plays'])
    op_start_plays = int(arguments['--op_start_plays'])
    table_depth = max(plays, op_plays, op_start_plays)
    initial_actions = [C] * table_depth
    param_args = [plays, op_plays, op_start_plays, initial_actions,
                  mutation_rate]

    objective = prepare_objective(name, turns, noise, repetitions, nmoran)
    population = Population(LookerUpParams, param_args, population, objective,
                            output_filename, bottleneck, processes=processes)
    population.run(generations)
