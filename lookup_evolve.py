"""Lookup Table Evolver

Usage:
    lookup_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--noise NOISE] [--nmoran NMORAN]
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
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --plays PLAYS               Number of recent plays in the lookup table [default: 2]
    --op_plays OP_PLAYS         Number of recent plays in the lookup table [default: 2]
    --op_start_plays OP_START_PLAYS   Number of opponent starting plays in the lookup table [default: 2]
"""

from itertools import repeat
from multiprocessing import Pool
from operator import itemgetter
import random
from statistics import mean, pstdev

from docopt import docopt
import numpy as np

from axelrod import Actions, flip_action
from axelrod.strategies.lookerup import LookerUp, create_lookup_table_keys
from axelrod_utils import Outputer, score_for, prepare_objective

import analyze_data

C, D = Actions.C, Actions.D

## Look up Tables


def random_params(plays, op_plays, op_start_plays):
    """Return randomly-generated lookup tables"""
    # The list of keys for the lookup table is just the product of these three
    # lists
    lookup_table_keys = create_lookup_table_keys(plays, op_plays, op_start_plays)

    # To get a pattern, we just randomly pick between C and D for each key
    pattern = ''.join([random.choice("CD") for _ in lookup_table_keys])

    # zip together the keys and the patterns to give a table
    table = dict(zip(lookup_table_keys, pattern))

    return table


def represent_params(table):
    """Return a string representing the values of a lookup table dict"""
    return ''.join([v for k, v in sorted(table.items())])


def params_from_representation(string_id, keys):
    """Return a lookup table dict from a string representing the values"""
    return dict(zip(keys, string_id))


def copy_params(table):
    return dict(table)


def score_single(table, objective):
    """
    Take a lookup table dict and return a tuple of the score and the table.
    """
    return (score_for(LookerUp, objective=objective, args=[table]), table)


def score_all(tables, pool, objective):
    """Use a multiprocessing Pool to take a bunch of tables and score them"""
    results = list(pool.starmap(score_single, zip(tables, repeat(objective))))
    return list(results)


## Evolutionary Algorithm


def crossover(tables):
    copies = []
    keys = list(sorted(tables[0].keys()))
    # Each table reproduces with each other table to produce one offspring
    for i, t1 in enumerate(tables):
        for j, t2 in enumerate(tables):
            if i == j:
                continue
            # For reproduction, pick a random crossover point
            crosspoint = random.randrange(len(keys))
            new_items = [(k, t1[k]) for k in keys[:crosspoint]]
            new_items += [(k, t2[k]) for k in keys[crosspoint:]]
            new_table = dict(new_items)
            copies.append(new_table)
    return copies


def mutate(copies, mutation_rate):
    randoms = np.random.random((len(copies), len(copies[0].keys())))
    for i, c in enumerate(copies):
        # Flip each value with a probability proportional to the mutation rate
        for j, (history, move) in enumerate(c.items()):
            if randoms[i][j] < mutation_rate:
                c[history] = flip_action(move)
    return copies


def evolve(starting_population, objective, generations, bottleneck,
           mutation_rate, processes, output_filename, param_args):
    """
    The function that does everything. Take a set of starting tables, and in
    each generation:
    - add a bunch more random tables
    - simulate recombination between each pair of tables
    - randomly mutate the current population of tables
    - calculate the fitness function i.e. the average score per turn
    - keep the best individuals and discard the rest
    - write out summary statistics to the output file
    """
    pool = Pool(processes=processes)
    outputer = Outputer(output_filename)

    current_bests = [[0, t] for t in starting_population]

    for generation in range(generations):
        # Because this is a long-running process we'll just keep appending to
        # the output file so we can monitor it while it's running
        print("Starting Generation " + str(generation))

        # The tables at the start of this generation are the best ones from
        # the previous generation (i.e. the second element of each tuple)
        # plus a bunch of random ones
        random_tables = [random_params(*param_args) for _ in range(4)]
        tables_to_copy = [copy_params(x[1]) for x in current_bests]
        tables_to_copy += random_tables

        # Crossover
        copies = crossover(tables_to_copy)
        # Mutate
        copies = mutate(copies, mutation_rate)

        # The population of tables we want to consider includes the
        # recombined, mutated copies, plus the originals
        population = copies + [copy_params(x[1]) for x in current_bests]
        # Map the population to get a list of (score, table) tuples

        # This list will be sorted by score, best tables first
        results = score_all(population, pool, objective)

        # The best tables from this generation become the starting tables
        # for the next generation
        results.sort(key=itemgetter(0), reverse=True)
        current_bests = results[0: bottleneck]

        # get all the scores for this generation
        scores = [score for (score, table) in results]

        # Write the data
        row = [generation, mean(scores), pstdev(scores), results[0][0],
               represent_params(results[0][1])]
        row.extend(results[0][1])
        outputer.write(row)

        print("Generation", generation, "| Best Score:", results[0][0],
              represent_params(results[0][1]))
    return current_bests



if __name__ == '__main__':
    arguments = docopt(__doc__, version='Lookup Evolver 0.2')
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
    noise = float(arguments['--noise'])
    nmoran = int(arguments['--nmoran'])

    # Lookup Tables
    plays = int(arguments['--plays'])
    op_plays = int(arguments['--op_plays'])
    op_start_plays = int(arguments['--op_start_plays'])
    param_args = [plays, op_plays, op_start_plays]

    objective = prepare_objective(name, noise, repetitions, nmoran)

    # Generate random starting population
    starting_population = [random_params(*param_args) for _ in range(population)]

    # else:
    #     keys = create_lookup_table_keys(plays, opp_plays, op_start_plays)
    #     data = analyze_data.read_top_tables(initial_population, starting_pop)
    #     starting_tables = []
    #     for row in data:
    #         starting_tables.append(table_from_id(row[1], keys))
    #     # If insufficient size in data augment with random tables:
    #     if len(starting_tables) < starting_pop:
    #         needed_num = starting_pop - len(starting_tables)
    #         starting_tables = starting_tables + random_params(
    #             plays, opp_plays, op_start_plays, needed_num)

    # Kick off the evolve function
    evolve(starting_population, objective, generations, bottleneck,
           mutation_rate, processes, output_filename, param_args)
