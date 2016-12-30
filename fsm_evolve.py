"""
Finite State Machine Evolver

Usage:
    fsm_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--noise NOISE] [--nmoran NMORAN]
    [--states NUM_STATES]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Starting population size  [default: 10]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 5]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: fsm_tables.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --states NUM_STATES         Number of FSM states [default: 8]
"""

from itertools import repeat
from multiprocessing import Pool
from operator import itemgetter
from random import randrange, choice
from statistics import mean, pstdev

from docopt import docopt
import numpy as np

from axelrod import Actions, flip_action
from axelrod.strategies.finite_state_machines import FSMPlayer
from axelrod_utils import Outputer, score_for, prepare_objective

C, D = Actions.C, Actions.D

## FSM Tables


def random_params(num_states):
    """Return randomly-generated parameters."""
    table = []
    actions = (C, D)
    for j in range(num_states):
        for action in actions:
            next_state = randrange(num_states)
            next_action = choice(actions)
            row = [j, action, next_state, next_action]
            table.append(row)
    return table


def represent_params(table):
    ss = []
    for row in table:
        ss.append("_".join(list(map(str, row))))
    return ":".join(ss)


def params_from_representation(rep):
    table = []
    lines = rep.split(':')
    for line in lines:
        row = line.split('_')
        table.append(row)
    return table


def copy_params(table):
    new = list(map(list, table))
    return new


def score_single(table, objective):
    """
    Take a lookup table dict and return a tuple of the score and the table.
    """
    return (score_for(FSMPlayer, objective=objective, args=[table, 0, 'C']),
            table)


def score_all(tables, pool, objective):
    """Use a multiprocessing Pool to take a bunch of tables and score them"""
    results = list(pool.starmap(score_single, zip(tables, repeat(objective))))
    return list(results)


## Evolutionary Algorithm


def crossover(tables):
    copies = []
    num_states = len(tables[0]) // 2
    # Each table reproduces with each other table to produce one offspring
    for i, t1 in enumerate(tables):
        for j, t2 in enumerate(tables):
            if i == j:
                continue
            # For reproduction, pick a random crossover point
            crosspoint = 2 * randrange(num_states)
            new_table  = copy_params(t1[:crosspoint])
            new_table += copy_params(t2[crosspoint:])
            copies.append(new_table)
    return copies


def mutate(copies, mutation_rate):
    randoms = np.random.random((len(copies), len(copies[0])))
    for i, table in enumerate(copies):
        # Flip each value with a probability proportional to the mutation rate
        for j, row in enumerate(table):
            if randoms[i][j] < mutation_rate:
                row[3] = flip_action(row[3])
        # Swap Nodes
        nodes = len(table) // 2
        n1 = randrange(nodes)
        n2 = randrange(nodes)
        for j, row in enumerate(table):
            if row[1] == n1:
                row[1] = n2
            elif row[1] == n2:
                row[1] = n1
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
    arguments = docopt(__doc__, version='FSM Evolver 0.2')
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

    # FSM
    num_states = int(arguments['--states'])
    param_args = [num_states]

    objective = prepare_objective(name, noise, repetitions, nmoran)

    # Generate random starting population
    starting_population = [random_params(*param_args) for _ in range(population)]

    # Kick off the evolve function
    evolve(starting_population, objective, generations, bottleneck,
           mutation_rate, processes, output_filename, param_args)
