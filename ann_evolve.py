"""
ANN evolver.
Trains ANN strategies with an evolutionary algorithm.

Original version by Martin Jones @mojones:
https://gist.github.com/mojones/b809ba565c93feb8d44becc7b93e37c6

Usage:
    ann_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--noise NOISE] [--nmoran NMORAN]
    [--features FEATURES] [--hidden HIDDEN] [--mu_distance DISTANCE]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Starting population size  [default: 10]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 5]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: ann_weights.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --features FEATURES         Number of ANN features [default: 17]
    --hidden HIDDEN             Number of hidden nodes [default: 10]
    --mu_distance DISTANCE      Delta max for weights updates [default: 5]
"""

from itertools import repeat
from multiprocessing import Pool
from operator import itemgetter
import random
from statistics import mean, pstdev

from docopt import docopt
import numpy as np

import axelrod as axl
from axelrod import Actions, flip_action
from axelrod.strategies.ann import ANN, split_weights
from axelrod_utils import Outputer, score_for, prepare_objective

C, D = Actions.C, Actions.D

## Neural network specifics

def num_weights(num_features, num_hidden):
    size = num_features * num_hidden + 2 * num_hidden
    return size

def random_params(num_features, num_hidden):
    size = num_weights(num_features, num_hidden)
    return [random.uniform(-1, 1) for _ in range(size)]


def represent_params(weights):
    """Return a string representing the values of a lookup table dict"""
    return ','.join(map(str, weights))


def params_from_representation(string_id):
    """Return a lookup table dict from a string representing the values"""
    return list(map(float, string_id.split(',')))


def copy_params(weights):
    return list(weights)


def score_single(weights, objective, num_features, num_hidden):
    args = [weights, num_features, num_hidden]
    return (score_for(ANN, args=args, objective=objective), weights)

def score_all(population, pool, objective, num_features, num_hidden):
    results = pool.starmap(
        score_single,
        zip(
            population,
            repeat(objective),
            repeat(num_features),
            repeat(num_hidden)
        )
    )
    return list(results)

## Evolutionary Algorithm

def crossover(weights_collection):
    copies = []
    for i, w1 in enumerate(weights_collection):
        for j, w2 in enumerate(weights_collection):
            if i == j:
                continue
            crosspoint = random.randrange(len(w1))
            new_weights = copy_params(w1[0:crosspoint]) + copy_params(w2[crosspoint:])
            copies.append(new_weights)
    return copies

def mutate(copies, mutation_rate, mutation_distance):
    randoms = np.random.random((len(copies), 190))
    for i, c in enumerate(copies):
        for j in range(len(c)):
            if randoms[i][j] < mutation_rate:
                r = 1 + random.uniform(-1, 1) * mutation_distance
                c[j] = c[j] * r
    return copies


def evolve(starting_population, objective, generations, bottleneck,
           mutation_rate, processes, output_filename, param_args,
           mutation_distance):
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
        copies = mutate(copies, mutation_rate, mutation_distance)

        # The population of tables we want to consider includes the
        # recombined, mutated copies, plus the originals
        population = copies + [copy_params(x[1]) for x in current_bests]
        # Map the population to get a list of (score, table) tuples

        # This list will be sorted by score, best tables first
        results = score_all(population, pool, objective, *param_args)

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

        # if decay:
        #     mutation_rate *= 0.995
        #     mutation_distance *= 0.995

    return current_bests




if __name__ == '__main__':
    arguments = docopt(__doc__, version='ANN Evolver 0.2')
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

    # ANN
    num_features = int(arguments['--features'])
    num_hidden = int(arguments['--hidden'])
    mutation_distance = float(arguments['--mu_distance'])
    param_args = [num_features, num_hidden]

    objective = prepare_objective(name, noise, repetitions, nmoran)

    starting_population = [random_params(*param_args) for _ in range(population)]

    # strategies = axl.short_run_time_strategies

    evolve(starting_population, objective, generations, bottleneck,
           mutation_rate, processes, output_filename, param_args, mutation_distance)
