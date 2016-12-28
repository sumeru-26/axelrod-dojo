"""
Training ANN strategies with an evolutionary algorithm.

Original code by Martin Jones @mojones:
https://gist.github.com/mojones/b809ba565c93feb8d44becc7b93e37c6

Usage:
    ann_evolve.py [-h] [-g GENERATIONS] [-u MUTATION_RATE] [-b BOTTLENECK]
    [-d MUTATION_DISTANCE] [-i PROCESSORS] [-o OUTPUT_FILE]
    [-k STARTING_POPULATION] [-n NOISE]

Options:
    -h --help                    show this
    -g GENERATIONS               how many generations to run the program for [default: 1000]
    -u MUTATION_RATE             mutation rate i.e. probability that a given value will flip [default: 0.4]
    -d MUTATION_DISTANCE         amount of change a mutation will cause [default: 5]
    -b BOTTLENECK                number of individuals to keep from each generation [default: 5]
    -i PROCESSORS                number of processors to use [default: 4]
    -o OUTPUT_FILE               file to write statistics to [default: weights.csv]
    -k STARTING_POPULATION       starting population size for the simulation [default: 10]
    -n NOISE                     match noise [default: 0.0]
"""

import csv
from itertools import repeat
from multiprocessing import Pool
import os
import random
from statistics import mean, pstdev

from docopt import docopt
import numpy as np

import axelrod as axl
from axelrod.strategies.ann import ANN, split_weights
from axelrod_utils import score_for, objective_match_score, objective_match_moran_win

## Neural network specifics

def get_random_weights(number):
    return [random.uniform(-1, 1) for _ in range(number)]

def score_weights(weights, strategies, noise, input_values=17,
                  hidden_layer_size=10):
    in2h, h2o, bias = split_weights(weights, input_values, hidden_layer_size)
    args = [in2h, h2o, bias]
    return (score_for(ANN, args=args, opponents=strategies, noise=noise,
                      objective=objective_match_score), weights)

def score_all_weights(population, strategies, noise, hidden_layer_size=10):
    results = pool.starmap(score_weights,
                           zip(population, repeat(strategies), repeat(noise),
                               repeat(10), repeat(hidden_layer_size)))
    return list(sorted(results, reverse=True))

## Evolutionary Algorithm

def crossover(weights_collection):
    copies = []
    for i, w1 in enumerate(weights_collection):
        for j, w2 in enumerate(weights_collection):
            if i == j:
                continue
            crosspoint = random.randrange(len(w1))
            new_weights = list(w1[0:crosspoint]) + list(w2[crosspoint:])
            copies.append(new_weights)
    return copies

def mutate(copies, mutation_rate):
    randoms = np.random.random((len(copies), 190))
    for i, c in enumerate(copies):
        for j in range(len(c)):
            if randoms[i][j] < mutation_rate:
                r = 1 + random.uniform(-1, 1) * mutation_distance
                c[j] = c[j] * r
    return copies

def evolve(starting_weights, mutation_rate, mutation_distance, generations,
           bottleneck, strategies, output_file, noise, hidden_layer_size=10):

    # Append scores of 0 to starting parameters
    current_bests = [[0, x] for x in starting_weights]

    with open(output_file, 'w') as output:
        writer = csv.writer(output)

        for generation in range(generations):
            print("Generation " + str(generation))
            size = 19 * hidden_layer_size
            random_weights = [get_random_weights(size) for _ in range(4)]
            weights_to_copy = [list(x[1]) for x in current_bests]
            weights_to_copy += random_weights

            # Crossover
            copies = crossover(weights_to_copy)
            # Mutate
            copies = mutate(copies, mutation_rate)

            population = copies + [list(x[1]) for x in current_bests] + random_weights

            # map the population to get a list of (score, weights) tuples
            # this list will be sorted by score, best weights first
            results = score_all_weights(population, strategies, noise=noise,
                                        hidden_layer_size=hidden_layer_size)

            results.sort(key=itemgetter(0), reverse=True)
            current_bests = results[0: bottleneck]

            # get all the scores for this generation
            scores = [score for (score, weights) in results]

            # Write the data
            row = [generation, mean(scores), pstdev(scores), mutation_rate,
                   mutation_distance, results[0][0]]
            row.extend(results[0][1])
            writer.writerow(row)
            output.flush()
            os.fsync(output.fileno())

            print("Generation", generation, "| Best Score:", results[0][0])

    return current_bests


if __name__ == '__main__':
    arguments = docopt(__doc__, version='ANN Evolver 0.2')
    # set up the process pool
    pool = Pool(processes=int(arguments['-i']))
    # Vars for the genetic algorithm
    mutation_rate = float(arguments['-u'])
    generations = int(arguments['-g'])
    bottleneck = int(arguments['-b'])
    mutation_distance = float(arguments['-d'])
    starting_population = int(arguments['-k'])
    output_file = arguments['-o']
    noise = float(arguments['-n'])

    hidden_layer_size = 10
    size = 19 * hidden_layer_size

    starting_weights = [get_random_weights(size) for _ in range(starting_population)]
    strategies = axl.short_run_time_strategies

    evolve(starting_weights, mutation_rate, mutation_distance, generations,
           bottleneck, strategies, output_file, noise,
           hidden_layer_size=hidden_layer_size)
