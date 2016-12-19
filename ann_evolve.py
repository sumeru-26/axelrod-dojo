"""
Training ANN strategies with an evolutionary algorithm.

Original code by Martin Jones @mojones:
https://gist.github.com/mojones/b809ba565c93feb8d44becc7b93e37c6

Usage:
    ann_evolve.py [-h] [-g GENERATIONS] [-u MUTATION_RATE] [-b BOTTLENECK]
    [-d mutation_distance] [-i PROCESSORS] [-o OUTPUT_FILE]
    [-k STARTING_POPULATION]

Options:
    -h --help                    show this
    -g GENERATIONS               how many generations to run the program for [default: 100]
    -u MUTATION_RATE             mutation rate i.e. probability that a given value will flip [default: 0.1]
    -d MUTATION_DISTANCE         amount of change a mutation will cause [default: 0.5]
    -b BOTTLENECK                number of individuals to keep from each generation [default: 10]
    -i PROCESSORS                number of processors to use [default: 1]
    -o OUTPUT_FILE               file to write statistics to [default: ann_out.csv]
    -k STARTING_POPULATION       starting population size for the simulation [default: 5]
"""


from __future__ import division

import copy
import random
from multiprocessing import Pool
from statistics import mean, pstdev

from docopt import docopt

import axelrod as axl
from axelrod.strategies.ann import ANN, split_weights
from axelrod_utils import mean, pstdev


def get_random_weights(number):
    return [random.uniform(-1, 1) for _ in range(number)]

def score_single(my_strategy_factory, other_strategy_factory, length=200):
    if other_strategy_factory().classifier['stochastic']:
        repetitions = 10
    else:
        repetitions = 1
    all_scores = []
    for _ in range(repetitions):
        me = my_strategy_factory()
        other = other_strategy_factory()
        me.set_match_attributes(length=length)
        other.set_match_attributes(length=length)

        g = axl.Game()
        for _ in range(length):
            me.play(other)
        iteration_score = sum([g.score(pair)[0] for pair in
                               zip(me.history, other.history)]) / length
        all_scores.append(iteration_score)
    return sum(all_scores) / repetitions

def score_for(my_strategy_factory, other_strategies, iterations=200):
    my_scores = list(map(
        lambda x: score_single(my_strategy_factory, x, iterations),
        other_strategies))
    my_average_score = sum(my_scores) / len(my_scores)
    return my_average_score

def score_weights(weights, strategies, input_values=17, hidden_layer_size=10):
    in2h, h2o, bias = split_weights(weights, input_values, hidden_layer_size)
    return (score_for(lambda: ANN(in2h, h2o, bias), strategies), weights)

from itertools import repeat

def score_all_weights(population, strategies):
    results = pool.starmap(score_weights, zip(population, repeat(strategies)))
    return list(sorted(results, reverse=True))

def evolve(starting_weights, mutation_rate, mutation_distance, generations,
           bottleneck, strategies, output_file):

    # Append scores
    current_bests = [[0, x] for x in starting_weights]

    for generation in range(generations):
        with open(output_file, "a") as output:
            weights_to_copy = [x[1] for x in current_bests]
            copies = []
            # Crossover
            for w1 in weights_to_copy:
                for w2 in weights_to_copy:
                    crossover = random.randrange(len(w1))
                    new_weights = copy.deepcopy(
                        w1[0:crossover]) + copy.deepcopy(w2[crossover:])
                    copies.append(new_weights)

            # Mutate
            for _, c in copies:
                for i in range(len(c)):
                    if random.random() < mutation_rate:
                        r = 1 + random.uniform(-1, 1) * mutation_distance
                        c[i] = c[i] * r

            population = copies + weights_to_copy

            # map the population to get a list of (score, weights) tuples
            # this list will be sorted by score, best weights first
            results = score_all_weights(population, strategies)

            current_bests = results[0: bottleneck]

            # get all the scores for this generation
            scores = [score for score, weights in results]

            for value in [generation, results[0][1], results[0][0],
                          mean(scores), pstdev(scores), mutation_rate,
                          mutation_distance]:
                output.write(str(value) + "\t")
            output.write("\n")
            print("Generation", generation, "| Best Score:", scores[0])
            mutation_rate *= 0.99
            mutation_distance *= 0.99

    return current_bests


if __name__ == '__main__':
    arguments = docopt(__doc__, version='ANN Evolver 0.1')
    # set up the process pool
    pool = Pool(processes=int(arguments['-i']))
    # Vars for the genetic algorithm
    mutation_rate = float(arguments['-u'])
    generations = int(arguments['-g'])
    bottleneck = int(arguments['-b'])
    mutation_distance = float(arguments['-d'])
    starting_population = int(arguments['-k'])
    output_file = arguments['-o']

    starting_weights = [[0, get_random_weights(190)] for _ in range(starting_population)]

    strategies = [s for s in axl.strategies
                  if not s().classifier['long_run_time']]

    evolve(starting_weights, mutation_rate, mutation_distance, generations,
           bottleneck, strategies, output_file)


