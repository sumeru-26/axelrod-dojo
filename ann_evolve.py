"""
ANN evolver.
Trains ANN strategies with an evolutionary algorithm.

Original version by Martin Jones @mojones:
https://gist.github.com/mojones/b809ba565c93feb8d44becc7b93e37c6

Usage:
    ann_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]
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
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --features FEATURES         Number of ANN features [default: 17]
    --hidden HIDDEN             Number of hidden nodes [default: 10]
    --mu_distance DISTANCE      Delta max for weights updates [default: 5]
"""

import random

from docopt import docopt
import numpy as np

from axelrod import Actions
from axelrod.strategies.ann import ANN
from evolve_utils import Params, Population, prepare_objective

C, D = Actions.C, Actions.D


## Todo: mutation decay
# if decay:
#     mutation_rate *= 0.995
#     mutation_distance *= 0.995


def num_weights(num_features, num_hidden):
    size = num_features * num_hidden + 2 * num_hidden
    return size


class ANNParams(Params):

    def __init__(self, num_features, num_hidden, mutation_rate=0.1,
                 mutation_distance=5, weights=None):
        self.PlayerClass = ANN
        self.num_features = num_features
        self.num_hidden = num_hidden

        self.mutation_distance = mutation_distance
        if mutation_rate is None:
            size = num_weights(num_features, num_hidden)
            self.mutation_rate = 10 / size
        else:
            self.mutation_rate = mutation_rate

        if weights is None:
            self.randomize()
        else:
            # Make sure to copy the lists
            self.weights = list(weights)

    def player(self):
        player = self.PlayerClass(self.weights, self.num_features,
                                  self.num_hidden)
        return player

    def copy(self):
        return ANNParams(
            self.num_features, self.num_hidden, self.mutation_rate,
            self.mutation_distance, list(self.weights))

    def randomize(self):
        size = num_weights(self.num_features, self.num_hidden)
        self.weights = [random.uniform(-1, 1) for _ in range(size)]

    @staticmethod
    def mutate_weights(weights, num_features, num_hidden, mutation_rate,
                       mutation_distance):
        size = num_weights(num_features, num_hidden)
        randoms = np.random.random(size)
        for i, r in enumerate(randoms):
            if r < mutation_rate:
                p = 1 + random.uniform(-1, 1) * mutation_distance
                weights[i] = weights[i] * p
        return weights

    def mutate(self):
        self.weights = self.mutate_weights(
            self.weights, self.num_features, self.num_hidden,
            self.mutation_rate, self.mutation_distance)
        # Add in layer sizes?

    @staticmethod
    def crossover_weights(w1, w2):
        crosspoint = random.randrange(len(w1))
        new_weights = list(w1[:crosspoint]) + list(w2[crosspoint:])
        return new_weights

    def crossover(self, other):
        # Assuming that the number of states is the same
        new_weights = self.crossover_weights(self.weights, other.weights)
        return ANNParams(
            self.num_features, self.num_hidden, self.mutation_rate,
            self.mutation_distance, new_weights)

    def __repr__(self):
        return "{}:{}:{}".format(
            self.num_features,
            self.num_hidden,
            ':'.join(map(str, self.weights))
        )

    @classmethod
    def parse_repr(cls, s):
        elements = list(map(float, s.split(':')))
        num_features = elements[0]
        num_hidden = elements[1]
        weights = elements[2:]
        return cls(num_features, num_hidden, weights)


if __name__ == '__main__':
    arguments = docopt(__doc__, version='ANN Evolver 0.3')
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

    # ANN
    num_features = int(arguments['--features'])
    num_hidden = int(arguments['--hidden'])
    mutation_distance = float(arguments['--mu_distance'])
    param_args = [num_features, num_hidden, mutation_rate, mutation_distance]

    objective = prepare_objective(name, turns, noise, repetitions, nmoran)
    population = Population(ANNParams, param_args, population, objective,
                            output_filename, bottleneck, processes=processes)
    population.run(generations)
