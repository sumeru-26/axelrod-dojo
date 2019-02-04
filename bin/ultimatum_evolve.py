"""
Ultimatum Evolver

Usage:
    ultimatum.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Population size  [default: 40]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 10]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: ultimatum_params.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
"""

from docopt import docopt
import random


from axelrod.ultimatum import DoubleThresholdsPlayer
from axelrod_dojo import (
    DoubleThresholdsPlayerParams, Population, prepare_objective)


def random_init_kwargs():
    lower_offer = random.random()
    upper_offer = random.uniform(lower_offer, 1)
    lower_accept = random.random()
    upper_accept = random.uniform(lower_accept, 1)
    return dict(
         lower_offer=lower_offer,
         upper_offer=upper_offer,
         lower_accept=lower_accept,
         upper_accept=upper_accept)


def random_population(size):
    players = [DoubleThresholdsPlayer(**random_init_kwargs())
               for _ in range(size)]
    return players


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Ultimatum Evolve 0.1')
    print(arguments)
    processes = int(arguments['--processes'])

    # Vars for the genetic algorithm
    population = int(arguments['--population'])
    mutation_probability = float(arguments['--mu'])
    generations = int(arguments['--generations'])
    bottleneck = int(arguments['--bottleneck'])
    output_filename = arguments['--output']

    # Objective
    name = str(arguments['--objective'])
    repetitions = int(arguments['--repetitions'])
    turns = int(arguments['--turns'])
    noise = float(arguments['--noise'])
    nmoran = int(arguments['--nmoran'])

    objective = prepare_objective("ultimatum_score", turns, noise, repetitions,
                                  nmoran)
    population = Population(DoubleThresholdsPlayerParams, dict(), population,
                            objective, output_filename, bottleneck,
                            mutation_probability, processes=processes,
                            opponents=random_population(population))
    population.run(generations)
