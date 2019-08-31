"""
ANN evolver.
Trains ANN strategies with an evolutionary algorithm.

Original version by Martin Jones @mojones:
https://gist.github.com/mojones/b809ba565c93feb8d44becc7b93e37c6

Usage:
    ann_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu mutation_probability] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]
    [--features FEATURES] [--hidden HIDDEN] [--mu_distance DISTANCE]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Starting population size  [default: 40]
    --mu mutation_probability          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 10]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: ann_params.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --features FEATURES         Number of ANN features [default: 17]
    --hidden HIDDEN             Number of hidden nodes [default: 10]
    --mu_distance DISTANCE      Delta max for weights updates [default: 10]
"""

from docopt import docopt

from axelrod import Action, ANN
from axelrod_dojo import Population, prepare_objective

C, D = Action.C, Action.D


if __name__ == '__main__':
    arguments = docopt(__doc__, version='ANN Evolver 0.4')
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

    # ANN
    num_features = int(arguments['--features'])
    num_hidden = int(arguments['--hidden'])
    mutation_distance = float(arguments['--mu_distance'])
    param_kwargs = {"num_features": num_features, 
                    "num_hidden": num_hidden,
                    "mutation_distance": mutation_distance}

    objective = prepare_objective(name, turns, noise, repetitions, nmoran)
    population = Population(ANN, param_kwargs, population, objective,
                            output_filename, bottleneck, 
                            mutation_probability,
                            processes=processes)
    population.run(generations)
