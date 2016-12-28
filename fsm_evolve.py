"""FSM Evolve.

Usage:
    fsm_evolve.py [-h] [-s NUM_STATES] [-g GENERATIONS]
    [-k STARTING_POPULATION] [-u MUTATION_RATE] [-b BOTTLENECK]
    [-i PROCESSORS] [-f OUTPUT_FILE] [-n NOISE]

Options:
    -h --help                   show this
    -s NUM_STATES               number FSM states [default: 16]
    -g GENERATIONS              how many generations to run the program for [default: 500]
    -k STARTING_POPULATION      starting population size for the simulation [default: 10]
    -u MUTATION_RATE            mutation rate i.e. probability that a given value will flip [default: 0.1]
    -b BOTTLENECK               number of individuals to keep from each generation [default: 10]
    -i PROCESSORS               number of processors to use [default: 1]
    -f OUTPUT_FILE              file to write data to [default: fsm_tables.csv]
    -n NOISE                    match noise [default: 0.00]
"""

import csv
from itertools import repeat
from multiprocessing import Pool
from operator import itemgetter
import os
from random import randrange, choice
from statistics import mean, pstdev

from docopt import docopt
import numpy as np

from axelrod import flip_action
from axelrod.strategies.finite_state_machines import FSMPlayer
from axelrod_utils import score_for, objective_match_score, objective_match_moran_win


## FSM Tables

def get_random_tables(num_states, num_tables):
    """Return randomly-generated tables."""
    tables = []
    for i in range(num_tables):
        table = []
        for j in range(num_states):
            next_state = randrange(num_states)
            next_action = choice(['C', 'D'])
            row = [j, 'C', next_state, next_action]
            table.append(row)
            next_state = randrange(num_states)
            next_action = choice(['C', 'D'])
            row = [j, 'D', next_state, next_action]
            table.append(row)
        tables.append(table)
    return tables

def table_representation(table):
    ss = []
    for row in table:
        ss.append("_".join(list(map(str, row))))
    return ":".join(ss)

def table_from_representation(rep):
    table = []
    lines = rep.split(':')
    for line in lines:
        row = line.split('_')
        table.append(row)
    return table

def score_table(table, noise):
    """
    Take a lookup table dict and return a tuple of the score and the table.
    """
    return (score_for(FSMPlayer, args=[table, 0, 'C'], noise=noise,
                      # objective=objective_match_moran_win), table)
                      objective=objective_match_score), table)


def score_all_tables(tables, pool, noise):
    """Use a multiprocessing Pool to take a bunch of tables and score them"""
    results = list(pool.starmap(score_table, zip(tables, repeat(noise))))
    results.sort(reverse=True, key=itemgetter(0))
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
            new_table = t1[:crosspoint]
            new_table += t2[crosspoint:]
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

def evolve(starting_tables, mutation_rate, generations, bottleneck, pool,
           num_states, starting_pop, output_file, noise):
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

    # Current_bests is a list of 2-tuples, each of which consists of a score
    # and a lookup table initially the collection of best tables are the ones
    # supplied to start with
    current_bests = [[0, t] for t in starting_tables]

    # keys = list(sorted(create_lookup_table_keys(plays, opp_plays, start_plays)))

    for generation in range(generations):
        # Because this is a long-running process we'll just keep appending to
        # the output file so we can monitor it while it's running
        with open(output_file, 'w') as output:
            writer = csv.writer(output)
            print("Generation " + str(generation))

            # The tables at the start of this generation are the best ones from
            # the previous generation (i.e. the second element of each tuple)
            # plus a bunch of random ones
            random_tables = get_random_tables(num_states, 4)
            tables_to_copy = [list(x[1]) for x in current_bests]
            tables_to_copy += random_tables

            # Crossover
            copies = crossover(tables_to_copy)
            # Mutate
            copies = mutate(copies, mutation_rate)

            # The population of tables we want to consider includes the
            # recombined, mutated copies, plus the originals
            population = copies + [list(x[1]) for x in current_bests]
            # Map the population to get a list of (score, table) tuples
            # This list will be sorted by score, best tables first
            results = score_all_tables(population, pool, noise)

            # The best tables from this generation become the starting tables
            # for the next generation
            results.sort(key=itemgetter(0), reverse=True)
            current_bests = results[0: bottleneck]

            # get all the scores for this generation
            scores = [score for (score, table) in results]

            # Write the data
            row = [generation, mean(scores), pstdev(scores), results[0][0],
                   table_representation(results[0][1])]
            row.extend(results[0][1])
            writer.writerow(row)
            output.flush()
            os.fsync(output.fileno())

            print("Generation", generation, "| Best Score:", results[0][0],
                  table_representation(results[0][1]))
    return current_bests


if __name__ == '__main__':
    arguments = docopt(__doc__, version='FSM Evolver 0.2')
    # set up the process pool
    pool = Pool(processes=int(arguments['-i']))

    # vars for the genetic algorithm
    starting_pop = int(arguments['-k'])
    mutation_rate = float(arguments['-u'])
    generations = int(arguments['-g'])
    bottleneck = int(arguments['-b'])
    num_states = int(arguments['-s'])
    output_file = arguments['-f']
    noise = float(arguments['-n'])

    # generate a starting population of tables and score them
    # these will start off the first generation
    starting_tables = get_random_tables(num_states, starting_pop)

    # real_starting_tables = score_all_tables(starting_tables, pool, noise)

    # kick off the evolve function
    evolve(starting_tables, mutation_rate, generations, bottleneck, pool,
           num_states, starting_pop, output_file, noise)
