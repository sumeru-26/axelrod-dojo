"""Lookup Evolve.

Usage:
    lookup_evolve.py [-h] [-p PLAYS] [-o OPP_PLAYS] [-s STARTING_PLAYS]
    [-g GENERATIONS] [-k STARTING_POPULATION] [-u MUTATION_RATE] [-b BOTTLENECK]
    [-i PROCESSORS] [-f OUTPUT_FILE] [-z INITIAL_POPULATION_FILE] [-n NOISE]

Options:
    -h --help                   show this
    -p PLAYS                    number of recent plays in the lookup table [default: 2]
    -o OPP_PLAYS                number of recent plays in the lookup table [default: 2]
    -s STARTING_PLAYS           number of opponent starting plays in the lookup table [default: 2]
    -g GENERATIONS              how many generations to run the program for [default: 500]
    -k STARTING_POPULATION      starting population size for the simulation [default: 20]
    -u MUTATION_RATE            mutation rate i.e. probability that a given value will flip [default: 0.1]
    -b BOTTLENECK               number of individuals to keep from each generation [default: 10]
    -i PROCESSORS               number of processors to use [default: 1]
    -f OUTPUT_FILE              file to write data to [default: tables.csv]
    -z INITIAL_POPULATION_FILE  file to read an initial population from [default: None]
    -n NOISE                    match noise [default: 0.00]
"""

from __future__ import division
import csv
from itertools import repeat
from multiprocessing import Pool
from operator import itemgetter
import os
import random
from statistics import mean, pstdev

from docopt import docopt
import numpy as np

from axelrod import flip_action
from axelrod.strategies.lookerup import LookerUp, create_lookup_table_keys
from axelrod_utils import score_for, objective_match_score, objective_match_moran_win
import analyze_data


## Look up Tables

def get_random_tables(plays, opp_plays, start_plays, number):
    """Return randomly-generated lookup tables"""
    # The list of keys for the lookup table is just the product of these three
    # lists
    lookup_table_keys = create_lookup_table_keys(plays, opp_plays, start_plays)

    # To get a pattern, we just randomly pick between C and D for each key
    patterns = [''.join([random.choice("CD") for _ in lookup_table_keys])
                for i in range(number)]

    # zip together the keys and the patterns to give a table
    tables = [dict(zip(lookup_table_keys, pattern)) for pattern in patterns]

    return tables

def id_for_table(table):
    """Return a string representing the values of a lookup table dict"""
    return ''.join([v for k, v in sorted(table.items())])

def table_from_id(string_id, keys):
    """Return a lookup table dict from a string representing the values"""
    return dict(zip(keys, string_id))

def score_table(table, noise):
    """
    Take a lookup table dict and return a tuple of the score and the table.
    """
    return (score_for(LookerUp, args=[table], noise=noise,
                      objective=objective_match_score), table)

def score_all_tables(tables, pool, noise):
    """Use a multiprocessing Pool to take a bunch of tables and score them"""
    results = list(pool.starmap(score_table, zip(tables, repeat(noise))))
    results.sort(reverse=True, key=itemgetter(0))
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

def evolve(starting_tables, mutation_rate, generations, bottleneck, pool, plays,
           opp_plays, start_plays, starting_pop, output_file, noise):
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
    current_bests = starting_tables

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
            random_tables = get_random_tables(
                plays, opp_plays, start_plays, starting_pop)
            tables_to_copy = [x[1] for x in current_bests] + random_tables

            # Crossover
            copies = crossover(tables_to_copy)
            # Mutate
            copies = mutate(copies, mutation_rate)

            # The population of tables we want to consider includes the
            # recombined, mutated copies, plus the originals
            population = copies + tables_to_copy

            # Map the population to get a list of (score, table) tuples
            # This list will be sorted by score, best tables first
            results = score_all_tables(population, pool, noise)

            # The best tables from this generation become the starting tables
            # for the next generation
            current_bests = results[0: bottleneck]

            # get all the scores for this generation
            scores = [score for (score, table) in results]

            # Write the data
            row = [generation, mean(scores), pstdev(scores), results[0][0],
                   id_for_table(results[0][1])]
            row.extend(results[0][1])
            writer.writerow(row)
            output.flush()
            os.fsync(output.fileno())

            print("Generation", generation, "| Best Score:", results[0][0],
                  id_for_table(results[0][1]))
    return current_bests


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Lookup Evolver 0.2')
    # set up the process pool
    pool = Pool(processes=int(arguments['-i']))

    # vars for the genetic algorithm
    starting_pop = int(arguments['-k'])
    mutation_rate = float(arguments['-u'])
    generations = int(arguments['-g'])
    bottleneck = int(arguments['-b'])
    plays = int(arguments['-p'])
    opp_plays = int(arguments['-o'])
    start_plays = int(arguments['-s'])
    initial_population = arguments['-z']
    output_file = arguments['-f']
    noise = float(arguments['-n'])

    # generate a starting population of tables and score them
    # these will start off the first generation
    if initial_population == 'None':
        starting_tables = get_random_tables(
            plays, opp_plays, start_plays, starting_pop)
    else:
        keys = create_lookup_table_keys(plays, opp_plays, start_plays)
        data = analyze_data.read_top_tables(initial_population, starting_pop)
        starting_tables = []
        for row in data:
            starting_tables.append(table_from_id(row[1], keys))
        # If insufficient size in data augment with random tables:
        if len(starting_tables) < starting_pop:
            needed_num = starting_pop - len(starting_tables)
            starting_tables = starting_tables + get_random_tables(
                plays, opp_plays, start_plays, needed_num)

    real_starting_tables = score_all_tables(starting_tables, pool, noise)

    # kick off the evolve function
    evolve(real_starting_tables, mutation_rate, generations, bottleneck, pool,
           plays, opp_plays, start_plays, starting_pop, output_file, noise)
