from functools import partial
from itertools import repeat
from multiprocessing import Pool
from operator import itemgetter
from random import randrange
from statistics import mean, pstdev
import csv
import os

import numpy as np
import axelrod as axl


## Output Evolutionary Algorithm results

class Outputer(object):
    def __init__(self, filename, mode='w'):
        self.output = open(filename, mode)
        self.writer = csv.writer(self.output)

    def write(self, row):
        self.writer.writerow(row)
        self.output.flush()
        os.fsync(self.output.fileno())

    def close(self):
        self.output.close()


## Objective functions for optimization

def prepare_objective(name="score", turns=200, noise=0., repetitions=None,
                      nmoran=None, match_attributes=None):
    name = name.lower()
    if name not in ["score", "score_diff", "moran"]:
        raise ValueError("Score must be one of score, score_diff, or moran")
    if name == "moran":
        if repetitions is None:
            repetitions = 1000
        if nmoran is None:
            nmoran = 4
        objective = partial(objective_moran_win, turns=turns, noise=noise,
                            repetitions=repetitions, N=nmoran,
                            match_attributes=match_attributes)
    elif name == "score":
        if repetitions is None:
            repetitions = 20
        objective = partial(objective_score, turns=turns, noise=noise,
                            repetitions=repetitions,
                            match_attributes=match_attributes)
    elif name == "score_diff":
        if repetitions is None:
            repetitions = 20
        objective = partial(objective_score_diff, turns=turns, noise=noise,
                            repetitions=repetitions,
                            match_attributes=match_attributes)
    return objective


def objective_score(me, other, turns, noise, repetitions, match_attributes=None):
    """Objective function to maximize total score over matches."""
    match = axl.Match((me, other), turns=turns, noise=noise,
                      match_attributes=match_attributes)
    if not match._stochastic:
        repetitions = 1
    scores_for_this_opponent = []

    for _ in range(repetitions):
        match.play()
        scores_for_this_opponent.append(match.final_score_per_turn()[0])
    return scores_for_this_opponent


def objective_score_diff(me, other, turns, noise, repetitions,
                         match_attributes=None):
    """Objective function to maximize total score difference over matches."""
    match = axl.Match((me, other), turns=turns, noise=noise,
                      match_attributes=match_attributes)
    if not match._stochastic:
        repetitions = 1
    scores_for_this_opponent = []

    for _ in range(repetitions):
        match.play()
        final_scores = match.final_score_per_turn()
        score_diff = final_scores[0] - final_scores[1]
        scores_for_this_opponent.append(score_diff)
    return scores_for_this_opponent


def objective_moran_win(me, other, turns, noise, repetitions, N=5,
                        match_attributes=None):
    """Objective function to maximize Moran fixations over N=4 matches"""
    population = []
    for _ in range(N):
        population.append(me.clone())
        population.append(other.clone())
    mp = axl.MoranProcess(population, turns=turns, noise=noise)

    scores_for_this_opponent = []

    for _ in range(repetitions):
        mp.reset()
        mp.play()
        if mp.winning_strategy_name == str(me):
            scores_for_this_opponent.append(1)
        else:
            scores_for_this_opponent.append(0)
    return scores_for_this_opponent

# Evolutionary Algorithm


class Params(object):
    """Abstract Base Class for Parameters Objects."""
    def mutate(self):
        pass

    def random(self):
        pass

    def __repr__(self):
        pass

    def from_repr(self):
        pass

    def copy(self):
        pass

    def player(self):
        pass

    def params(self):
        pass

    def crossover(self, other):
        pass


def score_params(params, objective, opponents, weights=None):
    """
    Return the overall mean score of a Params instance.
    """
    scores_for_all_opponents = []
    player = params.player()

    for opponent_class in opponents:
        player.reset()
        opponent = opponent_class()
        scores_for_this_opponent = objective(player, opponent)
        mean_vs_opponent = mean(scores_for_this_opponent)
        scores_for_all_opponents.append(mean_vs_opponent)

    overall_mean_score = np.average(scores_for_all_opponents,
                                    weights=weights)
    return overall_mean_score


class Population(object):
    """Population class that implements the evolutionary algorithm."""
    def __init__(self, params_class, params_args, size, objective, output_filename,
                 bottleneck=None, opponents=None, processes=1, weights=None):
        self.params_class = params_class
        self.bottleneck = bottleneck
        self.pool = Pool(processes=processes)
        self.outputer = Outputer(output_filename, mode='a')
        self.size = size
        self.objective = objective
        if not bottleneck:
            self.bottleneck = size // 4
        else:
            self.bottleneck = bottleneck
        if opponents is None:
            self.opponents = axl.short_run_time_strategies
        else:
            self.opponents = opponents
        self.generation = 0
        self.params_args = params_args
        self.population = [params_class(*params_args) for _ in range(self.size)]
        self.weights = weights

    def score_all(self):
        starmap_params = zip(
            self.population,
            repeat(self.objective),
            repeat(self.opponents),
            repeat(self.weights))
        results = self.pool.starmap(score_params, starmap_params)
        return results

    def subset_population(self, indices):
        population = []
        for i in indices:
            population.append(self.population[i])
        self.population = population

    @staticmethod
    def crossover(population, num_variants):
        new_variants = []
        for _ in range(num_variants):
            i = randrange(len(population))
            j = randrange(len(population))
            new_variant = population[i].crossover(population[j])
            new_variants.append(new_variant)
        return new_variants

    def evolve(self):
        self.generation += 1
        print("Scoring Generation {}".format(self.generation))

        # Score population
        scores = self.score_all()
        results = list(zip(scores, range(len(scores))))
        results.sort(key=itemgetter(0), reverse=True)

        # Report
        print("Generation", self.generation, "| Best Score:", results[0][0],
              repr(self.population[results[0][1]]))
        # Write the data
        row = [self.generation, mean(scores), pstdev(scores), results[0][0],
               repr(self.population[results[0][1]])]
        self.outputer.write(row)

        ## Next Population
        indices_to_keep = [p for (s, p) in results[0: self.bottleneck]]
        self.subset_population(indices_to_keep)
        # Add mutants of the best players
        best_mutants = [p.copy() for p in self.population]
        for p in best_mutants:
            p.mutate()
            self.population.append(p)
        # Add random variants
        random_params = [self.params_class(*self.params_args)
                         for _ in range(self.bottleneck // 2)]
        params_to_modify = [params.copy() for params in self.population]
        params_to_modify += random_params
        # Crossover
        size_left = self.size - len(params_to_modify)
        params_to_modify = self.crossover(params_to_modify, size_left)
        # Mutate
        for p in params_to_modify:
            p.mutate()
        self.population += params_to_modify

    def __iter__(self):
        return self

    def __next__(self):
        self.evolve()

    def run(self, generations):
        for _ in range(generations):
            next(self)
        self.outputer.close()


def load_params(params_class, filename, num):
    """Load the best num parameters from the given file."""
    parser = params_class.parse_repr
    all_params = []

    with open(filename) as datafile:
        reader = csv.reader(datafile)
        for line in reader:
            score, rep = float(line[-2]), line[-1]
            all_params.append((score, rep))
    all_params.sort(reverse=True)
    best_params = []
    for score, rep in all_params[:num]:
        best_params.append(parser(rep))
    return best_params


## This function is only used for PSO.

def score_for(strategy_factory, objective, args=None, opponents=None):
    """
    Given a function that will return a strategy, calculate the average score
    per turn against all ordinary strategies. If the opponent is classified as
    stochastic, then run 100 repetitions and take the average to get a good
    estimate.
    """
    if not args:
        args = []
    scores_for_all_opponents = []
    if not opponents:
        opponents = axl.short_run_time_strategies

    for opponent in opponents:
        me = strategy_factory(*args)
        other = opponent()
        scores_for_this_opponent = objective(me, other)
        mean_vs_opponent = mean(scores_for_this_opponent)
        scores_for_all_opponents.append(mean_vs_opponent)

    # Calculate the average for all opponents
    overall_mean_score = mean(scores_for_all_opponents)
    return overall_mean_score
