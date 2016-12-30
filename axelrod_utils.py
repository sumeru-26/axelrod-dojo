import csv
from functools import partial
import os
from statistics import mean

import axelrod as axl


class Outputer(object):
    def __init__(self, filename, mode='w'):
        self.output = open(filename, mode)
        self.writer = csv.writer(self.output)

    def write(self, row):
        self.writer.writerow(row)
        self.output.flush()
        os.fsync(self.output.fileno())


## Objective functions for optimization

def prepare_objective(name="score", noise=0., repetitions=None, N=None):
    name = name.lower()
    if name not in ["score", "score_diff", "moran"]:
        raise ValueError("Score must be one of score, score_diff, or moran")
    if name == "moran":
        if repetitions is None:
            repetitions = 1000
        if N is None:
            N = 4
        objective = partial(objective_moran_win, noise=noise,
                            repetitions=repetitions, N=N)
    elif name == "score":
        if repetitions is None:
            repetitions = 20
        objective = partial(objective_score, noise=noise,
                            repetitions=repetitions)
    elif name == "score_diff":
        if repetitions is None:
            repetitions = 20
        objective = partial(objective_score_diff, noise=noise,
                            repetitions=repetitions)
    return objective

def objective_score(me, other, turns, noise, repetitions):
    """Objective function to maximize total score over matches."""
    match = axl.Match((me, other), turns=turns, noise=noise)
    if not match._stochastic:
        repetitions = 1
    scores_for_this_opponent = []

    for _ in range(repetitions):
        match.play()
        scores_for_this_opponent.append(match.final_score_per_turn()[0])
    return scores_for_this_opponent

def objective_score_diff(me, other, turns, noise, repetitions):
    """Objective function to maximize total score difference over matches."""
    match = axl.Match((me, other), turns=turns, noise=noise)
    if not match._stochastic:
        repetitions = 1
    scores_for_this_opponent = []

    for _ in range(repetitions):
        match.play()
        final_scores = match.final_score_per_turn()
        score_diff = final_scores[0] - final_scores[1]
        scores_for_this_opponent.append(score_diff)
    return scores_for_this_opponent

def objective_moran_win(me, other, turns, noise, repetitions):
    """Objective function to maximize Moran fixations over N=4 matches"""
    assert(noise == 0)
    # N = 4 population
    population = (me, me.clone(), other, other.clone())
    mp = axl.MoranProcess(population, turns=turns, noise=noise)

    scores_for_this_opponent = []

    for _ in range(repetitions):
        mp.play()
        if mp.winning_strategy_name == str(me):
            scores_for_this_opponent.append(1)
        else:
            scores_for_this_opponent.append(0)
    return scores_for_this_opponent

# Evolutionary Algorithm

def score_for(strategy_factory, objective, args=None, opponents=None,
              turns=200):
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
        # opponents = [s for s in axl.strategies]
        opponents = axl.short_run_time_strategies

    for opponent in opponents:
        me = strategy_factory(*args)
        other = opponent()
        scores_for_this_opponent = objective(me, other, turns)
        mean_vs_opponent = mean(scores_for_this_opponent)
        scores_for_all_opponents.append(mean_vs_opponent)

    # Calculate the average for all opponents
    overall_mean_score = mean(scores_for_all_opponents)
    return overall_mean_score
