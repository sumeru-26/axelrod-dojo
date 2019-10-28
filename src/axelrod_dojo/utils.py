from collections import namedtuple
import csv
from functools import partial
from statistics import mean

import numpy as np
import axelrod as axl


PlayerInfo = namedtuple('PlayerInfo', ['strategy', 'init_kwargs'])


# Output Evolutionary Algorithm results

class Outputer(object):
    def __init__(self, filename, mode='a'):
        self.file = filename
        self.mode = mode

    def write_row(self, row):
        with open(self.file, self.mode, newline='') as file_writer:
            writer = csv.writer(file_writer)
            writer.writerow(row)


# Objective functions for optimization

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


def score_player(player, objective, opponents_information, weights=None, sample_count=None):
    """
    Return the overall mean score of a Player
    """
    scores_for_all_opponents = []

    if sample_count is not None:
        indices = np.random.choice(len(opponents_information), sample_count)
        opponents_information = [opponents_information[i] for i in indices]
        if weights is not None:
            weights = [weights[i] for i in indices]

    for strategy, init_kwargs in opponents_information:
        player.reset()
        opponent = strategy(**init_kwargs)
        scores_for_this_opponent = objective(player, opponent)
        mean_vs_opponent = mean(scores_for_this_opponent)
        scores_for_all_opponents.append(mean_vs_opponent)

    overall_mean_score = np.average(scores_for_all_opponents, weights=weights)
    return overall_mean_score


def load_params(player_class, filename, num):
    """Load the best num parameters from the given file."""
    parser = player_class.deserialize_parameters
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
