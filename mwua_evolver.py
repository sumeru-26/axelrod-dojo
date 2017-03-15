"""
MWUA Evolver

# https://jeremykun.com/2017/02/27/the-reasonable-effectiveness-of-the-multiplicative-weights-update-algorithm/

Usage:
    mwua_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN] [--team_size TEAM_SIZE]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Population size  [default: 40]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 10]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: mwua_params.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --team_size TEAM_SIZE       Initial Team Size [default: 10]
"""

import random
from random import randint, choice, uniform

from docopt import docopt
import numpy as np

import axelrod
from axelrod import Actions, flip_action
from axelrod.strategies.meta import MetaPlayer
from axelrod.moran import fitness_proportionate_selection

from evolve_utils import Params, Population, prepare_objective

C, D = Actions.C, Actions.D

DEFAULT_TEAM_SIZE = 10
STRATEGIES = [t for t in axelrod.short_run_time_strategies
              if not issubclass(t, MetaPlayer)]


# Variation: non-adaptive weights and just use meta-mixer

class MetaMWUA(MetaPlayer):
    def __init__(self, team=None, initial_weights=None, learning_rate=None):
        super().__init__(team=team)
        if not initial_weights:
            initial_weights = [1] * len(self.team)
        self.initial_weights = list(initial_weights)
        self.weights = list(initial_weights)
        if not learning_rate:
            learning_rate = 0.5
        self.learning_rate = learning_rate
        self.classifier["stochastic"] = True

    def _update_scores_weights(self, opponent):
        # Update the running score for each player, before determining the
        # next move.
        game = self.match_attributes["game"]
        for i, player in enumerate(self.team):
            last_round = (player.history[-1], opponent.history[-1])
            score = game.scores[last_round][0]
            self.weights[i] *= 1 + self.learning_rate * score

    def meta_strategy(self, results, opponent):
        if len(self.history) == 0:
            return C
        self._update_scores_weights(opponent)
        # Randomly choose a strategy proportionally to its weight
        index = fitness_proportionate_selection(self.weights)
        action = results[index]
        return action

    def reset(self):
        super().reset()
        self.weights = list(self.initial_weights)


class MWUAParams(Params):

    def __init__(self, mutation_rate=None, team=None, initial_weights=None,
                 learning_rate=0.2):
        self.PlayerClass = MetaMWUA
        self.learning_rate = learning_rate
        if mutation_rate is None:
            self.mutation_rate = 0.1
        else:
            self.mutation_rate = mutation_rate
        if (initial_weights is None) and (team is None):
            self.randomize()
        else:
            # # Make sure to copy the lists
            t = []
            w = []
            for player, weight in zip(team, initial_weights):
                if weight < 0.005:
                    continue
                t.append(player)
                w.append(weight)
            self.team = list(t)
            self.initial_weights = list(w)
            # self.team = list(team)
            # self.initial_weights = list(initial_weights)

    def player(self):
        player = self.PlayerClass(self.team, self.initial_weights,
                                  self.learning_rate)
        return player

    def copy(self):
        return MWUAParams(self.mutation_rate, self.team, self.initial_weights,
                          self.learning_rate)

    def randomize(self):
        self.team = random.sample(STRATEGIES, DEFAULT_TEAM_SIZE)
        self.initial_weights = [uniform(0.5, 2) for _ in range(len(self.team))]
        self.learning_rate = uniform(0, 0.3)

    def mutate(self):
        self.learning_rate *= (1 + uniform(-0.1, 0.1))
        # Randomly add or remove a team member
        r = random.random()
        to_remove = []
        if r < 0.05:
            r = random.random()
            if r < 0.5:
                i = random.randint(0, len(self.team) - 1)
                to_remove.append(i)
            else:
                self.team.append(
                    choice(STRATEGIES))
                self.initial_weights.append(
                    uniform(0.1, max(self.initial_weights)))
        for i in to_remove:
            self.team.pop(i)
            self.initial_weights.pop(i)
        # Reweight
        randoms = np.random.random(len(self.team))
        for i, r in enumerate(randoms):
            if r < mutation_rate:
                s = self.initial_weights[i]
                s *= 1 + uniform(-0.2, 0.2)
                s = max(s, 0)
                self.initial_weights[i] = s

    @staticmethod
    def crossover_lists(w1, w2):
        size = min(len(w1), len(w2))
        crosspoint = randint(0, size)
        new_weights = list(w1[:crosspoint]) + list(w2[crosspoint:])
        return new_weights

    def crossover(self, other):
        new_team = self.crossover_lists(self.team, other.team)
        new_weights = self.crossover_lists(self.initial_weights,
                                           other.initial_weights)
        return MWUAParams(mutation_rate=self.mutation_rate, team=new_team,
                          initial_weights=new_weights,
                          learning_rate=self.learning_rate)


    def __repr__(self):
        return "{}:{}:{}".format(
            ",".join(list(map(str, self.team))),
            ",".join(list(map(str, self.initial_weights))),
            self.learning_rate
        )

    @classmethod
    def parse_repr(cls, s):
        pass


if __name__ == '__main__':
    arguments = docopt(__doc__, version='MWUA Evolver 0.3')
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

    # MWUA
    DEFAULT_TEAM_SIZE = int(arguments['--team_size'])
    param_args = [mutation_rate]

    objective = prepare_objective(name, turns, noise, repetitions, nmoran)
    population = Population(MWUAParams, param_args, population, objective,
                            output_filename, bottleneck, processes=processes)
    population.run(generations)
