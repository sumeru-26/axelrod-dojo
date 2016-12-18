"""
Training ANN strategies.

Original code by Martin Jones @mojones:
https://gist.github.com/mojones/b809ba565c93feb8d44becc7b93e37c6
"""

from __future__ import division

import copy
import random
from statistics import mean, pstdev

import axelrod


def evolve(starting_weights, mutation_rate, mutation_distance, generations,
           bottleneck, starting_pop, output_file):

    current_bests = starting_weights

    for generation in range(generations):

        with open(output_file, "a") as output:

            weights_to_copy = [x[1] for x in current_bests]

            copies = []

            for w1 in weights_to_copy:
                for w2 in weights_to_copy:
                    crossover = random.randrange(len(w1))
                    new_weights = copy.deepcopy(
                        w1[0:crossover]) + copy.deepcopy(w2[crossover:])
                    copies.append(new_weights)

            for c in copies:
                for i in range(len(c)):
                    if random.random() < mutation_rate:
                        c[i] = c[i] * (
                        1 + (random.uniform(-1, 1) * mutation_distance))

            population = copies + weights_to_copy

            # map the population to get a list of (score, weights) tuples
            # this list will be sorted by score, best weights first
            results = score_all_weights(population)

            current_bests = results[0:bottleneck]

            # get all the scores for this generation
            scores = [score for score, table in results]

            for value in [generation, results[0][1], results[0][0],
                          mean(scores), pstdev(scores), mutation_rate,
                          mutation_distance]:
                output.write(str(value) + "\t")
            output.write("\n")

            mutation_rate *= 0.99
            mutation_distance *= 0.99

    return (current_bests)


def get_random_weights(number):
    return [random.uniform(-1, 1) for _ in range(number)]


def score_single(my_strategy_factory, other_strategy_factory, iterations=200,
                 debug=False):
    if other_strategy_factory.classifier['stochastic']:
        repetitions = 10
    else:
        repetitions = 1
    all_scores = []
    for _ in range(repetitions):
        me = my_strategy_factory()
        other = other_strategy_factory()
        me.set_tournament_attributes(length=iterations)
        other.set_tournament_attributes(length=iterations)

        g = axelrod.Game()
        for _ in range(iterations):
            me.play(other)
        # print(me.history)
        iteration_score = sum([g.score(pair)[0] for pair in
                               zip(me.history, other.history)]) / iterations
        all_scores.append(iteration_score)


def split_weights(weights, input_values, hidden_layer_size):
    number_of_input_to_hidden_weights = input_values * hidden_layer_size
    number_of_hidden_bias_weights = hidden_layer_size
    number_of_hidden_to_output_weights = hidden_layer_size

    input2hidden = []
    for i in range(0, number_of_input_to_hidden_weights, input_values):
        input2hidden.append(weights[i:i + input_values])

    hidden2output = weights[
                    number_of_input_to_hidden_weights:number_of_input_to_hidden_weights + number_of_hidden_to_output_weights]
    bias = weights[
           number_of_input_to_hidden_weights + number_of_hidden_to_output_weights:]

    return (input2hidden, hidden2output, bias)


def score_all_weights(population):
    return sorted(pool.map(score_weights, population), reverse=True)


def _score_weights(weights):
    in2h, h2o, bias = split_weights(weights, input_values, hidden_layer_size)
    return (score_for(lambda: ANN(in2h, h2o, bias), strategies), weights)


score_weights = _score_weights


def score_for(my_strategy_factory, other_strategies=strategies, iterations=200,
              debug=False):
    my_scores = map(
        lambda x: score_single(my_strategy_factory, x, iterations, debug=debug),
        other_strategies)
    my_average_score = sum(my_scores) / len(my_scores)
    return (my_average_score)
