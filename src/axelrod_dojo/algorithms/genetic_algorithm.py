from itertools import repeat, starmap
from multiprocessing import Pool, cpu_count
from operator import itemgetter
from random import randrange
from statistics import mean, pstdev

import axelrod as axl
from axelrod_dojo.utils import Outputer, PlayerInfo, score_params


class Population(object):
    """Population class that implements the evolutionary algorithm."""

    def __init__(self, params_class, params_kwargs, size, objective, output_filename,
                 bottleneck=None, mutation_probability=.1, opponents=None,
                 processes=1, weights=None,
                 sample_count=None, population=None):
        self.print_output = True
        self.params_class = params_class
        self.bottleneck = bottleneck
        if processes == 0:
            self.processes = cpu_count()
        else:
            self.processes = processes

        self.pool = Pool(processes=self.processes)

        self.outputer = Outputer(output_filename, mode='a')
        self.size = size
        self.objective = objective
        if not bottleneck:
            self.bottleneck = size // 4
        else:
            self.bottleneck = bottleneck
        if opponents is None:
            self.opponents_information = [
                PlayerInfo(s, {}) for s in axl.short_run_time_strategies]
        else:
            self.opponents_information = [
                PlayerInfo(p.__class__, p.init_kwargs) for p in opponents]
        self.generation = 0

        self.params_kwargs = params_kwargs
        if "mutation_probability" not in self.params_kwargs:
            self.params_kwargs["mutation_probability"] = mutation_probability

        if population is not None:
            self.population = population
        else:
            self.population = [params_class(**params_kwargs)
                               for _ in range(self.size)]

        self.weights = weights
        self.sample_count = sample_count

    def score_all(self):
        starmap_params_zip = zip(
            self.population,
            repeat(self.objective),
            repeat(self.opponents_information),
            repeat(self.weights),
            repeat(self.sample_count))
        if self.processes == 1:
            results = list(starmap(score_params, starmap_params_zip))
        else:
            results = self.pool.starmap(score_params, starmap_params_zip)
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
        if self.print_output:
            print("Scoring Generation {}".format(self.generation))

        # Score population
        scores = self.score_all()
        results = list(zip(scores, range(len(scores))))
        results.sort(key=itemgetter(0), reverse=True)

        # Report
        if self.print_output:
            print("Generation", self.generation, "| Best Score:", results[0][0], repr(self.population[results[0][
                1]]))  # prints best result
        # Write the data
        # Note: if using this for analysis, for reproducability it may be useful to
        # pass type(opponent) for each of the opponents. This will allow verification of results post run

        row = [self.generation, mean(scores), pstdev(scores), results[0][0],
               repr(self.population[results[0][1]])]
        self.outputer.write_row(row)

        # Next Population
        indices_to_keep = [p for (s, p) in results[0: self.bottleneck]]
        self.subset_population(indices_to_keep)
        # Add mutants of the best players
        best_mutants = [p.copy() for p in self.population]
        for p in best_mutants:
            p.mutate()
            self.population.append(p)
        # Add random variants
        random_params = [self.params_class(**self.params_kwargs)
                         for _ in range(self.bottleneck // 2)]
        params_to_modify = [params.copy() for params in self.population]
        params_to_modify += random_params
        # Crossover
        size_left = self.size - len(self.population)
        params_to_modify = self.crossover(params_to_modify, size_left)
        # Mutate
        for p in params_to_modify:
            p.mutate()
        self.population += params_to_modify

    def __iter__(self):
        return self

    def __next__(self):
        self.evolve()

    def run(self, generations, print_output=True):
        self.print_output = print_output

        for _ in range(generations):
            next(self)

