import unittest
import tempfile
import csv

import axelrod as axl
import axelrod_dojo as dojo

C, D = axl.Action.C, axl.Action.D

class TestFSMPopulation(unittest.TestCase):
    temporary_file = tempfile.NamedTemporaryFile()

    def test_score(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_states = 2
        mutation_rate = .1
        opponents = [s() for s in axl.demo_strategies]
        size = 10

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        population = dojo.Population(params_class=dojo.FSMParams,
                                     params_args=(num_states, mutation_rate),
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     bottleneck=2,
                                     processes=1)

        generations = 4
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 4)

        # Manually read from tempo file to find best strategy
        best_score, best_params = 0, None
        with open(self.temporary_file.name, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                _, mean_score, sd_score, max_score, arg_max = row
                if float(max_score) > best_score:
                    best_score = float(max_score)
                    best_params = arg_max

        # Test the load params function
        for num in range(1, 4 + 1):
            best = dojo.load_params(params_class=dojo.FSMParams,
                                    filename=self.temporary_file.name,
                                    num=num)
            self.assertEqual(len(best), num)

            for parameters in best:
                self.assertIsInstance(parameters, dojo.FSMParams)

        # Test that can use these loaded params in a new algorithm instance
        population = dojo.Population(params_class=dojo.FSMParams,
                                     params_args=(num_states, mutation_rate),
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     population=best,
                                     bottleneck=2,
                                     processes=1)
        generations = 4
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 4)

    def test_score_with_weights(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_states = 2
        mutation_rate = .1
        opponents = [s() for s in axl.demo_strategies]
        size = 10

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        population = dojo.Population(params_class=dojo.FSMParams,
                                     params_args=(num_states, mutation_rate),
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     weights=[5, 1, 1, 1, 1],
                                     bottleneck=2,
                                     processes=1)

        generations = 4
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 4)

        # Manually read from tempo file to find best strategy
        best_score, best_params = 0, None
        with open(self.temporary_file.name, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                _, mean_score, sd_score, max_score, arg_max = row
                if float(max_score) > best_score:
                    best_score = float(max_score)
                    best_params = arg_max

        # Test the load params function
        for num in range(1, 4 + 1):
            best = dojo.load_params(params_class=dojo.FSMParams,
                                    filename=self.temporary_file.name,
                                    num=num)
            self.assertEqual(len(best), num)

            for parameters in best:
                self.assertIsInstance(parameters, dojo.FSMParams)

            self.assertEqual(best[0].__repr__(), best_params)

    def test_score_with_sample_count(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_states = 2
        mutation_rate = .1
        opponents = [s() for s in axl.demo_strategies]
        size = 10

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        population = dojo.Population(params_class=dojo.FSMParams,
                                     params_args=(num_states, mutation_rate),
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     sample_count=2,  # Randomly sample 2 opponents at each step
                                     bottleneck=2,
                                     processes=1)

        generations = 4
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 4)

        # Manually read from tempo file to find best strategy
        best_score, best_params = 0, None
        with open(self.temporary_file.name, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                _, mean_score, sd_score, max_score, arg_max = row
                if float(max_score) > best_score:
                    best_score = float(max_score)
                    best_params = arg_max

        # Test the load params function
        for num in range(1, 4 + 1):
            best = dojo.load_params(params_class=dojo.FSMParams,
                                    filename=self.temporary_file.name,
                                    num=num)
            self.assertEqual(len(best), num)

            for parameters in best:
                self.assertIsInstance(parameters, dojo.FSMParams)

            self.assertEqual(best[0].__repr__(), best_params)


    def test_score_with_sample_count_and_weights(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_states = 2
        mutation_rate = .1
        opponents = [s() for s in axl.demo_strategies]
        size = 10

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        population = dojo.Population(params_class=dojo.FSMParams,
                                     params_args=(num_states, mutation_rate),
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     sample_count=2,  # Randomly sample 2 opponents at each step
                                     weights=[5, 1, 1, 1, 1],
                                     bottleneck=2,
                                     processes=1)

        generations = 4
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 4)

        # Manually read from tempo file to find best strategy
        best_score, best_params = 0, None
        with open(self.temporary_file.name, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                _, mean_score, sd_score, max_score, arg_max = row
                if float(max_score) > best_score:
                    best_score = float(max_score)
                    best_params = arg_max

        # Test the load params function
        for num in range(1, 4 + 1):
            best = dojo.load_params(params_class=dojo.FSMParams,
                                    filename=self.temporary_file.name,
                                    num=num)
            self.assertEqual(len(best), num)

            for parameters in best:
                self.assertIsInstance(parameters, dojo.FSMParams)

            self.assertEqual(best[0].__repr__(), best_params)

    def test_score_with_particular_players(self):
        """
        These are players that are known to be difficult to pickle
        """
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_states = 2
        mutation_rate = .1
        opponents = [axl.ThueMorse(),
                     axl.MetaHunter(),
                     axl.BackStabber(),
                     axl.Alexei()]
        size = 10

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        population = dojo.Population(params_class=dojo.FSMParams,
                                     params_args=(num_states, mutation_rate),
                                     size=size,
                                     objective=objective,
                                     output_filename=self.temporary_file.name,
                                     opponents=opponents,
                                     bottleneck=2,
                                     processes=0)

        generations = 4
        axl.seed(0)
        population.run(generations)
        self.assertEqual(population.generation, 4)
