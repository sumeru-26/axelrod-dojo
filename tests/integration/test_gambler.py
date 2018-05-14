import unittest
import tempfile
import csv

import axelrod as axl
import axelrod_dojo as dojo

C, D = axl.Action.C, axl.Action.D

class TestFSMPopulation(unittest.TestCase):
    name = "score"
    turns = 10
    noise = 0
    repetitions = 5
    num_states = 2
    opponents = [s() for s in axl.demo_strategies]
    size = 10

    objective = dojo.prepare_objective(name=name,
                                       turns=turns,
                                       noise=noise,
                                       repetitions=repetitions)

    def test_generations(self):
        temporary_file = tempfile.NamedTemporaryFile()

        params_kwargs = {"plays": 1, "op_plays": 1, "op_start_plays": 2}
        population = dojo.Population(params_class=dojo.GamblerParams,
                                     params_kwargs=params_kwargs,
                                     size=self.size,
                                     objective=self.objective,
                                     output_filename=temporary_file.name,
                                     opponents=self.opponents,
                                     bottleneck=2,
                                     mutation_probability=.01,
                                     processes=1)

        generations = 4
        axl.seed(0)
        population.run(generations, print_output=False)
        self.assertEqual(population.generation, generations)

        results = []
        with open(temporary_file.name, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                results.append(row)

        self.assertEqual(population.generation, len(results))

    def test_scores(self):
        temporary_file = tempfile.NamedTemporaryFile()
        params_kwargs = {"plays": 1, "op_plays": 1, "op_start_plays": 2}
        population = dojo.Population(params_class=dojo.GamblerParams,
                                     params_kwargs=params_kwargs,
                                     size=self.size,
                                     objective=self.objective,
                                     output_filename=temporary_file.name,
                                     opponents=self.opponents,
                                     bottleneck=2,
                                     mutation_probability=.01,
                                     processes=1)

        generations = 10
        axl.seed(1)
        population.run(generations, print_output=False)

        results = []
        with open(temporary_file.name, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                results.append(row)

        self.assertTrue(all([0<=float(result[3]) <= 5 for result in results]))
