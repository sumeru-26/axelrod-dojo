import tempfile
import unittest

import axelrod as axl
import axelrod_dojo as axl_dojo


class TestPopulationSizes(unittest.TestCase):

    def test_basic_pop_size(self):
        # Set up Tmp file
        temp_file = tempfile.NamedTemporaryFile()
        # we will set the objective to be
        cycler_objective = axl_dojo.prepare_objective(name="score", turns=10, repetitions=1)
        # Lets use an opponent_list of just one:
        opponent_list = [axl.TitForTat()]
        # params to pass through
        cycler_kwargs = {
            "cycle_length": 10
        }

        population_size = 20
        population = axl_dojo.Population(player_class=axl.EvolvableCycler,
                                         params_kwargs=cycler_kwargs,
                                         size=population_size,
                                         objective=cycler_objective,
                                         output_filename=temp_file.name,
                                         opponents=opponent_list)

        # Before run
        self.assertEqual(len(population.population), population_size)

        # After Run
        population.run(generations=5, print_output=False)
        self.assertEqual(len(population.population), population_size)

        # close the temp file
        temp_file.close()

    def test_bottleneck_pop_size(self):
        # Set up Tmp file
        temp_file = tempfile.NamedTemporaryFile()
        # we will set the objective to be
        cycler_objective = axl_dojo.prepare_objective(name="score", turns=10, repetitions=1)
        # Lets use an opponent_list of just one:
        opponent_list = [axl.TitForTat()]
        # params to pass through
        cycler_kwargs = {
            "cycle_length": 10
        }

        population_size = 20
        population = axl_dojo.Population(player_class=axl.EvolvableCycler,
                                         params_kwargs=cycler_kwargs,
                                         size=population_size,
                                         bottleneck=1,
                                         objective=cycler_objective,
                                         output_filename=temp_file.name,
                                         opponents=opponent_list)

        # Before run
        self.assertEqual(len(population.population), population_size)

        # After Run
        population.run(generations=5, print_output=False)
        self.assertEqual(len(population.population), population_size)

        # close the temp file
        temp_file.close()
