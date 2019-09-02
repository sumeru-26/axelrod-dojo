import os
import tempfile
import unittest

import axelrod as axl
import axelrod_dojo as axl_dojo


class TestCyclerParams(unittest.TestCase):
    def setUp(self):
        pass

    def test_default_single_opponent_e2e(self):
        temp_file = tempfile.NamedTemporaryFile()
        # we will set the objective to be
        cycler_objective = axl_dojo.prepare_objective(name="score", turns=10, repetitions=1)

        # Lets use an opponent_list of just one:
        opponent_list = [axl.TitForTat(), axl.Calculator()]
        cycler = axl.EvolvableCycler

        # params to pass through
        cycler_kwargs = {
            "sequence_length": 10
        }

        # assert file is empty to start
        self.assertEqual(temp_file.readline(), b'')  # note that .readline() reads bytes hence b''

        population = axl_dojo.Population(player_class=cycler,
                                         params_kwargs=cycler_kwargs,
                                         size=20,
                                         objective=cycler_objective,
                                         output_filename=temp_file.name,
                                         opponents=opponent_list)

        generations = 5
        population.run(generations, print_output=False)

        # assert the output file exists and is not empty
        self.assertTrue(os.path.exists(temp_file.name))
        self.assertNotEqual(temp_file.readline(), b'')  # note that .readline() reads bytes hence b''

        # close the temp file
        temp_file.close()
