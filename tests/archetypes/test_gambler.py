import axelrod as axl
import random
import unittest

from axelrod_dojo import GamblerParams
from axelrod.strategies.lookerup import Plays
from axelrod.strategies.lookerup import create_lookup_table_keys

C, D = axl.Action.C, axl.Action.D


class TestGamblerParams(unittest.TestCase):

    def test_repr(self):
        plays, op_plays, op_start_plays = 1, 1, 1

        pattern = [0.05, 0.07, 0.70, 0.03,
                   0.60, 0.70, 0.45, 0.10]

        gambler_params = GamblerParams(plays=plays, op_plays=op_plays,
                                       op_start_plays=op_start_plays,
                                       pattern=pattern)

        self.assertEqual(gambler_params.__repr__(),
                         '1:1:1:0.05|0.07|0.7|0.03|0.6|0.7|0.45|0.1')

    def test_genes_in_bounds(self):
        gambler_params = GamblerParams(plays=1, op_plays=1, op_start_plays=1,
                                       mutation_probability=0.8)

        for _ in range(100):
            _ = gambler_params.mutate()

        self.assertTrue(all(gene <= 1 for gene in gambler_params.pattern))
        self.assertTrue(all(gene >= 0 for gene in gambler_params.pattern))
