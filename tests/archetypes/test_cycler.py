import unittest

import axelrod as axl
from axelrod_dojo.archetypes.cycler import CyclerParams

C, D = axl.Action


class TestCyclerParams(unittest.TestCase):
    def setUp(self):
        self.instance = None

    # Basic creation methods setting the correct params

    def test_creation_seqLen(self):
        axl.seed(0)
        test_length = 10
        self.instance = CyclerParams(sequence_length=test_length)
        self.assertEqual(self.instance.sequence, [C, D, D, C, D, D, D, D, D, D])
        self.assertEqual(self.instance.sequence_length, test_length)
        self.assertEqual(len(self.instance.sequence), test_length)

    def test_creation_seq(self):
        test_seq = [C, C, D, C, C, D, D, C, D, D]
        self.instance = CyclerParams(sequence=test_seq)
        self.assertEqual(self.instance.sequence_length, len(test_seq))
        self.assertEqual(self.instance.sequence, test_seq)

    def test_crossover_even_length(self):
        # Even test
        test_seq_1 = [C] * 6
        test_seq_2 = [D] * 6
        result_seq = [C, C, D, D, D, D]

        self.instance = CyclerParams(sequence=test_seq_1)
        instance_two = CyclerParams(sequence=test_seq_2)
        out_cycler = self.instance.crossover(instance_two, in_seed=1)
        self.assertEqual(result_seq, out_cycler.sequence)

    def test_crossover_odd_length(self):
        # Odd Test
        test_seq_1 = [C] * 7
        test_seq_2 = [D] * 7
        result_seq = [C, C, D, D, D, D, D]

        self.instance = CyclerParams(sequence=test_seq_1)
        instance_two = CyclerParams(sequence=test_seq_2)
        out_cycler = self.instance.crossover(instance_two, in_seed=1)
        self.assertEqual(result_seq, out_cycler.sequence)

    def test_mutate(self):
        test_seq = [C, D, D, C, C, D, D]
        self.instance = CyclerParams(sequence=test_seq, mutation_probability=1)
        self.instance.mutate()
        # these are dependent on each other but testing both will show that we haven't just removed a gene
        self.assertEqual(len(test_seq), self.instance.sequence_length)
        self.assertNotEqual(test_seq, self.instance.sequence)

    def test_copy(self):
        test_seq = [C, D, D, C, C, D, D]
        self.instance = CyclerParams(sequence=test_seq)
        instance_two = self.instance.copy()
        self.assertFalse(self.instance is instance_two)
