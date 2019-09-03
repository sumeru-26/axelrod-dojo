import unittest
import axelrod as axl

from axelrod_dojo import HMMParams

C, D = axl.Action.C, axl.Action.D


class TestHMMParams(unittest.TestCase):
    def test_init(self):
        # Test with TFT.
        t_C = [[1, 0], [1, 0]]
        t_D = [[0, 1], [0, 1]]
        p = [1, 0]
        num_states = 2
        hmm_params = HMMParams(num_states=num_states,
                               transitions_C=t_C,
                               transitions_D=t_D,
                               emission_probabilities=p,
                               initial_state=0,
                               initial_action=C)
        self.assertEqual(hmm_params.PlayerClass, axl.HMMPlayer)
        self.assertEqual(hmm_params.num_states, num_states)
        self.assertEqual(hmm_params.mutation_probability, 2.5)
        self.assertEqual(hmm_params.transitions_C, t_C)
        self.assertEqual(hmm_params.transitions_D, t_D)
        self.assertEqual(hmm_params.emission_probabilities, p)
        self.assertEqual(hmm_params.initial_state, 0)
        self.assertEqual(hmm_params.initial_action, C)

    def test_init_without_defaults(self):
        """
        Check that by default have random rows.
        """
        num_states = 2
        transitions_Cs, transitions_Ds = [], []
        for seed in range(2):
            axl.seed(seed)
            hmm_params = HMMParams(num_states=num_states)
            transitions_Cs.append(hmm_params.transitions_C)
            transitions_Ds.append(hmm_params.transitions_D)
        self.assertNotEqual(*transitions_Cs)
        self.assertNotEqual(*transitions_Ds)

    def test_player(self):
        # Test with TFT.
        t_C = [[1, 0], [1, 0]]
        t_D = [[0, 1], [0, 1]]
        p = [1, 0]
        num_states = 2
        hmm_params = HMMParams(num_states=num_states,
                               transitions_C=t_C,
                               transitions_D=t_D,
                               emission_probabilities=p,
                               initial_state=0,
                               initial_action=C)
        self.assertEqual(hmm_params.player(),
                         axl.HMMPlayer(transitions_C=t_C,
                                       transitions_D=t_D,
                                       emission_probabilities=p,
                                       initial_state=0,
                                       initial_action=C))

    def test_repr_rows(self):
        num_states = 2
        hmm_params = HMMParams(num_states=num_states)

        t_C = [[1, 0], [1, 0]]
        string_repr = hmm_params.repr_rows(t_C)
        self.assertEqual(string_repr, '1_0|1_0')

    def test_repr(self):
        t_C = [[1, 0], [1, 0]]
        t_D = [[0, 1], [0, 1]]
        p = [1, 0]
        num_states = 2
        hmm_params = HMMParams(num_states=num_states,
                               transitions_C=t_C,
                               transitions_D=t_D,
                               emission_probabilities=p,
                               initial_state=0,
                               initial_action=C)

        self.assertEqual(hmm_params.__repr__(),
                         '0:C:1_0|1_0:0_1|0_1:1_0')

    def test_parse_repr(self):
        t_C = [[1, 0], [1, 0]]
        t_D = [[0, 1], [0, 1]]
        p = [1, 0]
        string_repr = '0:C:1_0|1_0:0_1|0_1:1_0'
        parameters = HMMParams.parse_repr(string_repr)
        self.assertEqual(parameters.initial_state, 0)
        self.assertEqual(parameters.initial_action, C)
        self.assertEqual(parameters.transitions_C, t_C)
        self.assertEqual(parameters.transitions_D, t_D)
        self.assertEqual(parameters.emission_probabilities, p)
