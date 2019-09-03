import axelrod as axl
import unittest

from axelrod_dojo import FSMParams

C, D = axl.Action.C, axl.Action.D


class TestFSMParams(unittest.TestCase):

    def test_repr_rows(self):
        num_states = 2
        fsm_params = FSMParams(num_states=num_states)

        rows = [[0, C, 1, D], [0, D, 0, D], [1, C, 1, C], [1, D, 1, D]]
        string_repr = fsm_params.repr_rows(rows)
        self.assertEqual(string_repr, '0_C_1_D:0_D_0_D:1_C_1_C:1_D_1_D')

    def test_repr(self):
        num_states = 2
        rows = [[0, C, 1, D], [0, D, 0, D], [1, C, 1, C], [1, D, 1, D]]
        fsm_params = FSMParams(num_states=num_states, rows=rows)

        self.assertEqual(fsm_params.__repr__(),
                         '0:C:0_C_1_D:0_D_0_D:1_C_1_C:1_D_1_D')

    def test_parse_repr(self):
        rows = [[0, C, 1, D], [0, D, 0, D], [1, C, 1, C], [1, D, 1, D]]
        string_repr = '0:C:0_C_1_D:0_D_0_D:1_C_1_C:1_D_1_D'
        parameters = FSMParams.parse_repr(string_repr)
        self.assertEqual(parameters.initial_state, 0)
        self.assertEqual(parameters.initial_action, C)
        self.assertEqual(parameters.rows, rows)

        string_repr = '1:D:0_C_1_D:0_D_0_D:1_C_1_C:1_D_1_D'
        parameters = FSMParams.parse_repr(string_repr)
        self.assertEqual(parameters.initial_state, 1)
        self.assertEqual(parameters.initial_action, D)
        self.assertEqual(parameters.rows, rows)

