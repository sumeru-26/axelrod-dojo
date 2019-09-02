import axelrod as axl
import unittest

from axelrod_dojo import FSMParams

C, D = axl.Action.C, axl.Action.D


class TestFSMParams(unittest.TestCase):


    def test_randomize(self):
        num_states = 2
        rows = [[0, C, 1, D], [0, D, 0, D], [1, C, 1, C], [1, D, 1, D]]
        fsm_params = FSMParams(num_states=num_states, rows=rows)

        axl.seed(5)
        fsm_params.randomize()
        self.assertNotEqual(rows, fsm_params.rows)
        self.assertNotEqual(C, fsm_params.initial_action)
        self.assertNotEqual(0, fsm_params.initial_state)

    def test_crossover_rows(self):
        num_states = 2
        fsm_params = FSMParams(num_states=num_states)

        rows1 = [[0, C, 1, D], [0, D, 0, D], [1, C, 1, C], [1, D, 1, D]]
        rows2 = [[0, D, 1, C], [0, C, 0, C], [1, D, 1, D], [1, C, 1, C]]

        axl.seed(0)
        rows = fsm_params.crossover_rows(rows1, rows2)
        self.assertEqual(rows, rows1[:2] + rows2[2:])

        axl.seed(1)
        rows = fsm_params.crossover_rows(rows1, rows2)
        self.assertEqual(rows, rows1[:0] + rows2[0:])

    def test_crossover(self):
        num_states = 2

        rows1 = [[0, C, 1, D], [0, D, 0, D], [1, C, 1, C], [1, D, 1, D]]
        fsm_params1 = FSMParams(num_states=num_states, rows=rows1)

        rows2 = [[0, D, 1, C], [0, C, 0, C], [1, D, 1, D], [1, C, 1, C]]
        fsm_params2 = FSMParams(num_states=num_states, rows=rows2)

        axl.seed(0)
        fsm_params = fsm_params1.crossover(fsm_params2)
        self.assertEqual(fsm_params.rows,
                         fsm_params1.rows[:2] + fsm_params2.rows[2:])

        self.assertEqual(fsm_params.PlayerClass, fsm_params1.PlayerClass)
        self.assertEqual(fsm_params.num_states, fsm_params1.num_states)
        self.assertEqual(fsm_params.initial_action,
                         fsm_params1.initial_action)
        self.assertEqual(fsm_params.initial_state,
                         fsm_params1.initial_state)

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

