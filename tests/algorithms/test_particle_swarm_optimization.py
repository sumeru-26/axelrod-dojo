import functools
import unittest
import numpy as np

import axelrod as axl
from axelrod import EvolvableGambler, EvolvableFSMPlayer
import axelrod_dojo as dojo
from axelrod_dojo import prepare_objective
from axelrod_dojo.algorithms.particle_swarm_optimization import PSO


class TestPSO(unittest.TestCase):
    def test_init_default(self):
        params = [1, 1, 1]
        objective = prepare_objective('score', 2, 0, 1, nmoran=False)

        pso = PSO(EvolvableGambler, params, objective=objective)

        self.assertIsInstance(pso.objective, functools.partial)
        self.assertEqual(len(pso.opponents_information), len(axl.short_run_time_strategies))
        self.assertEqual(pso.population, 1)
        self.assertEqual(pso.generations, 1)
        self.assertTrue(pso.debug)
        self.assertEqual(pso.phip, 0.8)
        self.assertEqual(pso.phig, 0.8)
        self.assertEqual(pso.omega, 0.8)
        self.assertEqual(pso.processes, 1)

    def test_init(self):
        params = [2, 1, 1]
        objective = prepare_objective('score', 2, 0, 1, nmoran=False)
        opponents = [axl.Defector(), axl.Cooperator()]
        population = 2
        generations = 10
        debug = False
        phip = 0.6
        phig = 0.6
        omega = 0.6

        pso = PSO(EvolvableGambler, params, objective=objective,
                  opponents=opponents, population=population,
                  generations=generations, debug=debug, phip=phip, phig=phig,
                  omega=omega)

        self.assertIsInstance(pso.objective, functools.partial)
        self.assertEqual(len(pso.opponents_information), len(opponents))
        self.assertEqual(pso.population, population)
        self.assertEqual(pso.generations, generations)
        self.assertFalse(pso.debug)
        self.assertEqual(pso.phip, phip)
        self.assertEqual(pso.phig, phig)
        self.assertEqual(pso.omega, omega)
        self.assertEqual(pso.processes, 1)

    def test_pso_with_gambler(self):
        name = "score"
        turns = 50
        noise = 0
        repetitions = 5
        num_plays = 1
        num_op_plays = 1
        num_op_start_plays = 1
        params_kwargs = {"parameters": (num_plays, num_op_plays, num_op_start_plays)}
        population = 10
        generations = 100
        opponents = [axl.Cooperator() for _ in range(5)]

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        pso = dojo.PSO(EvolvableGambler, params_kwargs, objective=objective, debug=False,
                       opponents=opponents, population=population, generations=generations)

        axl.seed(0)
        opt_vector, opt_objective_value = pso.swarm()

        self.assertTrue(np.allclose(opt_vector, np.array([[
            0., 0.89327795, 0.04140453, 0.73676965, 0., 0.20622436, 1., 0.68104353]])))
        self.assertEqual(abs(opt_objective_value), 3.56)

    def test_pso_with_fsm(self):
        name = "score"
        turns = 10
        noise = 0
        repetitions = 5
        num_states = 4
        params_kwargs = {"num_states": num_states}
        population = 10
        generations = 100
        opponents = [axl.Defector() for _ in range(5)]

        objective = dojo.prepare_objective(name=name,
                                           turns=turns,
                                           noise=noise,
                                           repetitions=repetitions)

        pso = PSO(EvolvableFSMPlayer, params_kwargs, objective=objective, debug=False,
                  opponents=opponents, population=population, generations=generations)

        axl.seed(0)
        opt_vector, opt_objective_value = pso.swarm()

        self.assertTrue(np.allclose(
            opt_vector,
            np.array([
                0.22825439, 0.06954976, 0.49462006, 0.27704876, 1., 0.81240316, 0.11818378, 0., 0.4289995,
                0.91397724, 1., 0.7404604, 0.35865552, 1., 0.53483268, 0.41643427, 0.71756716])))
        self.assertEqual(abs(opt_objective_value), 1)
