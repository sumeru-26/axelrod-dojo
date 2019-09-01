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

        self.assertTrue(np.allclose(opt_vector, np.array([[0. , 0. , 0.36158016,
                                                           0.35863128, 0. , 1.,
                                                           0.72670793, 0.67951873]])))
        self.assertEqual(abs(opt_objective_value), 4.96)

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

        self.assertTrue(np.allclose(opt_vector, np.array([0.0187898, 0.6176355,
                                                          0.61209572, 0.616934,
                                                          0.94374808, 0.6818203,
                                                          0.3595079, 0.43703195,
                                                          0.6976312, 0.06022547,
                                                          0.66676672, 0.67063787,
                                                          0.21038256, 0.1289263,
                                                          0.31542835, 0.36371077,
                                                          0.57019677])))
        self.assertEqual(abs(opt_objective_value), 1)
