import unittest

import io
import tempfile
import functools

import axelrod as axl
import axelrod_dojo.utils as utils


class TestOutputer(unittest.TestCase):
    temporary_file = tempfile.NamedTemporaryFile()
    outputer = utils.Outputer(filename=temporary_file.name)

    def test_init(self):
        self.assertIsInstance(self.outputer.output, io.TextIOWrapper)
        self.assertEqual(str(type(self.outputer.writer)),
                        "<class '_csv.writer'>")

    def test_write(self):
        self.assertIsNone(self.outputer.write([1, 2, 3]))
        with open(self.temporary_file.name, "r") as f:
            self.assertEqual("1,2,3\n", f.read())


class TestPrepareObjective(unittest.TestCase):
    def test_incorrect_objective_name(self):
        name = "not_correct_name"
        with self.assertRaises(ValueError):
            utils.prepare_objective(name=name)

    def test_score(self):
        objective = utils.prepare_objective(
                                    name="score",
                                    turns=200,
                                    noise=0,
                                    match_attributes={"length": float("inf")},
                                    repetitions=5)
        self.assertIsInstance(objective, functools.partial)
        self.assertIn("objective_score ", str(objective))

    def test_score_diff(self):
        objective = utils.prepare_objective(name="score_diff",
                                            turns=200,
                                            noise=0,
                                            repetitions=5)
        self.assertIsInstance(objective, functools.partial)
        self.assertIn("objective_score_diff ", str(objective))

    def test_moran(self):
        objective = utils.prepare_objective(name="moran",
                                            turns=200,
                                            noise=0,
                                            repetitions=5)
        self.assertIsInstance(objective, functools.partial)
        self.assertIn("objective_moran_win ", str(objective))


class TestObjectiveScore(unittest.TestCase):
    def test_deterministic_player_opponent(self):
        player = axl.TitForTat()
        opponent = axl.Defector()
        expected_scores = [1 / 2]
        scores = utils.objective_score(player,
                                       opponent,
                                       turns=2,
                                       repetitions=5,
                                       noise=0)
        self.assertEqual(expected_scores, scores)

        player = axl.TitForTat()
        opponent = axl.Alternator()
        expected_scores = [8 / 3]
        scores = utils.objective_score(player,
                                       opponent,
                                       turns=3,
                                       repetitions=5,
                                       noise=0)
        self.assertEqual(expected_scores, scores)

    def test_noisy_match(self):
        axl.seed(0)
        player = axl.Cooperator()
        opponent = axl.Defector()
        scores = utils.objective_score(player,
                                       opponent,
                                       turns=2,
                                       repetitions=3,
                                       noise=.8)
        # Stochastic so more repetitions are run
        self.assertEqual(len(scores), 3)
        # Cooperator should score 0 but noise implies it scores more
        self.assertNotEqual(max(scores), 0)


class TestObjectiveScoreDiff(unittest.TestCase):
    def test_deterministic_player_opponent(self):
        player = axl.TitForTat()
        opponent = axl.Defector()
        expected_score_diffs = [1 / 2 - 6 / 2]
        score_diffs = utils.objective_score_diff(player,
                                                 opponent,
                                                 turns=2,
                                                 repetitions=5,
                                                 noise=0)
        self.assertEqual(expected_score_diffs, score_diffs)

        player = axl.TitForTat()
        opponent = axl.Alternator()
        expected_score_diffs = [8 / 3 - 8 / 3]
        score_diffs = utils.objective_score_diff(player,
                                                 opponent,
                                                 turns=3,
                                                 repetitions=5,
                                                 noise=0)
        self.assertEqual(expected_score_diffs, score_diffs)

    def test_noisy_match(self):
        axl.seed(0)
        player = axl.Cooperator()
        opponent = axl.Defector()
        score_diffs = utils.objective_score_diff(player,
                                                 opponent,
                                                 turns=2,
                                                 repetitions=3,
                                                 noise=.8)
        # Stochastic so more repetitions are run
        self.assertEqual(len(score_diffs), 3)
        # Cooperator should score 0 (for a diff of -5)
        # but noise implies it scores more
        self.assertNotEqual(max(score_diffs), -5)


class TestObjectiveMoran(unittest.TestCase):
    def test_deterministic_cooperator_never_fixes(self):
        player = axl.Cooperator()
        opponent = axl.Defector()
        expected_fixation_probabilities = [0 for _ in range(5)]
        fixation_probabilities = utils.objective_moran_win(player,
                                                           opponent,
                                                           turns=2,
                                                           repetitions=5,
                                                           noise=0)
        self.assertEqual(fixation_probabilities,
                         expected_fixation_probabilities)

    def test_stochastic_outcomes(self):
        for seed, expected in zip(range(2), [[0, 1, 0, 1, 0], [0, 0, 0, 1, 1]]):
            axl.seed(seed)
            player = axl.TitForTat()
            opponent = axl.Defector()
            expected_fixation_probabilities = expected
            fixation_probabilities = utils.objective_moran_win(player,
                                                               opponent,
                                                               turns=3,
                                                               repetitions=5,
                                                               noise=0)
            self.assertEqual(fixation_probabilities,
                             expected_fixation_probabilities)


class TestBaseParametersClass(unittest.TestCase):
    def test_null_methods(self):
        parameters = utils.Params()
        self.assertIsNone(parameters.mutate(mutation_rate=.1))
        self.assertIsNone(parameters.random())
        self.assertIsNone(parameters.__repr__())
        self.assertIsNone(parameters.from_repr())
        self.assertIsNone(parameters.copy())
        self.assertIsNone(parameters.player())
        self.assertIsNone(parameters.params())
        other = axl.Player()
        self.assertIsNone(parameters.crossover(other))
        vector = [0.2, 0.4, 0.6, 0.8]
        self.assertIsNone(parameters.receive_vector(vector))
        self.assertIsNone(parameters.vector_to_instance())
        self.assertIsNone(parameters.create_vector_bounds())


class DummyParams(utils.Params):
    """
    Dummy Params class for testing purposes
    """
    def player(self):
        return axl.Cooperator()


class TestScoreParams(unittest.TestCase):
    def test_score(self):
        axl.seed(0)
        opponents_information = [utils.PlayerInfo(s, {})
                                 for s in axl.demo_strategies]
        objective = utils.prepare_objective()
        params = DummyParams()
        score = utils.score_params(params,
                                   objective=objective,
                                   opponents_information=opponents_information)
        expected_score = 2.0949
        self.assertEqual(score, expected_score)

    def test_with_init_kwargs(self):
        axl.seed(0)
        opponents_information = [utils.PlayerInfo(axl.Random, {"p": 0})]
        objective = utils.prepare_objective()
        params = DummyParams()
        score = utils.score_params(params,
                                   objective=objective,
                                   opponents_information=opponents_information)
        expected_score = 0
        self.assertEqual(score, expected_score)

        opponents_information = [utils.PlayerInfo(axl.Random, {"p": 1})]
        objective = utils.prepare_objective()
        params = DummyParams()
        score = utils.score_params(params,
                                   objective=objective,
                                   opponents_information=opponents_information)
        expected_score = 3.0
        self.assertEqual(score, expected_score)

    def test_score_with_weights(self):
        axl.seed(0)
        opponents_information = [utils.PlayerInfo(s, {})
                                 for s in axl.demo_strategies]
        objective = utils.prepare_objective()
        params = DummyParams()
        score = utils.score_params(params,
                                   objective=objective,
                                   opponents_information=opponents_information,
                                   # All weight on Coop
                                   weights=[1, 0, 0, 0, 0])
        expected_score = 3
        self.assertEqual(score, expected_score)

        score = utils.score_params(params,
                                   objective=objective,
                                   opponents_information=opponents_information,
                                   # Shared weight between Coop and Def
                                   weights=[2, 2, 0, 0, 0])
        expected_score = 1.5
        self.assertEqual(score, expected_score)

        score = utils.score_params(params,
                                   objective=objective,
                                   opponents_information=opponents_information,
                                   # Shared weight between Coop and Def
                                   weights=[2, -.5, 0, 0, 0])
        expected_score = 4.0
        self.assertEqual(score, expected_score)
