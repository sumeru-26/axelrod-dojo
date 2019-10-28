import unittest

import tempfile
import functools

import axelrod as axl
import axelrod_dojo.utils as utils


class TestOutputer(unittest.TestCase):
    temporary_file = tempfile.NamedTemporaryFile()
    outputer = utils.Outputer(filename=temporary_file.name, mode='a')

    def test_init(self):
        self.assertEqual(self.outputer.file, self.temporary_file.name)
        self.assertEqual(self.outputer.mode, 'a')

    def test_write_and_clear(self):
        writing_line = [1, "something", 3.0]
        self.outputer.write_row(writing_line)
        self.outputer.write_row(writing_line)
        with open(self.temporary_file.name, "r") as f:
            self.assertEqual("1,something,3.0\n1,something,3.0\n", f.read())


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


class TestScoreParams(unittest.TestCase):
    def test_score(self):
        axl.seed(0)
        opponents_information = [utils.PlayerInfo(s, {})
                                 for s in axl.demo_strategies]
        objective = utils.prepare_objective()
        score = utils.score_player(axl.Cooperator(),
                                   objective=objective,
                                   opponents_information=opponents_information)
        expected_score = 2.0949
        self.assertEqual(score, expected_score)

    def test_with_init_kwargs(self):
        axl.seed(0)
        opponents_information = [utils.PlayerInfo(axl.Random, {"p": 0})]
        objective = utils.prepare_objective()
        score = utils.score_player(axl.Cooperator(),
                                   objective=objective,
                                   opponents_information=opponents_information)
        expected_score = 0
        self.assertEqual(score, expected_score)

        opponents_information = [utils.PlayerInfo(axl.Random, {"p": 1})]
        objective = utils.prepare_objective()
        score = utils.score_player(axl.Cooperator(),
                                   objective=objective,
                                   opponents_information=opponents_information)
        expected_score = 3.0
        self.assertEqual(score, expected_score)

    def test_score_with_weights(self):
        axl.seed(0)
        opponents_information = [utils.PlayerInfo(s, {})
                                 for s in axl.demo_strategies]
        objective = utils.prepare_objective()
        score = utils.score_player(axl.Cooperator(),
                                   objective=objective,
                                   opponents_information=opponents_information,
                                   # All weight on Coop
                                   weights=[1, 0, 0, 0, 0])
        expected_score = 3
        self.assertEqual(score, expected_score)

        score = utils.score_player(axl.Cooperator(),
                                   objective=objective,
                                   opponents_information=opponents_information,
                                   # Shared weight between Coop and Def
                                   weights=[2, 2, 0, 0, 0])
        expected_score = 1.5
        self.assertEqual(score, expected_score)

        score = utils.score_player(axl.Cooperator(),
                                   objective=objective,
                                   opponents_information=opponents_information,
                                   # Shared weight between Coop and Def
                                   weights=[2, -.5, 0, 0, 0])
        expected_score = 4.0
        self.assertEqual(score, expected_score)
