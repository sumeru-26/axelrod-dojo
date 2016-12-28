from statistics import mean

import axelrod as axl


def objective_match_score(me, other, turns, noise):
    """Objective function to maximize total score over matches."""
    match = axl.Match((me, other), turns=turns, noise=noise)
    if match._stochastic:
        repetitions = 20
    else:
        repetitions = 1
    scores_for_this_opponent = []

    for _ in range(repetitions):
        match.play()
        scores_for_this_opponent.append(match.final_score_per_turn()[0])
    return scores_for_this_opponent

def objective_match_score_difference(me, other, turns, noise):
    """Objective function to maximize total score difference over matches."""
    match = axl.Match((me, other), turns=turns, noise=noise)
    if match._stochastic:
        repetitions = 20
    else:
        repetitions = 1
    scores_for_this_opponent = []

    for _ in range(repetitions):
        match.play()
        final_scores = match.final_score_per_turn()
        score_diff = final_scores[0] - final_scores[1]
        scores_for_this_opponent.append(score_diff)
    return scores_for_this_opponent

def objective_match_moran_win(me, other, turns, noise=0, repetitions=100):
    """Objective function to maximize Moran fixations over N=4 matches"""
    assert(noise == 0)
    # N = 4 population
    population = (me, me.clone(), other, other.clone())
    mp = axl.MoranProcess(population, turns=turns, noise=noise)

    scores_for_this_opponent = []

    for _ in range(repetitions):
        mp.play()
        if mp.winning_strategy_name == str(me):
            scores_for_this_opponent.append(1)
        else:
            scores_for_this_opponent.append(0)
    return scores_for_this_opponent

def score_for(my_strategy_factory, args=None, opponents=None, turns=200,
              noise=0., objective=objective_match_score):
    """
    Given a function that will return a strategy, calculate the average score
    per turn against all ordinary strategies. If the opponent is classified as
    stochastic, then run 100 repetitions and take the average to get a good
    estimate.
    """
    if not args:
        args = []
    scores_for_all_opponents = []
    if not opponents:
        # opponents = [s for s in axl.strategies]
        opponents = axl.short_run_time_strategies

    for opponent in opponents:
        me = my_strategy_factory(*args)
        other = opponent()
        scores_for_this_opponent = objective(me, other, turns, noise)
        mean_vs_opponent = mean(scores_for_this_opponent)
        scores_for_all_opponents.append(mean_vs_opponent)

    # Calculate the average for all opponents
    overall_mean_score = mean(scores_for_all_opponents)
    return overall_mean_score
