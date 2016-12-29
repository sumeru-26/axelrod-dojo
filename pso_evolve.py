"""
Particle Swarm strategy training code.

Original version by Georgios Koutsovoulos @GDKO :
  https://gist.github.com/GDKO/60c3d0fd423598f3c4e4
Based on Martin Jones @mojones original LookerUp code

Usage:
    pso_evolve.py [-h] [-p PLAYS] [-s STARTING_PLAYS] [-g GENERATIONS]
    [-i PROCESSORS] [-o OPP_PLAYS] [-n NOISE]

Options:
    -h --help             show help
    -p PLAYS              number of recent plays in the lookup table [default: 2]
    -o OPP_PLAYS          number of recent opponent's plays in the lookup table [default: 2]
    -s STARTING_PLAYS     number of opponent starting plays in the lookup table [default: 2]
    -i PROCESSORS         number of processors to use [default: 1]
    -n NOISE              match noise [default: 0.0]
"""

from docopt import docopt
import pyswarm

from axelrod import Gambler
from axelrod.strategies.lookerup import (
    create_lookup_table_keys, create_lookup_table_from_pattern)
from axelrod_utils import score_for, objective_match_score, objective_moran_win


def optimizepso(plays, opp_plays, opp_start_plays, noise=0):
    def f(pattern):
        lookup_table = create_lookup_table_from_pattern(
            plays=plays, opp_plays=opp_plays,
            opponent_start_plays=opp_start_plays,
            pattern=pattern)
        return -score_for(Gambler, [lookup_table], noise=noise,
                          objective=objective_match_score)
    return f

if __name__ == "__main__":
    arguments = docopt(__doc__, version='PSO Evolve 0.2')
    plays = int(arguments['-p'])
    opp_plays = int(arguments['-o'])
    start_plays = int(arguments['-s'])
    noise = float(arguments['-n'])

    size = len(create_lookup_table_keys(plays, opp_plays, start_plays))
    lb = [0] * size
    ub = [1] * size

    optimizer = optimizepso(plays, opp_plays, start_plays, noise=noise)

    # There is a multiprocessing version (0.7) of pyswarm available at
    # https://github.com/tisimst/pyswarm, just pass processes=X
    # Pip installs version 0.6
    if pyswarm.__version__ == "0.7":
        processes = int(arguments['-i'])
        xopt, fopt = pyswarm.pso(
            optimizer, lb, ub, swarmsize=100, maxiter=20, debug=True,
            phip=0.8, phig=0.8, omega=0.8, processes=processes)
    else:
        xopt, fopt = pyswarm.pso(
            optimizer, lb, ub, swarmsize=100, maxiter=20, debug=True,
            phip=0.8, phig=0.8, omega=0.8)
    print(xopt)
    print(fopt)
