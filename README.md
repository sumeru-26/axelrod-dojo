# Axelrod Evolvers

This repository contains reinforcement learning training code for the following
strategy types:
* Lookup tables (LookerUp)
* Particle Swarm algorithms (PSOGambler)
* Feed Forward Neural Network (EvolvedANN)
* Finite State Machine (FSMPlayer)

The training is done by evolutionary algorithms or particle swarm algorithms. There
is another repository that trains Neural Networks with gradient descent. In this
repository there are scripts for each strategy type:

* [looker_evolve.py](looker_evolve.py)
* [pso_evolve.py](pso_evolve.py)
* [ann_evolve.py](ann_evolve.py)
* [fsm_evolve.py](fsm_evolve.py)

In the original iteration the strategies were run against all the default
strategies in the Axelrod library. This is slow and probably not necessary. For
example the Meta players are just combinations of the other players, and very
computationally intensive; it's probably ok to remove those. So by default the
training strategies are the `short_run_time_strategies` from the Axelrod library.

## The Strategies

The LookerUp strategies are based on lookup tables with three parameters:
* n1, the number of rounds of trailing history to use and
* n2, the number of rounds of trailing opponent history to use
* m, the number of rounds of initial opponent play to use

PSOGambler is a stochastic version of LookerUp, trained with a particle swarm
algorithm. The resulting strategies are generalizations of memory-N strategies.

EvolvedANN is one hidden layer feed forward neural network based algorithm.
Various features are derived from the history of play. The number of nodes in
the hidden layer can be changed.

EvolvedFSM searches over finite state machines with a given number of states.

Note that large values of the parameters will make the strategies prone to
overfitting.

## Optimization Functions

There are three objective functions:
* Maximize mean match score over all opponents with `objective_match_score`
* Maximize mean match score difference over all opponents with `objective_match_score_difference`
* Maximize Moran process fixation probability with `objective_match_moran_win`

## Running

### Look up Tables

```bash
$ python lookup_evolve.py -h
Lookup Evolve.

Usage:
    lookup_evolve.py [-h] [-p PLAYS] [-o OPP_PLAYS] [-s STARTING_PLAYS]
    [-g GENERATIONS] [-k STARTING_POPULATION] [-u MUTATION_RATE] [-b BOTTLENECK]
    [-i PROCESSORS] [-f OUTPUT_FILE] [-z INITIAL_POPULATION_FILE] [-n NOISE]

Options:
    -h --help                   show this
    -p PLAYS                    number of recent plays in the lookup table [default: 2]
    -o OPP_PLAYS                number of recent plays in the lookup table [default: 2]
    -s STARTING_PLAYS           number of opponent starting plays in the lookup table [default: 2]
    -g GENERATIONS              how many generations to run the program for [default: 500]
    -k STARTING_POPULATION      starting population size for the simulation [default: 20]
    -u MUTATION_RATE            mutation rate i.e. probability that a given value will flip [default: 0.1]
    -b BOTTLENECK               number of individuals to keep from each generation [default: 10]
    -i PROCESSORS               number of processors to use [default: 1]
    -f OUTPUT_FILE              file to write data to [default: tables.csv]
    -z INITIAL_POPULATION_FILE  file to read an initial population from [default: None]
    -n NOISE                    match noise [default: 0.00]
```

There are a number of options and you'll want to set the
mutation rate appropriately. The number of keys defining the strategy is
`2**{n + m + 1}` so you want a mutation rate in the neighborhood of `2**(-n-m)`
so that there's enough variation introduced.

### Particle Swarm

```bash
$ python pso_evolve.py -h
Particle Swarm strategy training code.

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
```

Note that to use the multiprocessor version you'll need to install pyswarm 0.70
directly (pip installs 0.60 which lacks mutiprocessing support).

### Neural Network

```bash
$ python ann_evolve.py -h
Training ANN strategies with an evolutionary algorithm.

Usage:
    ann_evolve.py [-h] [-g GENERATIONS] [-u MUTATION_RATE] [-b BOTTLENECK]
    [-d MUTATION_DISTANCE] [-i PROCESSORS] [-o OUTPUT_FILE]
    [-k STARTING_POPULATION] [-n NOISE]

Options:
    -h --help                    show this
    -g GENERATIONS               how many generations to run the program for [default: 10000]
    -u MUTATION_RATE             mutation rate i.e. probability that a given value will flip [default: 0.4]
    -d MUTATION_DISTANCE         amount of change a mutation will cause [default: 10]
    -b BOTTLENECK                number of individuals to keep from each generation [default: 6]
    -i PROCESSORS                number of processors to use [default: 4]
    -o OUTPUT_FILE               file to write statistics to [default: weights.csv]
    -k STARTING_POPULATION       starting population size for the simulation [default: 5]
    -n NOISE                     match noise [default: 0.0]
```

### Finite State Machines

```bash
$ python fsm_evolve.py -h
FSM Evolve.

Usage:
    fsm_evolve.py [-h] [-s NUM_STATES] [-g GENERATIONS]
    [-k STARTING_POPULATION] [-u MUTATION_RATE] [-b BOTTLENECK]
    [-i PROCESSORS] [-f OUTPUT_FILE] [-n NOISE]

Options:
    -h --help                   show this
    -s NUM_STATES               number FSM states [default: 16]
    -g GENERATIONS              how many generations to run the program for [default: 500]
    -k STARTING_POPULATION      starting population size for the simulation [default: 20]
    -u MUTATION_RATE            mutation rate i.e. probability that a given value will flip [default: 0.1]
    -b BOTTLENECK               number of individuals to keep from each generation [default: 10]
    -i PROCESSORS               number of processors to use [default: 1]
    -f OUTPUT_FILE              file to write data to [default: fsm_tables.csv]
    -n NOISE                    match noise [default: 0.00]
```

## Open questions

* What's the best table for n1, n2, m for LookerUp and PSOGambler? What's the
smallest value of the parameters that gives good results?
* Similarly what's the optimal number of states for a finite state machine
strategy?
* What's the best table against parameterized strategies? For example, if the
opponents are `[RandomPlayer(x) for x in np.arange(0, 1, 0.01)], what lookup
table is best? Is it much different from the generic table?
* Are there other features that would improve the performance of EvolvedANN?
