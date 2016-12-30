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
* Maximize mean match score over all opponents with `objective_score`
* Maximize mean match score difference over all opponents with `objective_score_difference`
* Maximize Moran process fixation probability with `objective_moran_win`

Parameters for the objective functions can be specified in the command line
arguments for each evolver.

## Running

### Look up Tables

```bash
$ python lookup_evolve.py -h
Lookup Table Evolver

Usage:
    lookup_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--noise NOISE] [--nmoran NMORAN]
    [--plays PLAYS] [--op_plays OP_PLAYS] [--op_start_plays OP_START_PLAYS]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Starting population size  [default: 10]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 5]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: lookup_tables.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --plays PLAYS               Number of recent plays in the lookup table [default: 2]
    --op_plays OP_PLAYS         Number of recent plays in the lookup table [default: 2]
    --op_start_plays OP_START_PLAYS   Number of opponent starting plays in the lookup table [default: 2]
```

There are a number of options and you'll want to set the
mutation rate appropriately. The number of keys defining the strategy is
`2**{n + m + 1}` so you want a mutation rate in the neighborhood of `2**(-n-m)`
so that there's enough variation introduced.

### Particle Swarm

```bash
$ python pso_evolve.py -h
Particle Swarm strategy training code.

Original version by Georgios Koutsovoulos @GDKO :
  https://gist.github.com/GDKO/60c3d0fd423598f3c4e4
Based on Martin Jones @mojones original LookerUp code

Usage:
    pso_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--noise NOISE] [--nmoran NMORAN]
    [--plays PLAYS] [--op_plays OP_PLAYS] [--op_start_plays OP_START_PLAYS]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Starting population size  [default: 10]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 5]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: pso_tables.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --plays PLAYS               Number of recent plays in the lookup table [default: 2]
    --op_plays OP_PLAYS         Number of recent plays in the lookup table [default: 2]
    --op_start_plays OP_START_PLAYS     Number of opponent starting plays in the lookup table [default: 2]
```

Note that to use the multiprocessor version you'll need to install pyswarm 0.70
directly (pip installs 0.60 which lacks mutiprocessing support).

### Neural Network

```bash
$ python ann_evolve.py -h
ANN evolver.
Trains ANN strategies with an evolutionary algorithm.

Original version by Martin Jones @mojones:
https://gist.github.com/mojones/b809ba565c93feb8d44becc7b93e37c6

Usage:
    ann_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--noise NOISE] [--nmoran NMORAN]
    [--features FEATURES] [--hidden HIDDEN] [--mu_distance DISTANCE]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Starting population size  [default: 10]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 5]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: ann_weights.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --features FEATURES         Number of ANN features [default: 17]
    --hidden HIDDEN             Number of hidden nodes [default: 10]
    --mu_distance DISTANCE      Delta max for weights updates [default: 5]
```

### Finite State Machines

```bash
$ python fsm_evolve.py -h
Finite State Machine Evolver

Usage:
    fsm_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--noise NOISE] [--nmoran NMORAN]
    [--states NUM_STATES]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Starting population size  [default: 10]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 5]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: fsm_tables.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --states NUM_STATES         Number of FSM states [default: 8]
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
