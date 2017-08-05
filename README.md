[![Coverage
Status](https://coveralls.io/repos/github/Axelrod-Python/axelrod-dojo/badge.svg?branch=packaging)](https://coveralls.io/github/Axelrod-Python/axelrod-dojo?branch=packaging)

[![Build
Status](https://travis-ci.org/Axelrod-Python/axelrod-dojo.svg?branch=master)](https://travis-ci.org/Axelrod-Python/axelrod-dojo)

# Axelrod Evolvers


This repository contains reinforcement learning training code for the following
strategy types:

* Lookup tables (LookerUp)
* Particle Swarm algorithms (PSOGambler), a stochastic version of LookerUp
* Feed Forward Neural Network (EvolvedANN)
* Finite State Machine (FSMPlayer)
* Hidden Markov Models (HMMPLayer), essentially a stochastic version of a finite state machine

Model training is by [evolutionary algorithms](https://en.wikipedia.org/wiki/Evolutionary_algorithm)
 or [particle swarm algorithms](https://en.wikipedia.org/wiki/Particle_swarm_optimization).
There is another repository in the [Axelrod project](https://github.com/Axelrod-Python/Axelrod)
that trains Neural Networks with gradient descent (using tensorflow)
that will likely be incorporated here.

You can use this as a library.

## Development

Install a development version of the library:

    $ python setup.py develop

Run the tests:

    $ python -m unittest discover tests

## Scripts

In this repository there are scripts
for each strategy type with a similar interface:

* `looker_evolve.py`
* `pso_evolve.py`
* `ann_evolve.py`
* `fsm_evolve.py`
* `hmm_evolve.py`


See below for usage instructions.

In the original iteration the strategies were run against all the default
strategies in the Axelrod library. This is slow and probably not necessary.
By default the training strategies are the `short_run_time_strategies` from the
Axelrod library. You may specify any other set of strategies for training.

Basic testing is done by running the trained model against the full set of
strategies in various tournaments. Depending on the optimization function
testing methods will vary.

## The Strategies

**LookerUp** is based on lookup tables with three parameters:
* n1, the number of rounds of trailing history to use
* n2, the number of rounds of trailing opponent history to use
* m, the number of rounds of initial opponent play to use

These are generalizations of deterministic memory-N strategies.

**PSOGambler** is a stochastic version of LookerUp, trained with a particle
swarm algorithm. The resulting strategies are generalizations of memory-N
strategies.

**EvolvedANN** is based on a [feed forward neural network](https://en.wikipedia.org/wiki/Feedforward_neural_network)
with a single hidden layer. Various features are derived from the history of play.
The number of nodes in the hidden layer can also be changed.

**EvolvedFSM** searches over [finite state machines](https://en.wikipedia.org/wiki/Finite-state_machine)
with a given number of states.

**EvolvedHMM** implements a simple [hidden markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model)
based strategy, a stochastic finite state machine.

Note that large values of some parameters will make the strategies prone to
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
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]
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
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --plays PLAYS               Number of recent plays in the lookup table [default: 2]
    --op_plays OP_PLAYS         Number of recent plays in the lookup table [default: 2]
    --op_start_plays OP_START_PLAYS   Number of opponent starting plays in the lookup table [default: 2]
```

### Particle Swarm

```bash
$ python pso_evolve.py -h
Particle Swarm strategy training code.

Original version by Georgios Koutsovoulos @GDKO :
  https://gist.github.com/GDKO/60c3d0fd423598f3c4e4
Based on Martin Jones @mojones original LookerUp code

Usage:
    pso_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--processes PROCESSORS] [--output OUTPUT_FILE] [--objective OBJECTIVE]
    [--repetitions REPETITIONS] [--turns TURNS] [--noise NOISE]
    [--nmoran NMORAN]
    [--plays PLAYS] [--op_plays OP_PLAYS] [--op_start_plays OP_START_PLAYS]

Options:
    -h --help                   Show help
    --population POPULATION     Starting population size  [default: 10]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: pso_tables.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
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
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]
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
    --turns TURNS               Turns in each match [default: 200]
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
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]
    [--states NUM_STATES]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Population size  [default: 40]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 10]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: fsm_tables.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --states NUM_STATES         Number of FSM states [default: 8]
```


### Hidden Markov Model

```bash
$ python hmm_evolve.py -h
Hidden Markov Model Evolver

Usage:
    fsm_evolve.py [-h] [--generations GENERATIONS] [--population POPULATION]
    [--mu MUTATION_RATE] [--bottleneck BOTTLENECK] [--processes PROCESSORS]
    [--output OUTPUT_FILE] [--objective OBJECTIVE] [--repetitions REPETITIONS]
    [--turns TURNS] [--noise NOISE] [--nmoran NMORAN]
    [--states NUM_STATES]

Options:
    -h --help                   Show help
    --generations GENERATIONS   Generations to run the EA [default: 500]
    --population POPULATION     Population size  [default: 40]
    --mu MUTATION_RATE          Mutation rate [default: 0.1]
    --bottleneck BOTTLENECK     Number of individuals to keep from each generation [default: 10]
    --processes PROCESSES       Number of processes to use [default: 1]
    --output OUTPUT_FILE        File to write data to [default: fsm_tables.csv]
    --objective OBJECTIVE       Objective function [default: score]
    --repetitions REPETITIONS   Repetitions in objective [default: 100]
    --turns TURNS               Turns in each match [default: 200]
    --noise NOISE               Match noise [default: 0.00]
    --nmoran NMORAN             Moran Population Size, if Moran objective [default: 4]
    --states NUM_STATES         Number of FSM states [default: 5]
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

