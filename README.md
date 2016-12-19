# Axelrod Evolvers

This repository contains training code for the strategies LookerUp, PSOGambler, and EvolvedANN (feed-forward neural network).
There are three scripts, one for each strategy:
* looker_evolve.py
* pso_evolve.py
* ann_evolve.py

In the original iteration the strategies were run against all the default strategies in the Axelrod library. This is slow and probably not necessary. For example the Meta players are just combinations of the other players, and very computationally intensive; it's probably ok to remove those.

## The Strategies

The LookerUp strategies are based on lookup tables with two parameters:
* n, the number of rounds of trailing history to use and
* m, the number of rounds of initial opponent play to use

PSOGambler is a stochastic version of LookerUp, trained with a particle swarm algorithm.

EvolvedANN is one hidden layer feed forward neural network based algorithm.

All three strategies are trained with an evolutionary algorithm and are examples of reinforcement learning.

### Open questions

* What's the best table for n, m for LookerUp and PSOGambler?
* What's the best table against parameterized strategies? For example, if the opponents are `[RandomPlayer(x) for x in np.arange(0, 1, 0.01)], what lookup table is best? Is it much different from the generic table?
* Can we separate n into n1 and n2 where different amounts of history are used for the player and the opponent?
* Are there other features that would improve the performance of EvolvedANN?


## Running

`python lookup-evolve.py -h`

will display help. There are a number of options and you'll want to set the mutation rate appropriately. The number of keys defining the strategy is `2**{n + m + 1}` so you want a mutation rate in the neighborhood of `2**(-n-m)` so that there's enough variation introduced.


Here are some recommended defaults:
```
python lookup_evolve.py -p 3 -s 3  -g 100000 -k 20 -u 0.01 -b 20 -i 4 -o evolve3-3.csv

python lookup_evolve.py -p 3 -s 2  -g 100000 -k 20 -u 0.03 -b 20 -i 4 -o evolve3-2.csv

python lookup_evolve.py -p 3 -s 1  -g 100000 -k 20 -u 0.06 -b 20 -i 4 -o evolve3-1.csv

python lookup_evolve.py -p 1 -s 3  -g 100000 -k 20 -u 0.03 -b 20 -i 4 -o evolve1-3.csv

python lookup_evolve.py -p 2 -s 3  -g 100000 -k 20 -u 0.03 -b 20 -i 4 -o evolve2-3.csv
```
### 2, 2 is the current winner:
```
python lookup_evolve.py -p 2 -s 2  -g 100000 -k 20 -u 0.06 -b 20 -i 4 -o evolve2-2.csv

python lookup_evolve.py -p 1 -s 2  -g 100000 -k 20 -u 0.1 -b 20 -i 2 -o evolve1-2.csv

python lookup_evolve.py -p 1 -s 2  -g 100000 -k 20 -u 0.1 -b 20 -i 2 -o evolve2-1.csv

```
### 4, 4 (might take for ever / need a ton of ram)
```
python lookup_evolve.py -p 4 -s 4  -g 100000 -k 20 -u 0.002 -b 20 -i 4 -o evolve4-4.csv
```
## Analyzing

The output files `evolve{n}-{m}.csv` can be easily sorted by `analyze_data.py`, which will output the best performing tables. These can be added back into Axelrod.
