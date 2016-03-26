## Installation

Install axelrod:

```
pip install axelrod numpy cloudpickle
```

Clone this repository

## Some Changes

In the original repository the strategies were run against all the default strategies in the Axelrod library. This is slow and probably not necessary. For example the Meta* players are just combinations of the other players, and very computationally intensive; it's probably ok to remove those.

This fork uses a subset of about 90 strategies, excluding the most computationally intensives (e.g. the hunters).

## The strategies

The LookerUp strategies are based on lookup tables with two parameters:
* n, the number of rounds of trailing history to use and
* m, the number of rounds of initial opponent play to use

### Open questions

* What's the best table for n, m?
* What's the best table against parameterized strategies. For example, if the opponents are `[RandomPlayer(x) for x in np.arange(0, 1, 0.01)], what lookup table is best? Is it much different from the generic table?
* Can we separate n into n1 and n2 where different amounts of history are used for the player and the opponent?
* Incorporate @GKDO's swarm model that makes the tables non-deterministic, for the same values of n and m. Does this produce better results for all n and m?


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
```
### 4, 4
```
python lookup_evolve.py -p 4 -s 4  -g 100000 -k 20 -u 0.01 -b 20 -i 4 -o evolve4-4.csv
```
## Analyzing

The output files `evolve{n}-{m}.csv` can be easily sorted by `analyze_data.py`, which will output the best performing tables. These can be added back into Axelrod.