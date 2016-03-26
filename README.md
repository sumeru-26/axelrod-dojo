## Installation

Install axelrod:

```
pip install axelrod numpy cloudpickle
```

Clone this repository

## Running

`python lookup-evolve.py -h`

will display help. There are a number of options and you'll want to set the mutation rate appropriately. The number of keys defining the strategy is `2**{n + m + 1}` so you want a mutation rate in the neighborhood of `2**(-n-m)` so that there's enough variation introduced.


Here are some recommended defaults:

python lookup_evolve.py -p 3 -s 3  -g 100000 -k 20 -u 0.01 -b 20 -i 4 -o evolve3-3.csv

python lookup_evolve.py -p 3 -s 2  -g 100000 -k 20 -u 0.03 -b 20 -i 4 -o evolve3-2.csv

python lookup_evolve.py -p 3 -s 1  -g 100000 -k 20 -u 0.06 -b 20 -i 4 -o evolve3-1.csv

python lookup_evolve.py -p 1 -s 3  -g 100000 -k 20 -u 0.03 -b 20 -i 4 -o evolve1-3.csv

python lookup_evolve.py -p 2 -s 3  -g 100000 -k 20 -u 0.03 -b 20 -i 4 -o evolve2-3.csv

### 2, 2 is the current winner:

python lookup_evolve.py -p 2 -s 2  -g 100000 -k 20 -u 0.06 -b 20 -i 4 -o evolve2-2.csv

### 4, 4

python lookup_evolve.py -p 4 -s 4  -g 100000 -k 20 -u 0.01 -b 20 -i 4 -o evolve4-4.csv