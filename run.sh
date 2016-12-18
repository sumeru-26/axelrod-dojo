# 2**n * 2 * 2**m == 2**{n + m + 1}

# mutation: 2**{-n -m}

python lookup_evolve.py -p 1 -s 1  -g 100000 -k 10 -u 0.1 -b 10 -i 4 -o evolve1-1.csv

python lookup_evolve.py -p 1 -s 2  -g 100000 -k 20 -u 0.1 -b 20 -i 2 -o evolve1-2.csv

python lookup_evolve.py -p 2 -s 1  -g 100000 -k 20 -u 0.1 -b 20 -i 2 -o evolve2-1.csv



python lookup_evolve.py -p 1 -s 1  -g 100000 -k 10 -u 0.1 -b 10 -i 4 -o evolve1-1.csv

python lookup_evolve.py -p 1 -s 1  -g 100000 -k 10 -u 0.1 -b 10 -i 4 -o evolve1-1.csv




python lookup_evolve.py -p 3 -s 3  -g 100000 -k 20 -u 0.001 -b 20 -i 4 -o evolve3-3.csv
