from operator import itemgetter
import sys

if __name__ == "__main__":
    data_filename = sys.argv[1]
    results = []
    with open(data_filename) as data:
        for line in data:
            results.append(line.split(','))
    results.sort(reverse=True, key=itemgetter(2))
    for result in results[:5]:
        print(result[2], result[1])
