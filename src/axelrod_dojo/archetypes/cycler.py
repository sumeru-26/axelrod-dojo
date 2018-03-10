import axelrod as axl
from numpy import random

from axelrod_dojo.utils import Params

C, D = axl.Action


class CyclerParams(Params):
    """
    Cycler params is a class to aid with the processes of calculating the best sequence of moves for any given set of
    opponents. Each of the population in our algorithm will be an instance of this class for putting into our
    genetic lifecycle.
    """

    def __init__(self, sequence=None, sequence_length: int = 200, mutation_probability=0.1, mutation_potency=1):
        if sequence is None:
            # generates random sequence if none is given
            self.sequence = self.generate_random_sequence(sequence_length)
        else:
            #  when passing a sequence, make a copy of the sequence to ensure mutation is for the instance only.
            self.sequence = list(sequence)

        self.sequence_length = len(self.sequence)
        self.mutation_probability = mutation_probability
        self.mutation_potency = mutation_potency

    def __repr__(self):
        return "{}".format(self.sequence)

    @staticmethod
    def generate_random_sequence(sequence_length):
        """
        Generate a sequence of random moves

        Parameters
        ----------
        sequence_length - length of random moves to generate

        Returns
        -------
        list - a list of C & D actions: list[Action]
        """
        # D = axl.Action(0) | C = axl.Action(1)
        return list(map(axl.Action, random.randint(0, 2, (sequence_length, 1))))

    def crossover(self, other_cycler, in_seed=0):
        """
        creates and returns a new CyclerParams instance with a single crossover point in the middle

        Parameters
        ----------
        other_cycler - the other cycler where we get the other half of the sequence

        Returns
        -------
        CyclerParams

        """
        seq1 = self.sequence
        seq2 = other_cycler.sequence

        if not in_seed == 0:
            # only seed for when we explicitly give it a seed
            random.seed(in_seed)

        midpoint = int(random.randint(0, len(seq1)) / 2)
        new_seq = seq1[:midpoint] + seq2[midpoint:]
        return CyclerParams(sequence=new_seq)

    def mutate(self):
        """
        Basic mutation which may change any random gene(s) in the sequence.
        """
        if random.rand() <= self.mutation_probability:
            mutated_sequence = self.sequence
            for _ in range(self.mutation_potency):
                index_to_change = random.randint(0, len(mutated_sequence))
                mutated_sequence[index_to_change] = mutated_sequence[index_to_change].flip()
            self.sequence = mutated_sequence

    def player(self):
        """
        Create and return a Cycler player with the sequence that has been generated with this run.

        Returns
        -------
        Cycler(sequence)
        """
        return axl.Cycler(self.get_sequence_str())

    def copy(self):
        """
        Returns a copy of the current cyclerParams

        Returns
        -------
        CyclerParams - a separate instance copy of itself.
        """
        # seq length will be provided when copying, no need to pass
        return CyclerParams(sequence=self.sequence, mutation_probability=self.mutation_probability)

    def get_sequence_str(self):
        """
        Concatenate all the actions as a string for constructing Cycler players

        [C,D,D,C,D,C] -> "CDDCDC"
        [C,C,D,C,C,C] -> "CCDCCC"
        [D,D,D,D,D,D] -> "DDDDDD"

        Returns
        -------
        str
        """
        string_sequence = ""
        for action in self.sequence:
            string_sequence += str(action)

        return string_sequence
