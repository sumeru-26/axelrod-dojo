import random
import itertools
from random import randrange, choice

import numpy as np
from axelrod import Action, FSMPlayer
from axelrod.action import UnknownActionError
from axelrod.player import Player

from axelrod_dojo.utils import Params

C, D = Action.C, Action.D


def copy_lists(rows):
    new_rows = list(map(list, rows))
    return new_rows


class FSMParams(Params):

    def __init__(self, num_states, rows=None,
                 initial_state=0, initial_action=C,
                 mutation_probability=0):
        self.PlayerClass = FSMPlayer
        self.num_states = num_states
        self.mutation_probability = mutation_probability
        if rows is None:
            self.randomize()
        else:
            # Make sure to copy the lists
            self.rows = copy_lists(rows)
        self.initial_state = initial_state
        self.initial_action = initial_action

    def player(self):
        player = self.PlayerClass(self.rows, self.initial_state,
                                  self.initial_action)
        return player

    def copy(self):
        return FSMParams(self.num_states, self.rows,
                         self.initial_state, self.initial_action)

    @staticmethod
    def random_params(num_states):
        rows = []
        actions = (C, D)
        for j in range(num_states):
            for action in actions:
                next_state = randrange(num_states)
                next_action = choice(actions)
                row = [j, action, next_state, next_action]
                rows.append(row)
        initial_state = randrange(num_states)
        initial_action = choice([C, D])
        return rows, initial_state, initial_action

    def randomize(self):
        rows, initial_state, initial_action = self.random_params(self.num_states)
        self.rows = rows
        self.initial_state = initial_state
        self.initial_action = initial_action

    @staticmethod
    def mutate_rows(rows, mutation_probability):
        randoms = np.random.random(len(rows))
        # Flip each value with a probability proportional to the mutation rate
        for i, row in enumerate(rows):
            if randoms[i] < mutation_probability:
                row[3] = row[3].flip()
        # Swap Two Nodes?
        if random.random() < 0.5:
            nodes = len(rows) // 2
            n1 = randrange(nodes)
            n2 = randrange(nodes)
            for j, row in enumerate(rows):
                if row[0] == n1:
                    row[0] = n2
                elif row[0] == n2:
                    row[0] = n1
            rows.sort(key=lambda x: (x[0], 0 if x[1]==C else 1))
        return rows

    def mutate(self):
        self.rows = self.mutate_rows(self.rows, self.mutation_probability)
        if random.random() < self.mutation_probability / 10:
            self.initial_action = self.initial_action.flip()
        if random.random() < self.mutation_probability / (10 * self.num_states):
            self.initial_state = randrange(self.num_states)

    @staticmethod
    def crossover_rows(rows1, rows2):
        num_states = len(rows1) // 2
        crosspoint = 2 * randrange(num_states)
        new_rows = copy_lists(rows1[:crosspoint])
        new_rows += copy_lists(rows2[crosspoint:])
        return new_rows

    def crossover(self, other):
        # Assuming that the number of states is the same
        new_rows = self.crossover_rows(self.rows, other.rows)
        return FSMParams(num_states=self.num_states, 
                         mutation_probability=self.mutation_probability, 
                         rows=new_rows,
                         initial_state=self.initial_state, 
                         initial_action=self.initial_action)

    @staticmethod
    def repr_rows(rows):
        ss = []
        for row in rows:
            ss.append("_".join(list(map(str, row))))
        return ":".join(ss)

    def __repr__(self):
        return "{}:{}:{}".format(
            self.initial_state,
            self.initial_action,
            self.repr_rows(self.rows)
        )

    @classmethod
    def parse_repr(cls, s):
        rows = []
        lines = s.split(':')
        initial_state = int(lines[0])
        initial_action = Action.from_char(lines[1])

        for line in lines[2:]:
            row = []

            for element in line.split('_'):
                try:
                    row.append(Action.from_char(element))
                except UnknownActionError:
                    row.append(int(element))

            rows.append(row)
        num_states = len(rows) // 2
        return cls(num_states, rows, initial_state, initial_action)

    def receive_vector(self, vector):
        """
        Read a serialized vector into the set of FSM parameters (less initial
        state).  Then assign those FSM parameters to this class instance.

        The vector has three parts. The first is used to define the next state 
        (for each of the player's states - for each opponents action).

        The second part is the player's next moves (for each state - for 
        each opponent's actions).

        Finally, a probability to determine the player's first move.
        """
        state_scale = vector[:self.num_states * 2]
        next_states = [int(s * (self.num_states - 1)) for s in state_scale]
        actions = vector[self.num_states * 2: -1]
        
        self.initial_action = C if round(vector[-1]) == 0 else D
        self.initial_state = 1

        self.rows = []
        for i, (initial_state, action) in enumerate(
                itertools.product(range(self.num_states), [C, D])):
            next_action = C if round(actions[i]) == 0 else D
            self.rows.append([initial_state, action, next_states[i], next_action])

    def create_vector_bounds(self):
        """Creates the bounds for the decision variables."""
        size = len(self.rows) * 2 + 1

        lb = [0] * size
        ub = [1] * size

        return lb, ub


class SimpleFSM(object):
    """Simple implementation of a finite state machine that transitions
    between states based on the last round of play.

    https://en.wikipedia.org/wiki/Finite-state_machine
    """

    def __init__(self, transitions: tuple, initial_state: int) -> None:
        """
        transitions is a list of the form
        ((state, last_opponent_action, next_state, next_action), ...)

        TitForTat would be represented with the following table:
        ((1, C, 1, C), (1, D, 1, D))
        with initial play C and initial state 1.

        """
        self._state = initial_state
        self._state_transitions = {(current_state, input_action): (next_state, output_action) for
                                   current_state, input_action, next_state, output_action in transitions}  # type: dict

        self._raise_error_for_bad_input()

    def _raise_error_for_bad_input(self):
        callable_states = set(pair[0] for pair in self._state_transitions.values())
        callable_states.add(self._state)
        for state in callable_states:
            self._raise_error_for_bad_state(state)

    def _raise_error_for_bad_state(self, state: int):
        if (state, C) not in self._state_transitions or (state, D) not in self._state_transitions:
            raise ValueError('state: {} does not have values for both C and D'.format(state))

    @property
    def state(self) -> int:
        return self._state

    @state.setter
    def state(self, new_state: int):
        self._raise_error_for_bad_state(new_state)
        self._state = new_state

    @property
    def state_transitions(self) -> dict:
        return self._state_transitions.copy()

    def move(self, opponent_action: Action) -> Action:
        """Computes the response move and changes state."""
        next_state, next_action = self._state_transitions[(self._state, opponent_action)]
        self._state = next_state
        return next_action

    def __eq__(self, other) -> bool:
        """Equality of two FSMs"""
        if not isinstance(other, SimpleFSM):
            return False
        return (self._state, self._state_transitions) == (other.state, other.state_transitions)


class FSMPlayer(Player):
    """Abstract base class for finite state machine players."""

    name = "FSM Player"

    classifier = {
        'memory_depth': 1,
        'stochastic': False,
        'makes_use_of': set(),
        'long_run_time': False,
        'inspects_source': False,
        'manipulates_source': False,
        'manipulates_state': False
    }

    def __init__(self, transitions: tuple = ((1, C, 1, C), (1, D, 1, D)),
                 initial_state: int = 1,
                 initial_action: Action = C) -> None:

        super().__init__()
        self.initial_state = initial_state
        self.initial_action = initial_action
        self.fsm = SimpleFSM(transitions, initial_state)
        self.transitions = transitions

    def mutate(self):
        num_states = len(set(x[1] for x in self.transitions))
        params = FSMParams(
            num_states,
            self.transitions,
            self.initial_state,
            self.initial_action)
        params.mutate()
        return params.player()

    def strategy(self, opponent: Player) -> Action:
        if len(self.history) == 0:
            return self.initial_action
        else:
            action = self.fsm.move(opponent.history[-1])
            return action
