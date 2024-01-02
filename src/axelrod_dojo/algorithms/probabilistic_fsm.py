import itertools
from typing import Any, Dict, List, Sequence, Text, Tuple

from axelrod.strategies.finite_state_machines import FSMPlayer, SimpleFSM
from axelrod.action import Action
from axelrod.evolvable_player import (
    EvolvablePlayer,
    InsufficientParametersError,
    copy_lists,
)
from axelrod.player import Player

C, D = Action.C, Action.D
actions = (C, D)
Transition = Tuple[int, Action, int, Action]


class EvolvablePFSMPlayer(FSMPlayer, EvolvablePlayer):
    """
    Abstract base class for evolvable finite state machine players.
    
    Instead of representing what result to return, a probability value is store (0.0-1.0).
    This value is used to determine the probability that C will be chosen.
    """

    name = "EvolvablePFSMPlayer"

    classifier = {
        "memory_depth": 1,
        "stochastic": False,
        "long_run_time": False,
        "inspects_source": False,
        "manipulates_source": False,
        "manipulates_state": False,
    }

    def __init__(
        self,
        transitions: tuple = None,
        initial_state: int = None,
        initial_action: Action = None,
        num_states: int = None,
        mutation_probability: float = 0.1,
        seed: int = None,
    ) -> None:
        """If transitions, initial_state, and initial_action are None
        then generate random parameters using num_states."""
        EvolvablePlayer.__init__(self, seed=seed)
        (
            transitions,
            initial_state,
            initial_action,
            num_states,
        ) = self._normalize_parameters(
            transitions, initial_state, initial_action, num_states
        )
        FSMPlayer.__init__(
            self,
            transitions=transitions,
            initial_state=initial_state,
            initial_action=initial_action,
        )
        self.mutation_probability = mutation_probability
        self.overwrite_init_kwargs(
            transitions=transitions,
            initial_state=initial_state,
            initial_action=initial_action,
            num_states=self.num_states,
        )

    @classmethod
    def normalize_transitions(
        cls, transitions: Sequence[Sequence]
    ) -> Tuple[Tuple[Any, ...], ...]:
        """Translate a list of lists to a tuple of tuples."""
        normalized = []
        for t in transitions:
            normalized.append(tuple(t))
        return tuple(normalized)

    def _normalize_parameters(
        self,
        transitions: Tuple = None,
        initial_state: int = None,
        initial_action: Action = None,
        num_states: int = None,
    ) -> Tuple[Tuple, int, Action, int]:
        if not (
            (transitions is not None)
            and (initial_state is not None)
            and (initial_action is not None)
        ):
            if not num_states:
                raise InsufficientParametersError(
                    "Insufficient Parameters to instantiate EvolvableFSMPlayer"
                )
            transitions, initial_state, initial_action = self.random_params(
                num_states
            )
        transitions = self.normalize_transitions(transitions)
        num_states = len(transitions) // 2
        return transitions, initial_state, initial_action, num_states

    @property
    def num_states(self) -> int:
        return self.fsm.num_states()

    def random_params(
        self, num_states: int
    ) -> Tuple[Tuple[Transition, ...], int, Action]:
        rows = []
        for j in range(num_states):
            for action in actions:
                next_state = self._random.randint(num_states)
                next_action = self._random.choice(actions)
                row = (j, action, next_state, next_action)
                rows.append(row)
        initial_state = self._random.randint(0, num_states)
        initial_action = self._random.choice(actions)
        return tuple(rows), initial_state, initial_action

    def mutate_rows(self, rows: List[List], mutation_probability: float):
        rows = list(rows)
        randoms = self._random.random(len(rows))
        # Flip each value with a probability proportional to the mutation rate
        for i, row in enumerate(rows):
            if randoms[i] < mutation_probability:
                row[3] = row[3].flip()
        # Swap Two Nodes?
        if self._random.random() < 0.5:
            nodes = len(rows) // 2
            n1 = self._random.randint(0, nodes)
            n2 = self._random.randint(0, nodes)
            for j, row in enumerate(rows):
                if row[0] == n1:
                    row[0] = n2
                elif row[0] == n2:
                    row[0] = n1
            rows.sort(key=lambda x: (x[0], 0 if x[1] == C else 1))
        return rows

    def mutate(self):
        initial_action = self.initial_action
        if self._random.random() < self.mutation_probability / 10:
            initial_action = self.initial_action.flip()
        initial_state = self.initial_state
        if self._random.random() < self.mutation_probability / (
            10 * self.num_states
        ):
            initial_state = self._random.randint(0, self.num_states)
        try:
            transitions = self.mutate_rows(
                self.fsm.transitions(), self.mutation_probability
            )
            self.fsm = SimpleFSM(transitions, self.initial_state)
        except ValueError:
            # If the FSM is malformed, try again.
            return self.mutate()
        return self.create_new(
            transitions=transitions,
            initial_state=initial_state,
            initial_action=initial_action,
        )

    def crossover_rows(
        self, rows1: List[List], rows2: List[List]
    ) -> List[List]:
        num_states = len(rows1) // 2
        cross_point = 2 * self._random.randint(0, num_states)
        new_rows = copy_lists(rows1[:cross_point])
        new_rows += copy_lists(rows2[cross_point:])
        return new_rows

    def crossover(self, other):
        if other.__class__ != self.__class__:
            raise TypeError(
                "Crossover must be between the same player classes."
            )
        transitions = self.crossover_rows(
            self.fsm.transitions(), other.fsm.transitions()
        )
        transitions = self.normalize_transitions(transitions)
        return self.create_new(transitions=transitions)

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
        num_states = self.fsm.num_states()
        state_scale = vector[: num_states * 2]
        next_states = [int(s * (num_states - 1)) for s in state_scale]
        actions = vector[num_states * 2 : -1]

        self.initial_action = C if round(vector[-1]) == 0 else D
        self.initial_state = 1

        transitions = []
        for i, (initial_state, action) in enumerate(
            itertools.product(range(num_states), [C, D])
        ):
            next_action = C if round(actions[i]) == 0 else D
            transitions.append(
                [initial_state, action, next_states[i], next_action]
            )
        transitions = self.normalize_transitions(transitions)
        self.fsm = SimpleFSM(transitions, self.initial_state)
        self.overwrite_init_kwargs(
            transitions=transitions,
            initial_state=self.initial_state,
            initial_action=self.initial_action,
        )

    def create_vector_bounds(self):
        """Creates the bounds for the decision variables."""
        size = len(self.fsm.transitions()) * 2 + 1
        lb = [0] * size
        ub = [1] * size
        return lb, ub