
from axelrod import Action, FSMPlayer
from axelrod.action import UnknownActionError

from axelrod_dojo.utils import Params

C, D = Action.C, Action.D


def copy_lists(rows):
    new_rows = list(map(list, rows))
    return new_rows



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
