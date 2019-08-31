from axelrod import Action

C, D = Action.C, Action.D


    @staticmethod
    def repr_rows(rows):
        ss = []
        for row in rows:
            ss.append("_".join(list(map(str, row))))
        return "|".join(ss)

    def __repr__(self):
        return "{}:{}:{}:{}:{}".format(
            self.initial_state,
            self.initial_action,
            self.repr_rows(self.transitions_C),
            self.repr_rows(self.transitions_D),
            self.repr_rows([self.emission_probabilities])
        )

    @classmethod
    def parse_repr(cls, s):
        def parse_vector(line):
            row = line.split('_')
            row = list(map(float, row))
            return row
        def parse_matrix(sm):
            rows = []
            lines = sm.split('|')
            for line in lines:
                rows.append(parse_vector(line))
            return rows
        lines = s.split(':')
        initial_state = int(lines[0])
        initial_action = Action.from_char(lines[1])
        t_C = parse_matrix(lines[2])
        t_D = parse_matrix(lines[3])
        p = parse_vector(lines[4])
        num_states = len(t_C)
        return cls(num_states=num_states,
                   transitions_C=t_C,
                   transitions_D=t_D,
                   emission_probabilities=p,
                   initial_state=initial_state,
                   initial_action=initial_action)


