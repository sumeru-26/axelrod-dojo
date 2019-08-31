from axelrod import Action, Gambler


C, D = Action.C, Action.D


    def __repr__(self):
        return "{}:{}:{}:{}".format(
            self.plays,
            self.op_plays,
            self.op_start_plays,
            '|'.join(str(v) for v in self.pattern)
        )
