
class Params(object):
    """Abstract Base Class for Parameters Objects."""

    def mutate(self):
        pass

    def random(self):
        pass

    def __repr__(self):
        pass

    def from_repr(self):
        pass

    def copy(self):
        pass

    def player(self):
        pass

    def params(self):
        pass

    def crossover(self, other):
        pass

    def receive_vector(self, vector):
        """Receives a vector and creates an instance attribute called
        vector."""
        pass

    def vector_to_instance(self):
        """Turns the attribute vector in to an axelrod player instance."""
        pass

    def create_vector_bounds(self):
        """Creates the bounds for the decision variables."""
        pass
