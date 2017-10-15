Use different objective functions
=================================

It is currently possible to optimise players for 3 different objectives:

- Score;
- Score difference;
- Probability of fixation in a Moran process.

This is done by passing a different objective :code:`name` to the
:code:`prepare_objective` function::

    >>> import axelrod_dojo as dojo
    >>> score_objective = dojo.prepare_objective(name="score", turns=10, repetitions=1)
    >>> diff_objective = dojo.prepare_objective(name="score_diff", turns=10, repetitions=1)
    >>> moran_objective = dojo.prepare_objective(name="moran", turns=10, repetitions=1)
