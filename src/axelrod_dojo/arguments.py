from docopt import docopt
from .utils import prepare_objective
from .algorithms.genetic_algorithm import Population
from .algorithms.particle_swarm_optimization import PSO


def parse_arguments(doc, version=None):
    arguments = docopt(doc, version=version)
    try:
        algorithm = arguments["--algorithm"].lower()
    except KeyError:
        algorithm = "ea"
    algorithm_arguments = {
        "processes": int(arguments['--processes']),
        # Vars for the genetic algorithm
        "population": int(arguments['--population']),
        "generations": int(arguments['--generations']),
        "bottleneck": int(arguments['--bottleneck']),
        "mutation_probability": float(arguments['--mu']),
        "output_filename": arguments['--output'],
        # Objective
        "name": str(arguments['--objective']),
        "repetitions": int(arguments['--repetitions']),
        "turns": int(arguments['--turns']),
        "noise": float(arguments['--noise']),
        "nmoran": int(arguments['--nmoran'])
    }

    objective = prepare_objective(
        algorithm_arguments["name"],
        algorithm_arguments["turns"],
        algorithm_arguments["noise"],
        algorithm_arguments["repetitions"],
        algorithm_arguments["nmoran"]
    )

    return arguments, algorithm, algorithm_arguments, objective


def invoke(doc, version, player_class, player_kwargs_func):
    arguments, algorithm, algorithm_arguments, objective = parse_arguments(doc, version)
    player_kwargs = player_kwargs_func(arguments)
    print(arguments)

    # Evolutionary Algorithm
    if algorithm == "ea":
        population = Population(
            player_class,
            player_kwargs,
            algorithm_arguments["population"],
            objective,
            algorithm_arguments["output_filename"],
            algorithm_arguments["bottleneck"],
            algorithm_arguments["mutation_probability"],
            processes=algorithm_arguments["processes"])

        population.run(algorithm_arguments["generations"])

        # Get the best member of the population to output.
        scores = population.score_all()
        record, record_holder = 0, -1
        for i, s in enumerate(scores):
            if s >= record:
                record = s
                record_holder = i
        xopt, fopt = population.population[record_holder], record

    # Particle Swarm Algorithm
    elif algorithm == "ps":
        pso = PSO(player_class,
                  player_kwargs,
                  objective=objective,
                  population=algorithm_arguments["population"],
                  generations=algorithm_arguments["generations"]
                  )
        xopt_helper, fopt = pso.swarm()
        xopt = player_class(**player_kwargs)
        # xopt.read_vector(xopt_helper, num_states)

    else:
        print("Algorithm must be one of EA or PS.")
        exit()

    # Print best performer.
    print("Best Score: {} {}".format(fopt, xopt))
