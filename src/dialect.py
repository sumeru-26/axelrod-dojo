from collections import defaultdict

import axelrod as axl
from axelrod_dojo import FSMParams

C, D = axl.Action.C, axl.Action.D


# def cooperates(player, actions):
#
#     pass
#
# def has_handshake(player, repeat=5):
#     for x in itertools.product([C, D], repeat=repeat):
#         print(x)
#
#
#     pass


def convergence(mp, size):
    # Grab the populations first.
    pops = [[], []]
    for cluster in [0, 1]:
        for i in range(size):
            vertex = "{}:{}".format(cluster, i)
            player = mp.players[mp.index[vertex]]
            pops[cluster].append(player)
    # Check that players compete with the others in their population but
    # not with opponents in the other
    for cluster in [0, 1]:
        for i, p1 in enumerate(pops[cluster]):
            for j, p2 in enumerate(pops[cluster]):
                if i == j:
                    continue
                m = axl.Match((p1, p2))
                m.play()
                if m.final_score_per_turn()[0] < 2.3:
                    return False

    for i, p1 in enumerate(pops[0]):
        for j, p2 in enumerate(pops[1]):
            if i == j:
                continue
            m = axl.Match((p1, p2))
            m.play()
            if m.final_score_per_turn()[0] > 2.3:
                return False

    return True


def main(size=4, num_states=16, iterations=10000):
    igraph = axl.graph.attached_complete_graphs(size, directed=False, loops=False)
    # rgraph = axl.graph.attached_complete_graphs(size, directed=False, loops=True)


    # Random players
    # players = [FSMParams(num_states).player() for _ in range(2 * size)]

    # Clones
    player = FSMParams(num_states).player()
    players = [player.clone() for _ in range(2 * size)]

    mp = axl.MoranProcess(players, mutation_rate=0.1,
                          interaction_graph=igraph,
                          # reproduction_graph=rgraph,
                          mutation_method="atomic")

    for i in range(iterations):
        mp.__next__()
        done = convergence(mp, size)
        if done:
            return True
    return False


if __name__ == "__main__":
    d = defaultdict(int)
    for i in range(1000):
        result = main(size=4, num_states=16, iterations=10000)
        d[result] += 1
        print(d[True], d[False], result)
