from fth import *
import networkx as nx
from functools import lru_cache

# Path to ontology
path_onto = "ontology_emd.txt"

#-------#


@lru_cache(maxsize=100000)
def wu_palmer(x: str, y: str, path: str, rootnode="All") -> float:
    ontologie = nx.read_adjlist(path, create_using=nx.DiGraph)
    return (2.0 * nx.shortest_path_length(ontologie, rootnode, nx.lowest_common_ancestor(ontologie, x, y))) / (
            nx.shortest_path_length(ontologie, rootnode, x) + nx.shortest_path_length(ontologie, rootnode, y))


# Define the similarity used
def sim(x: str, y: str) -> float:
    return wu_palmer(x, y, path_onto)


if __name__ == '__main__':
    S_alice = Temporal_seq(['1', '133', '100', '11', '100', '51', '131', '12', '1'],
                           [210, 20, 10, 250, 15, 60, 15, 290, 570])

    S_bob = Temporal_seq(['1', '100', '133', '11', '100', '1'],
                         [230, 10, 30, 480, 60, 630])

    print("FTH(S_alice, S_bob) = ", fth(S_bob, S_alice, sim, 240, cost_delta))