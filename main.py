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


##
# Extraction of all semantic-temporal sequences from a file .csv
# path : path of the file
# sep  : separator
# id   : id sequence colonne
##
def extract_temp_seq(path: str, sep=";", id="id") -> List[Temporal_seq]:
    df = pd.read_csv(path, sep=sep)
    max_seq = max(df[id]) + 1

    # Activity seq
    seq_act = [[str(x) for x in df[df[id] == i].iloc[:, 1].values.tolist()]
                for i in range(1, max_seq)]
    # Temporal seq
    seq_temp = [[t for t in df[df[id] == i].iloc[:, 2].values.tolist()]
                for i in range(1, max_seq)]

    T_max = np.sum(seq_temp[0])
    
    # Verify if for all seq_temp, sum(seq_temp) = T_max
    for i in range(max_seq - 1):
        if np.sum(seq_temp[i]) != T_max:
            raise NameError("Séquences ", i, " de longueur différente")

    return [Temporal_seq(seq_act[i], seq_temp[i]) for i in range(max_seq - 1)]


if __name__ == '__main__':
    S_alice = Temporal_seq(['1', '133', '100', '11', '100', '51', '131', '12', '1'],
                           [210, 20, 10, 250, 15, 60, 15, 290, 570])

    S_bob = Temporal_seq(['1', '100', '133', '11', '100', '1'],
                         [230, 10, 30, 480, 60, 630])

    print("FTH(S_alice, S_bob) = ", fth(S_bob, S_alice, sim, 240, cost_delta))
