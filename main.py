from fth import *
import networkx as nx
from functools import lru_cache, partial
import pandas as pd

# Path to ontology
path_onto = "ontology_emd.txt"

#-------#


@lru_cache(maxsize=100000)
def wu_palmer(x: str, y: str, path: str, rootnode="All") -> float:
    ontology = nx.read_adjlist(path, create_using=nx.DiGraph)
    return (2.0 * nx.shortest_path_length(ontology, rootnode, nx.lowest_common_ancestor(ontology, x, y))) / (
            nx.shortest_path_length(ontology, rootnode, x) + nx.shortest_path_length(ontology, rootnode, y))


def extract_temp_seq(path: str, sep=";", id="id") -> List[TemporalSeq]:
    df = pd.read_csv(path, sep=sep)
    max_seq = max(df[id]) + 1

    seq_act = [[str(x) for x in df[df[id] == i].iloc[:, 1].values.tolist()]
                for i in range(1, max_seq)]
    seq_temp = [[t for t in df[df[id] == i].iloc[:, 2].values.tolist()]
                for i in range(1, max_seq)]

    T_max = np.sum(seq_temp[0])
    for i in range(max_seq - 1):
        if np.sum(seq_temp[i]) != T_max:
            raise NameError("Seq ", i, " has a different temporal size")

    return [TemporalSeq(seq_act[i], seq_temp[i]) for i in range(max_seq - 1)]


def main():
    seq_alice = TemporalSeq(['1', '133', '100', '11', '100', '51', '131', '12', '1'],
                             [210, 20, 10, 250, 15, 60, 15, 290, 570])
    seq_bob = TemporalSeq(['1', '100', '133', '11', '100', '1'],
                           [230, 10, 30, 480, 60, 630])

    print("FTH(S_alice, S_bob) = ", fth(seq_bob, seq_alice, partial(wu_palmer, path=path_onto), 240, cost_delta))

    T_seq = extract_temp_seq("test_seq.csv")


if __name__ == '__main__':
    main()
