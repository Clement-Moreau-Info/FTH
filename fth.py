import skfuzzy as fuzz
from edit import *
import numpy as np
from temporal_seq import *
from scipy import signal
from numba import prange
from typing import Callable

###
# GLOBAL VARIABLES
###
interval_step = 5

#####


def fuzzy_context(e: Edit, beta=480) -> List[float]:
    """
    :param e:       Edit operation
    :param beta:    Boundary in unit_of_time
    :return:        Fuzzy temporal function
    """
    return fuzz.trapmf(np.arange(0, np.sum(e.seq_i.times), interval_step),
                       [e.t_edit - beta, e.t_edit, e.t_edit + e.delta, e.t_edit + e.delta + beta])


def sim_e(e: Edit, t: float, sim: Callable[[T, T], float]) -> float:
    """
    :param e:     Edit operation
    :param t:     Time t
    :param sim:   Similarity function between symbols
    :return:      Similarity over the sequence at time t
    """
    for i in prange(len(e.seq_i.times)):
        if np.sum(e.seq_i.times[:i]) <= t < np.sum(e.seq_i.times[:i + 1]):
            return sim(e.seq_i.acts[i], e.x)
    return 0


def cost_gamma(e: Edit, sim: Callable[[T, T], float], beta: float) -> float:
    """
    :param e:     Edit operation
    :param sim:   Similarity function between symbols
    :param beta:  Boundary in unit_of_time
    :return:      Cost of the operation e
    """
    I = np.arange(0, np.sum(e.seq_i.times), interval_step)                  # Time interval
    mu = fuzz.trapmf(I, [e.t_edit - beta, e.t_edit, e.t_edit + e.delta,     # Fuzzy function
                     e.t_edit + e.delta + beta])
    sim_fun = [sim_e(e, t, sim) for t in I]                                 # Similarity over the sequence
    sim_context = [mu[i] * sim_fun[i] for i in range(len(I))]               # Merge of fuzzy function and similarity
    tab_gate = np.arange(0, e.delta, interval_step)                         # Interval of gate function
    gate = fuzz.trapmf(tab_gate, [0, 0, e.delta, e.delta])                  # Encoding as fuzzy function 
    convo = signal.convolve(sim_context, gate, mode='same') / (e.delta / interval_step) # Convolution product of gate and merged fuzzy-sim function
    return 1 - np.max(convo)


def cost_delta(e: Edit, sim: Callable[[T, T], float], beta: float) -> float:
    """
    :param e:     Edit operation
    :param sim:   Similarity function between symbols
    :param beta:  Boundary in unit_of_time
    :return:      Cost of the operation e (Hamming version weighted by time)
    """
    return e.delta * cost_gamma(e, sim, beta)


def one_sided_fth(seq1: TemporalSeq, seq2: TemporalSeq, sim: Callable[[T, T], float], beta: float, f=cost_delta) -> float:
    """
    :param seq1:    Semantic-temporal seq 1
    :param seq2:    Semantic-temporal seq 2
    :param sim:     Similarity function between symbols
    :param beta:    Boundary in unit_of_time
    :param f:       Cost function (choose `cost_delta` or `cost_gamma`)
    :return:        Sum of costs to change seq1 to seq2
    """
    sum_cost = 0
    for i in prange(len(seq1.acts)):
        e = Edit(seq1.acts[i], seq1.times[i], np.sum(seq1.times[:i]), seq2)
        sum_cost += f(e, sim, beta)
    return sum_cost


def fth(seq1: TemporalSeq, seq2: TemporalSeq, sim: Callable[[T, T], float], beta: float, f=cost_delta) -> float:
    """
    :param seq1:    Semantic-temporal seq 1
    :param seq2:    Semantic-temporal seq 2
    :param sim:     Similarity function between symbols
    :param beta:    Boundary in unit_of_time
    :param f:       Cost function (choose `cost_delta` or `cost_gamma`)
    :return:        FTH(seq1, seq2)
    """
    return max(one_sided_fth(seq1, seq2, sim, beta, f),
               one_sided_fth(seq2, seq1, sim, beta, f))
