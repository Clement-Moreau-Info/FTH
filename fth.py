import skfuzzy as fuzz
from edit import *
import numpy as np
from temporal_seq import *
from scipy import signal
import networkx as nx
from functools import lru_cache
from numba import prange
from typing import Callable, TypeVar

######################
## GLOBAL VARIABLES ##
######################
interval_step = 5


######################


##
# Fuzzy context function 
# e     : Edit operation
# beta  : Boundary in min
##
def fuzzy_context(e: Edit, beta=480) -> List[float]:
    return fuzz.trapmf(np.arange(0, np.sum(e.S_i.times), interval_step),
                     [e.t_edit - beta, e.t_edit, e.t_edit + e.delta, e.t_edit + e.delta + beta])

##
# Similarity over the sequence at time t
# e     : Edit operation
# t     : Time t
# sim   : Similarity function between symbols
##
def sim_e(e: Edit, t: float, sim: Callable[[str, str], float]) -> float:
    for i in prange(len(e.S_i.times)):
        if np.sum(e.S_i.times[:i]) <= t < np.sum(e.S_i.times[:i + 1]):
            return sim(e.S_i.acts[i], e.x)
    return 0

##
# Gamma cost function
# e     : Edit operation
# sim   : Similarity function between symbols
# beta  : Boundary in min
##
def cost_gamma(e: Edit, sim: Callable[[str, str], float], beta: float) -> float:
    I = np.arange(0, np.sum(e.S_i.times), interval_step)                           # Time interval
    mu = fuzz.trapmf(I, [e.t_edit - beta, e.t_edit, e.t_edit + e.delta,            # Fuzzy function
                       e.t_edit + e.delta + beta])
    sim_fun = [sim_e(e, t, sim) for t in I]                                        # Similarity over the sequence
    sim_context = [mu[i] * sim_fun[i] for i in range(len(I))]                      # Merge of fuzzy function and similarity
    tab_gate = np.arange(0, e.delta, interval_step)                                # Interval of gate function 
    gate = fuzz.trapmf(tab_gate, [0, 0, e.delta, e.delta])                         # Encoding as fuzzy function 
    convo = signal.convolve(sim_context, gate, mode='same') / (e.delta / interval_step)  # Convolution product of gate and merged fuzzy-sim function 
    return 1 - np.max(convo)

##
# Delta cost function
# e     : Edit operation
# sim   : Similarity function between symbols
# beta  : Boundary in min
##
def cost_delta(e: Edit, sim: Callable[[str, str], float], beta: float) -> float:
    return e.delta * cost_gamma(e, sim, beta)

##
# One sided FTH
# S1    : Semantic-temporal seq 1
# S2    : Semantic-temporal seq 1
# sim   : Similarity function between symbols
# beta  : Boundary in min
# f     : Cost function (choose `cost_delta` or `cost_gamma`
##
def one_sided_fth(S1: Temporal_seq, S2: Temporal_seq, sim: Callable[[str, str], float], beta: float, f=cost_delta) -> float:
    sum_cost = 0
    for i in prange(len(S1.acts)):
        e = Edit(S1.acts[i], S1.times[i], np.sum(S1.times[:i]), S2)
        sum_cost += f(e, sim, beta)
    return sum_cost

##
# FTH
# S1    : Semantic-temporal seq 1
# S2    : Semantic-temporal seq 1
# sim   : Similarity function between symbols
# beta  : Boundary in min
# f     : Cost function (choose `cost_delta` or `cost_gamma`
##
def fth(S1: Temporal_seq, S2: Temporal_seq, sim: Callable[[str, str], float], beta: float, f=cost_delta) -> float:
    return max(one_sided_fth(S1, S2, sim, beta, f),
               one_sided_fth(S2, S1, sim, beta, f))
