from temporal_seq import *


class Edit:
    def __init__(self, x: T, delta: float, t_edit: float, seq_i: TemporalSeq):
        """
        :param x:       Edited symbol
        :param delta:   Duration of x
        :param t_edit:  Time of edition in the sequence seq_i
        :param seq_i:   Edited sequence
        """
        self.x = x
        self.delta = delta
        self.t_edit = t_edit
        self.seq_i = seq_i
