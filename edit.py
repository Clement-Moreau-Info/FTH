from temporal_seq import *
from typing import TypeVar

T = TypeVar('T')


class Edit:
    def __init__(self, x: T, delta: float, t_edit: float, seq_i: Temporal_seq):
        self.x = x
        self.delta = delta
        self.t_edit = t_edit
        self.seq_i = seq_i
