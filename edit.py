from temporal_seq import *


class Edit:
    def __init__(self, x: str, delta: float, t_edit: float, S_i: Temporal_seq):
        self.x = x
        self.delta = delta
        self.t_edit = t_edit
        self.S_i = S_i