from typing import List

class Temporal_seq:
    def __init__(self, acts: List[str], times: List[float]) -> None:
        self.acts = acts
        self.times = times