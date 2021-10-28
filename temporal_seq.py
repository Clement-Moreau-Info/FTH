from typing import List, TypeVar

T = TypeVar('T')


class TemporalSeq:
    def __init__(self, acts: List[T], times: List[float]) -> None:
        self.acts = acts
        self.times = times
