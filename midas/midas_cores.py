import count_min_sketch
from typing import *


class CMSCounter:
    def __init__(self, nrows: int, ncolumns: int, scale: float = 0):
        self.current = count_min_sketch.CMS(nrows, ncolumns)
        self.total = count_min_sketch.clone_cms(self.current)
        self.scale = scale
        self.timestamp = 0

    def add(self, edge: Union[int, Tuple[int, int]], time: int) -> float:
        h = hash(edge)
        if time > self.timestamp:
            self.roll_time(time)
        nc = self.current.insert(h)
        nt = self.total.insert(h)
        return self.score(nc, nt, time)

    def roll_time(self, time: int):
        self.current.scale(self.scale)
        self.timestamp = time

    def score(self, current_count: int, all_time_count: int, t: int) -> float:
        return 0 if all_time_count == 0 or t == 1 else pow((current_count - all_time_count / t) * t, 2) / (all_time_count * (t - 1))

    def run(self, edges: List[Tuple[int, int, int]]) -> List[float]:
        scores = []
        for (s, d, t) in edges:
            score = self.add((s, d), t)
            scores.append(score)
        return scores


class MidasR:
    def __init__(self, nrows: int, ncolumns: int, scale: float = 0.5):
        self.source = CMSCounter(nrows, ncolumns, scale)
        self.dest = CMSCounter(nrows, ncolumns, scale)
        self.combined = CMSCounter(nrows, ncolumns, scale)
        self.timestamp = 0

    def add(self, edge: Tuple[int, int], time: int) -> float:

        return max(self.add_all(edge, time))

    def add_all(self, edge: Tuple[int, int], time: int) -> Tuple[float, float, float]:
        if time > self.timestamp:
            self.roll_time(time)

        source = edge[0]
        dest = edge[1]

        s_s = self.source.add(source, time)
        s_d = self.dest.add(dest, time)
        s_c = self.combined.add(edge, time)
        return (s_s, s_d, s_c)

    def roll_time(self, time):
        self.source.roll_time(time)
        self.dest.roll_time(time)
        self.combined.roll_time(time)
        self.timestamp = time

    def run(self, edges: List[Tuple[int, int, int]]) -> List[float]:
        scores = []
        for (s, d, t) in edges:
            score = self.add((s, d), t)
            scores.append(score)
        return scores

    def run_one(self, edge: Tuple[int, int], time: int) -> Tuple[float, float, float]:

        return self.add_all(edge, time)
