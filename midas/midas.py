#!python3

import count_min_sketch
import time
from typing import *
from sklearn.metrics import roc_auc_score
from argparse import ArgumentParser


class CMSCounter:
    def __init__(self, nrows: int, ncolumns: int, scale: float = 0):
        self.current = count_min_sketch.CMS(nrows, ncolumns)
        self.total = count_min_sketch.clone_cms(self.current)
        self.scale = scale
        self.timestamp = 0

    def add(self, edge, time):
        h = hash(edge)
        if time > self.timestamp:
            self.roll_time(time)
        nc = self.current.insert(h)
        nt = self.total.insert(h)
        return self.score(nc, nt, time)

    def roll_time(self, time):
        self.current.scale(self.scale)
        self.timestamp = time

    def score(self, a, s, t):
        return 0 if s == 0 or t == 1 else pow((a - s / t) * t, 2) / (s * (t - 1))

    def run(self, edges: List[Tuple[int, int, int]]) -> List[float]:
        scores = []
        for (s, d, t) in edges:
            score = counter.add((s, d), t)
            scores.append(score)
        return scores


class MidasR:
    def __init__(self, nrows: int, ncolumns: int, scale: float = 0.5):
        self.source = CMSCounter(nrows, ncolumns, scale)
        self.dest = CMSCounter(nrows, ncolumns, scale)
        self.combined = CMSCounter(nrows, ncolumns, scale)
        self.timestamp = 0

    def add(self, edge: Tuple[int, int], time: int) -> float:
        if time > self.timestamp:
            self.roll_time(time)

        source = edge[0]
        dest = edge[1]

        s_s = self.source.add(source, time)
        s_d = self.dest.add(dest, time)
        s_c = self.combined.add(edge, time)

        return max(s_s, s_d, s_c)

    def roll_time(self, time):
        self.source.roll_time(time)
        self.dest.roll_time(time)
        self.combined.roll_time(time)
        self.timestamp = time

    def run(self, edges: List[Tuple[int, int, int]]) -> List[float]:
        scores = []
        for (s, d, t) in edges:
            score = counter.add((s, d), t)
            scores.append(score)
        return scores


def parse_input() -> Dict:
    parser = ArgumentParser()
    parser.add_argument("input",
                        help="file to read input from, source,dest,time on newlines", metavar="FILE")
    parser.add_argument("labels",
                        help="file to read labels from, each on a new line", metavar="FILE")
    parser.add_argument("-o", "--output",
                        dest="output",
                        default="",
                        help="file to write output to")
    parser.add_argument("-t", "--type",
                        dest="type",
                        default="R",
                        choices={"R", "r", "F", "f", "N", "n"},
                        help="choice of cores. Choice of R for relational, F for filtering, N for normal")

    parser.add_argument(
        "-s", "--scale", help="Factor to decay current time counter by", type=float, default=0)
    args = vars(parser.parse_args())

    return args


def read_data(input: str, labels: str) -> Tuple[list[Tuple[int, int, int]], list[str]]:
    print("reading")
    edges = []
    with open(args["input"], 'r') as f:
        for line in f.readlines():
            (s, d, t) = [int(item.strip()) for item in line.split(',')]
            edges.append((s, d, t))

    with open(args["labels"], 'r') as f:
        truth = f.readlines()
    return edges, truth


def write_scores(output_file: str, output: list[float]):
    print("writing")
    with open(args["output"], 'w') as fout:
        for score in scores:
            fout.write(f"{score}\n")


if __name__ == "__main__":
    args = parse_input()

    match args["type"].lower():
        case "r":
            counter = MidasR(20, 2048, args["scale"])
        case "f":
            counter = CMSCounter(20, 2048, args["scale"])
        case _:
            counter = CMSCounter(20, 2048)

    edges, truth = read_data(args["input"], args["labels"])

    start = time.time()
    print("scoring")
    scores = counter.run(edges)
    end = time.time()

    print(f"ROC-AUC = {roc_auc_score(truth, scores):.4f}\n in {end-start}s")

    if len(args["output"]) > 1:
        write_scores(args["output"], scores)
