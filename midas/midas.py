#!python3

import time
from typing import *
from sklearn.metrics import roc_auc_score
from argparse import ArgumentParser
from midas_cores import CMSCounter, MidasR


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

    edges = []
    with open(args["input"], 'r') as f:
        for line in f.readlines():
            (s, d, t) = [int(item.strip()) for item in line.split(',')]
            edges.append((s, d, t))

    with open(args["labels"], 'r') as f:
        truth = f.readlines()
    return edges, truth


def write_scores(output_file: str, output: list[float]):

    with open(args["output"], 'w') as fout:
        for score in scores:
            fout.write(f"{score}\n")


if __name__ == "__main__":
    args = parse_input()

    match args["type"].lower():
        case "r":
            print("Using relational core")
            counter = MidasR(20, 2048, args["scale"])
        case "f":
            print("Using filtering core")
            counter = CMSCounter(20, 2048, args["scale"])
        case _:
            print("Using normal core")
            counter = CMSCounter(20, 2048)
    print("reading")
    edges, truth = read_data(args["input"], args["labels"])

    start = time.time()
    print("scoring")
    scores = counter.run(edges)
    end = time.time()

    print(f"ROC-AUC = {roc_auc_score(truth, scores):.4f}\n in {end-start}s")

    if len(args["output"]) > 1:
        print("writing")
        write_scores(args["output"], scores)
