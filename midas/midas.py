import count_min_sketch
from sklearn.metrics import roc_auc_score


class CMSCounter:
    def __init__(self, nrows: int, ncolumns: int):
        self.current = count_min_sketch.CMS(nrows, ncolumns)
        self.total = count_min_sketch.clone_cms(self.current)
        self.timestamp = 0

    def add(self, edge, time):
        if time > self.timestamp:
            self.roll_time(time)
        nc = self.current.insert(edge)
        nt = self.total.insert(edge)
        return self.score(nc, nt, time)

    def roll_time(self, time):
        self.current.clear()
        self.timestamp = time

    def score(self, a, s, t):
        return 0 if s == 0 or t == 1 else pow((a - s / t) * t, 2) / (s * (t - 1))


if __name__ == "__main__":

    counter = CMSCounter(2, 1024)
    print("reading")
    with open('./midas/darpa_processed.csv', 'r') as f:
        lines = f.readlines()

    with open('./midas/darpa_ground_truth.csv', 'r') as f:
        truth = f.readlines()

    print("scoring")
    output = []
    for line in lines:
        (s, d, t) = [int(item.strip()) for item in line.split(',')]

        score = counter.add((s, d), t)
        output.append((s, d, t, score))
        # print(output[-1])

    print("writing")
    with open('darpa_scored.csv', 'w+') as fout:
        for (s, d, t, score) in output:
            fout.write(f"{score}\n")
