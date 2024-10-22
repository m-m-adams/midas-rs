import numpy as np
import torch
import unittest
from midas import read_data
from midas_cores import MidasR


class MidasEnv():
    def __init__(self, edges: list[tuple[int, int, int]], decay=0.99, seed=0):

        self.decay = decay
        self.edges = edges
        self.midas = MidasR(20, 2048)
        self.i = 0

    def start(self) -> torch.tensor:

        # set self.reward_state_term tuple
        reward = 0.0

        (s, d, t) = self.edges[self.i]
        scores = self.midas.run_one((s, d), t)
        state = [0]
        state.extend(scores)
        self.is_terminal = False
        self.state = state
        self.i += 1

        # return first state from the environment
        return self.state

    def reward(self) -> float:
        scores = self.state[1:]
        count = self.state[0]
        return max(scores) - (count)

    def step(self, action: int) -> tuple[torch.tensor, torch.tensor, bool]:
        state = [self.state[0]*self.decay+action]

        (s, d, t) = self.edges[self.i]
        scores = self.midas.run_one((s, d), t)
        state.extend(scores)

        self.state = state

        self.i += 1

        if self.i == len(self.edges):
            terminal = True
        else:
            terminal = False

        if action == 0:

            reward = 0
        else:
            reward = self.reward()

        return reward, self.state, terminal


class TestEnv(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        edges, truth = read_data(
            './data/darpa_processed.csv', './data/darpa_ground_truth.csv')
        env_info = {
            "seed": 0,
            "decay": 0.6,
        }
        self. env = MidasEnv(edges, env_info)

    def test_init(self):

        self.assertEqual(self.env.edges[0], (7577, 9765, 1))

    def test_start(self):
        s = self.env.env_start()
        e = torch.tensor([0, 0, 0, 0])
        r = ((s-e)**2).mean().numpy()
        self.assertAlmostEqual(r, 0)

    def test_flag(self):
        self.env.env_start()
        self.env.step(0)

        r, s, t = self.env.step(1)
        self.assertAlmostEqual(r.numpy(), 1)

        r, s, t = self.env.step(1)
        self.assertAlmostEqual(r.numpy(), 1.25)

        r, s, t = self.env.step(1)
        self.assertAlmostEqual(r.numpy(), 1.02, 3)

    def test_end(self):
        self.env.env_start()
        self.env.step(0)
        self.env.i = len(self.env.edges)-1
        r, s, t = self.env.step(0)

        self.assertEqual(t, True)


if __name__ == '__main__':
    unittest.main()
