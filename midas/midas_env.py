import numpy as np
import torch
from midas import read_data
from midas_cores import MidasR


class RandomWalkEnvironment():
    def env_init(self, edges_path: str, label_path: str, env_info={}):

        # set random seed for each run

        self.rand_generator = np.random.RandomState(env_info.get("seed"))
        self.edges, self.truth = read_data(
            edges_path, label_path)
        self.midas = MidasR(20, 2048)

    def env_start(self):

        # set self.reward_state_term tuple
        reward = 0.0

        nodes = self.edges[0]
        scores = self.midas(nodes)
        state = torch.tensor(nodes+scores)
        is_terminal = False

        self.reward_state_term = (reward, state, is_terminal)

        # return first state from the environment
        return self.reward_state_term[1]

    def env_step(self, action):
        """A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state,
                and boolean indicating if it's terminal.
        """

        last_state = self.reward_state_term[1]

        # set reward, current_state, and is_terminal
        #
        # action: specifies direction of movement - 0 (indicating left) or 1 (indicating right)  [int]
        # current state: next state after taking action from the last state [int]
        # reward: -1 if terminated left, 1 if terminated right, 0 otherwise [float]
        # is_terminal: indicates whether the episode terminated [boolean]
        #
        # Given action (direction of movement), determine how much to move in that direction from last_state
        # All transitions beyond the terminal state are absorbed into the terminal state.

        if action == 0:  # left
            current_state = max(self.left_terminal_state, last_state +
                                self.rand_generator.choice(range(-100, 0)))
        elif action == 1:  # right
            current_state = min(
                self.right_terminal_state, last_state + self.rand_generator.choice(range(1, 101)))
        else:
            raise ValueError("Wrong action value")

        # terminate left
        if current_state == self.left_terminal_state:
            reward = -1.0
            is_terminal = True

        # terminate right
        elif current_state == self.right_terminal_state:
            reward = 1.0
            is_terminal = True

        else:
            reward = 0.0
            is_terminal = False

        self.reward_state_term = (reward, current_state, is_terminal)

        return self.reward_state_term
