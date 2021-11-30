import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import optimizer
import numpy as np


class TDAgent():
    def __init__(self):
        self.name = "td_agent"
        pass

    class Net(nn.Module):

        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(3, 10)
            self.fc2 = nn.Linear(10, 1)

        def forward(self, x):

            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    def agent_init(self, agent_info={}):

        # Set random seed for weights initialization for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.valuenetwork = self.Net().to(self.device)

        self.optimizer = optim.Adam(
            self.valuenetwork.parameters(), lr=agent_info.get("step_size"))
        # Set random seed for policy for each run
        self.policy_rand_generator = np.random.RandomState(
            agent_info.get("seed"))

        self.gamma = agent_info.get("discount_factor")
        self.beta = agent_info.get("beta_v")
        self.epsilon = agent_info.get("epsilon")
        self.action_space = torch.tensor([0, 1])
        self.meanR = 0
        self.last_state = None
        self.last_action = None

    def agent_policy(self, state):

        # Set chosen_action as 0 or 1 with equal probability.
        maxv = -np.Inf
        a = self.action_space[0]
        for action in self.action_space:
            features = torch.cat((action.unsqueeze(0), state), 0)
            v = self.valuenetwork(features.to(self.device))
            if v > maxv:
                maxv = v
                a = action
        r = self.policy_rand_generator.uniform(0, 1)
        if r > self.epsilon:
            return a
        else:
            chosen_action = torch.tensor(
                self.policy_rand_generator.choice([0, 1]))
            return chosen_action

    def agent_start(self, state):

        self.last_state = state
        self.last_action = self.agent_policy(state)

        return self.last_action

    def agent_step(self, reward, state):

        # Compute TD error (5 lines)
        # delta = None
        self.optimizer.zero_grad()

        # add the last action to the feature vector
        features = features = torch.cat(
            (self.last_action.unsqueeze(0), self.last_state), 0)
        v_last = self.valuenetwork(features.to(self.device))

        a = self.agent_policy(state)
        features = torch.cat((a.unsqueeze(0), state), 0)
        v = self.valuenetwork(features.to(self.device))

        delta = reward - self.meanR + v - v_last

        self.meanR = self.meanR + self.beta*reward

        #print(v_last, target)
        L = nn.MSELoss()
        loss = L(v_last, delta.detach())

        loss.backward()
        self.optimizer.step()

        # update self.last_state and self.last_action (2 lines)

        ### START CODE HERE ###
        self.last_state = state
        self.last_action = a
        ### END CODE HERE ###

        return self.last_action

    def agent_end(self, reward):

        features = features = torch.cat(
            (self.last_action.unsqueeze(0), self.last_state), 0)
        v_last = self.valuenetwork(features.to(self.device))

        L = nn.MSELoss()
        loss = L(v_last, torch.tensor([[reward]]).to(self.device))
        loss.backward()
        self.optimizer.step()

    def get_state_value(self):
        with torch.no_grad():
            state_value = np.zeros(self.num_states)
            state_action = np.zeros(self.num_states)
            for state in range(self.num_states):
                maxv = -np.Inf
                a = 0
                for action in self.action_space:
                    features = torch.cat((action.unsqueeze(0), state), 0)
                    v = self.valuenetwork(features.to(self.device))
                    if v > maxv:
                        maxv = v
                        a = action
                state_value[state] = maxv.to("cpu").data.numpy()
                state_action[state] = a.to("cpu").data.numpy()
            return state_value, state_action
