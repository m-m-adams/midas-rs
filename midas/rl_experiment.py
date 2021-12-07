import matplotlib.pyplot as plt
import numpy as np
import torch
from rlthreshold import DQNAgent
from midas_env import MidasEnv
from tqdm import tqdm
from midas import read_data
from sklearn.metrics import f1_score


def run_experiment(environment, agent, env_info, experiment_parameters):
    actions = []
    # one agent setting

    agent = DQNAgent(state_space=(4, 1),
                     action_space=2,
                     max_memory_size=30000,
                     batch_size=32,
                     gamma=0.90,
                     lr=0.00025,
                     dropout=0.2,
                     exploration_max=1.0,
                     exploration_min=0.02,
                     exploration_decay=0.99,
                     pretrained=False)
    edges, truth = read_data('./data/darpa_processed.csv',
                             './data/darpa_ground_truth.csv')
    environment = environment(edges)
    edges = edges[:10000]
    truth = truth[:10000]
    state = environment.start()
    state = torch.Tensor([state]).unsqueeze(-1)
    for _ in tqdm(range(len(edges)-1)):

        action = agent.act(state)

        reward, state_next, terminal = environment.step(int(action[0]))
        state_next = torch.Tensor([state_next]).unsqueeze(-1)
        reward = torch.tensor([reward]).unsqueeze(0)

        terminal = torch.tensor([int(terminal)]).unsqueeze(0)
        agent.remember(state, action, reward, state_next, terminal)
        agent.experience_replay()

        state = state_next

        actions.append(action)

    plt.plot(np.array(truth[:len(actions)], np.int64) -
             np.array(actions, np.int64))
    plt.show()
    print(f1_score(truth[:len(actions)], actions))
    with open('./output.csv', 'w') as f:
        for label, pred in zip(truth, actions):
            f.write(f"{label}, {pred}\n")


if __name__ == "__main__":
    experiment_parameters = {
        "num_runs": 1,
        "num_episodes": 1000,
        "episode_eval_frequency": 100  # evaluate every 10 episode
    }

    # Environment parameters
    environment_parameters = {
        "decay": 0.99
    }

    current_env = MidasEnv
    current_agent = DQNAgent

    # run experiment
    run_experiment(current_env, current_agent,
                   environment_parameters, experiment_parameters)
