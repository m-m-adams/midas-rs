import matplotlib.pyplot as plt
import numpy as np

from rlthreshold import TDAgent
from midas_env import MidasEnv
from tqdm import tqdm
from midas import read_data
from sklearn.metrics import f1_score


def run_experiment(environment, agent, env_info, agent_info, experiment_parameters):
    actions = []
    # one agent setting
    edges, truth = read_data('./data/darpa_processed.csv',
                             './data/darpa_ground_truth.csv')
    edges = edges[:10000]
    truth = truth[:10000]

    environment = environment(edges, env_info)
    agent = agent(agent_info)
    state = environment.start()
    agent.agent_start(state)
    action = agent.agent_policy(state)
    actions.append(action)
    for _ in tqdm(range(len(edges)-1)):
        (reward, state, terminal) = environment.step(action)
        agent.agent_step(reward, state)
        action = agent.agent_policy(state)
        actions.append(action)

    plt.plot(np.array(truth, np.int64)-np.array(actions, np.int64))
    plt.show()
    print(f1_score(truth, actions))
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
        "decay": 0.6
    }

    # Agent parameters
    agent_parameters = {
        "num_hidden_layer": 1,
        "step_size": 0.01,
        "beta_m": 0.9,
        "beta_v": 0.999,
        "epsilon": 0.01,
    }

    current_env = MidasEnv
    current_agent = TDAgent

    # run experiment
    run_experiment(current_env, current_agent, environment_parameters,
                   agent_parameters, experiment_parameters)
