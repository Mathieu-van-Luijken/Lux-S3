import numpy as np


class ReplayBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []

    def store(self, obs, actions, rewards, next_obs, done):
        self.observations.append(obs)
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.next_observations.append(next_obs)
        self.dones.append(done)

    def sample(self):
        return (
            np.array(self.observations),
            np.array(self.actions),
            np.array(self.rewards),
            np.array(self.next_observations),
            np.array(self.dones),
        )

    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.dones = []
