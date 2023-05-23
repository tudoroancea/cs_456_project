from abc import ABC, abstractmethod
from typing import Union
import gym
import numpy as np
import torch


class NormalizedEnv(gym.ActionWrapper):
    """Wrap action"""

    def action(self, action):
        act_k = (self.action_space.high - self.action_space.low) / 2.0
        act_b = (self.action_space.high + self.action_space.low) / 2.0
        return act_k * action + act_b

    def reverse_action(self, action):
        act_k_inv = 2.0 / (self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low) / 2.0
        return act_k_inv * (action - act_b)


class Agent(ABC):
    @abstractmethod
    def compute_action(
        self, state: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        pass


class RandomAgent(Agent):
    def __init__(self, env):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

    def compute_action(self, state: np.ndarray):
        state = np.atleast_2d(state)
        return np.random.uniform(-1, 1, size=(state.shape[0], self.action_size))
