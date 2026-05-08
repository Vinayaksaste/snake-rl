

import pickle
import random
from typing import Dict, Tuple

import numpy as np


class QLearningAgent:

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
    ):

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.q_table: Dict[
            Tuple[int, ...],
            np.ndarray
        ] = {}

    def _ensure_state(
        self,
        state: Tuple[int, ...]
    ):

        if state not in self.q_table:

            self.q_table[state] = np.zeros(
                3,
                dtype=np.float32
            )

    def choose_action(
        self,
        state: Tuple[int, ...],
        training: bool = True
    ) -> int:

        self._ensure_state(state)

        if training and random.random() < self.epsilon:

            return random.randint(0, 2)

        return int(
            np.argmax(self.q_table[state])
        )

    def learn(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool,
    ):

        self._ensure_state(state)
        self._ensure_state(next_state)

        current_q = self.q_table[state][action]

        if done:

            target = reward

        else:

            target = (
                reward
                + self.gamma
                * np.max(self.q_table[next_state])
            )

        self.q_table[state][action] = (
            current_q
            + self.alpha
            * (target - current_q)
        )

    def end_episode(self):

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )

    def save(self, path):

        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "q_table": self.q_table,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def save_custom(self, path):

        data = {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
            "q_table": self.q_table,
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path):

        with open(path, "rb") as f:
            data = pickle.load(f)

        agent = cls(
            alpha=data["alpha"],
            gamma=data["gamma"],
            epsilon=data["epsilon"],
            epsilon_decay=data["epsilon_decay"],
            min_epsilon=data["min_epsilon"],
        )

        agent.q_table = data["q_table"]

        return agent