from __future__ import annotations

import random
from typing import Any

import gymnasium as gym


def argmax_action(values: dict[Any, float]) -> Any:
    best_value = max(values.values())
    best_actions = [action for action, value in values.items() if value == best_value]
    return random.choice(best_actions)


class ValueAgent:
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.98,
        epsilon: float = 0.2,
        learning_rate: float = 0.02,
        total_episodes: int = 5_000,
    ) -> None:
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.total_episodes = total_episodes
        self.q = self.init_q_table(env.observation_space.n, env.action_space.n)

    def init_q_table(self, n_states: int, n_actions: int, initial_value: float = 0.0) -> dict[int, dict[int, float]]:
        return {
            state: {action: initial_value for action in range(n_actions)}
            for state in range(n_states)
        }

    def epsilon_greedy(self, state: int, exploration: bool = True) -> int:
        if exploration and random.random() < self.epsilon:
            return self.env.action_space.sample()
        return argmax_action(self.q[state])

    def best_run(self, max_steps: int = 100) -> tuple[list[tuple[int, int, float]], bool]:
        episode = []
        state, _ = self.env.reset()

        for _ in range(max_steps):
            action = self.epsilon_greedy(state, exploration=False)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            episode.append((state, action, reward))

            if terminated or truncated:
                return episode, True

            state = next_state

        return episode, False

    def discounted_return(self, episode: list[tuple[int, int, float]], done: bool = False) -> float | None:
        if not done:
            return None

        return sum((self.gamma ** index) * reward for index, (_, _, reward) in enumerate(episode))


class DoubleQLearningAgent(ValueAgent):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.98,
        epsilon: float = 0.2,
        learning_rate: float = 0.02,
        total_episodes: int = 5_000,
    ) -> None:
        super().__init__(env, gamma, epsilon, learning_rate, total_episodes)
        self.q1 = self.init_q_table(env.observation_space.n, env.action_space.n)
        self.q2 = self.init_q_table(env.observation_space.n, env.action_space.n)

    def choose_action(self, state: int) -> int:
        if random.random() < self.epsilon:
            return self.env.action_space.sample()

        combined_values = {
            action: self.q1[state][action] + self.q2[state][action]
            for action in range(self.env.action_space.n)
        }
        return argmax_action(combined_values)

    def learn(self) -> list[float]:
        episode_rewards = []

        for _ in range(self.total_episodes):
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0

            while not (terminated or truncated):
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                total_reward += reward

                if random.random() < 0.5:
                    best_next_action = argmax_action(self.q1[next_state])
                    target = reward + self.gamma * self.q2[next_state][best_next_action]
                    self.q1[state][action] += self.learning_rate * (target - self.q1[state][action])
                else:
                    best_next_action = argmax_action(self.q2[next_state])
                    target = reward + self.gamma * self.q1[next_state][best_next_action]
                    self.q2[state][action] += self.learning_rate * (target - self.q2[state][action])

                state = next_state

            episode_rewards.append(total_reward)

        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                self.q[state][action] = (self.q1[state][action] + self.q2[state][action]) / 2.0

        return episode_rewards
