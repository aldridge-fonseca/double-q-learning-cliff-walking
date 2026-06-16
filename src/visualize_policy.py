from __future__ import annotations

import argparse
import random
import time

import gymnasium as gym

from .double_q_agent import DoubleQLearningAgent


ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and visualize a greedy Double Q-learning run.")
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--delay", type=float, default=0.15)
    args = parser.parse_args()

    random.seed(args.seed)
    training_env = gym.make("CliffWalking-v1")
    training_env.action_space.seed(args.seed)
    agent = DoubleQLearningAgent(env=training_env, total_episodes=args.episodes)
    agent.learn()
    training_env.close()

    render_env = gym.make("CliffWalking-v1", render_mode="human")
    agent.env = render_env
    episode, done = agent.best_run(max_steps=100)

    print("Greedy episode")
    for index, (state, action, reward) in enumerate(episode, start=1):
        print(f"Step {index}: state={state}, action={action} ({ACTION_NAMES[action]}), reward={reward}")
        time.sleep(args.delay)

    print(f"Goal reached: {done}")
    render_env.close()


if __name__ == "__main__":
    main()
