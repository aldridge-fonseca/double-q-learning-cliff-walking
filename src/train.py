from __future__ import annotations

import argparse
import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt

from .double_q_agent import DoubleQLearningAgent


ACTION_NAMES = ["UP", "RIGHT", "DOWN", "LEFT"]


def moving_average(values: list[float], window_size: int) -> list[float]:
    return [
        sum(values[index:index + window_size]) / len(values[index:index + window_size])
        for index in range(0, len(values), window_size)
    ]


def save_learning_curve(rewards: list[float], output_path: Path, window_size: int = 10) -> None:
    averages = moving_average(rewards, window_size)
    x_values = list(range(window_size, len(rewards) + 1, window_size))

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, averages, linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel(f"Average reward per {window_size} episodes")
    plt.title("Double Q-learning on Cliff Walking")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def write_episode(episode: list[tuple[int, int, float]], done: bool, total_return: float | None, output_path: Path) -> None:
    trajectory = []
    for state, action, reward in episode:
        trajectory.extend([state, action, reward])

    with output_path.open("w", encoding="utf-8") as file:
        file.write("Optimal episode\n")
        file.write(f"Trajectory: {tuple(trajectory)}\n\n")

        for index, (state, action, reward) in enumerate(episode, start=1):
            file.write(f"Step {index}: state={state}, action={action} ({ACTION_NAMES[action]}), reward={reward}\n")

        file.write(f"\nTotal steps: {len(episode)}\n")
        if total_return is None:
            file.write("Discounted return: not available\n")
        else:
            file.write(f"Discounted return: {total_return:.2f}\n")
        file.write(f"Goal reached: {done}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Double Q-learning agent on Cliff Walking.")
    parser.add_argument("--episodes", type=int, default=5_000, help="Number of training episodes.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducible runs.")
    parser.add_argument("--plot", default="assets/learning_curve.png", help="Output path for the learning curve.")
    parser.add_argument("--episode", default="examples/optimal_episode.txt", help="Output path for the greedy episode summary.")
    args = parser.parse_args()

    random.seed(args.seed)
    env = gym.make("CliffWalking-v1")
    env.action_space.seed(args.seed)

    agent = DoubleQLearningAgent(env=env, total_episodes=args.episodes)
    rewards = agent.learn()

    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    save_learning_curve(rewards, plot_path)

    episode, done = agent.best_run(max_steps=100)
    total_return = agent.discounted_return(episode, done)

    episode_path = Path(args.episode)
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    write_episode(episode, done, total_return, episode_path)

    print(f"Learning curve saved to {plot_path}")
    print(f"Greedy episode saved to {episode_path}")
    print(f"Total steps: {len(episode)}")
    if total_return is None:
        print("Discounted return: not available")
    else:
        print(f"Discounted return: {total_return:.2f}")
    print(f"Goal reached: {done}")

    env.close()


if __name__ == "__main__":
    main()
