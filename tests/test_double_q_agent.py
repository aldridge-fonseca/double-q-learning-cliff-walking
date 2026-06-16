import random
import unittest

import gymnasium as gym

from src.double_q_agent import DoubleQLearningAgent, argmax_action


class DoubleQLearningAgentTest(unittest.TestCase):
    def test_argmax_action_returns_one_of_the_best_actions(self):
        random.seed(1)
        action = argmax_action({0: 1.0, 1: 3.0, 2: 3.0})
        self.assertIn(action, {1, 2})

    def test_q_tables_match_environment_shape(self):
        env = gym.make("CliffWalking-v1")
        agent = DoubleQLearningAgent(env, total_episodes=1)

        self.assertEqual(len(agent.q1), env.observation_space.n)
        self.assertEqual(len(agent.q2[0]), env.action_space.n)
        env.close()

    def test_learning_returns_one_reward_per_episode(self):
        random.seed(2)
        env = gym.make("CliffWalking-v1")
        env.action_space.seed(2)
        agent = DoubleQLearningAgent(env, total_episodes=3)

        rewards = agent.learn()

        self.assertEqual(len(rewards), 3)
        self.assertTrue(all(isinstance(reward, float) for reward in rewards))
        env.close()


if __name__ == "__main__":
    unittest.main()
