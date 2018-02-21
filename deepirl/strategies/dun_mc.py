import tensorflow as tf
import numpy as np

from deepirl.models.base import RQModelBase
from deepirl.environments.base import Environment
from deepirl.strategies.base import Strategy
from deepirl.utils import IncrementalMean
from deepirl.utils.replay import GenericMemory


class DunMcStrategy(Strategy):
    """
        Updates Q values with the accumulated rewards over a whole episode
    """
    def __init__(self, env: Environment, model: RQModelBase,
                 memory_capacity=100000,
                 discount_factor=0.96,
                 batch_size=64,
                 epsilon=0.5):
        super(DunMcStrategy, self).__init__(env, model)
        self.replay = GenericMemory(memory_capacity, [
            ('state', np.float32, env.state_shape),
            ('q', np.float32, env.num_actions),
        ])
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.mean_reward = IncrementalMean()

    def run(self, sess: tf.Session, num_episodes: int, *args, **kwargs):
        for episode in range(num_episodes):
            states, actions, rewards, u = self.play_episode(sess)
            total_reward = rewards.sum()
            self.mean_reward.add(total_reward)

            # DQN Targets:
            # Q(s, a, w') = r + gamma * max[a] Q(s', a, w)

            # Q value of the last action = R(s)
            # U value of the last action = 0.0
            u[-1, actions[-1]] = 0.0
            self.replay.append(states[-1], u[-1])

            # Discount rewards
            # From end to start:
            # Q(s, a, t) <- R(s, a, t) + gamma * Q(s, a, t + 1)  assuming that the next Q is: max a' [Q(s, a')]
            # U(s, a, y) <- gamma * (R(s, a, t + 1) + U(s, a, t + 1))
            for i in reversed(range(len(actions) - 1)):
                u[i, actions[i]] = self.discount_factor * (rewards[i + 1] + u[i + 1, actions[i + 1]])
                # q[i, actions[i]] = rewards[i] + self.discount_factor * np.max(q[i + 1])
                self.replay.append(states[i], u[i])

            if len(self.replay) > self.batch_size:
                batch_states, batch_u = self.replay.sample(self.batch_size)
                loss = self.model.train_u(sess, batch_states, batch_u)

                print('MC: Episode: {0}/{1} Loss={2:.5f} R: {3:.3f}  Avg R: {4:.3f}'
                      .format(episode, num_episodes, loss, total_reward, self.mean_reward.value))

    def play_episode(self, sess: tf.Session, use_policy: bool = False):
        states = []
        u_values = []
        actions = []
        rewards = []

        last_state = self.env.reset()

        while True:
            predicted_r, predicted_u, _, predicted_policy = self.model.predict_r_u_q_p(sess, [last_state])

            if use_policy:
                action = np.random.choice(self.env.num_actions, p=predicted_policy[0])
            else:
                # E-greedy
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(0, self.env.num_actions)
                else:
                    # Q = R + U
                    action = np.argmax(predicted_r[0] + predicted_u[0])

            new_state = self.env.do_action(action)

            states.append(last_state)
            actions.append(action)
            rewards.append(predicted_r[0][action])
            u_values.append(predicted_u[0])

            if self.env.is_terminal():
                break

            last_state = new_state

        return np.array(states), np.array(actions), np.array(rewards), np.array(u_values)
