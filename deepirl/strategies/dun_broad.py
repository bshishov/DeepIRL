import tensorflow as tf
import numpy as np

from deepirl.models.base import RQModelBase
from deepirl.environments.base import Environment
from deepirl.strategies.base import Strategy
from deepirl.utils import IncrementalMean
from deepirl.utils.replay import GenericMemory


class DunBroadStrategy(Strategy):
    def __init__(self, env: Environment, model: RQModelBase,
                 memory_capacity=10000,
                 discount_factor=0.96,
                 batch_size=32,
                 epsilon=0.5):
        super(DunBroadStrategy, self).__init__(env, model)
        self.replay = GenericMemory(memory_capacity, [
            ('state', np.float32, env.state_shape),
            ('u', np.float32, env.num_actions),
        ])
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.actions_to_sample = min(env.num_actions, batch_size)
        self.next_states = np.zeros((self.actions_to_sample,) + env.state_shape, dtype=np.float32)
        self.mean_reward = IncrementalMean(100)

    def run(self, sess: tf.Session, num_episodes: int, *args, **kwargs):
        for episode in range(num_episodes):
            # Episode start
            state = self.env.reset()
            episode_reward = 0.0

            while True:
                # If state is terminal
                is_terminal = self.env.is_terminal()

                # Predict R(s, a), U(s, a), Q(s, a), policy(s, a)  for current state s
                r, u, _, p = self.model.predict_r_u_q_p(sess, [state])
                r = r[0]
                u = u[0]
                p = p[0]

                if not is_terminal:
                    # Select N best action w.r.t. policy
                    # Selected actions are like receptive field
                    #actions = np.random.choice(self.env.num_actions, self.actions_to_sample, p=p, replace=False)

                    # Select N completely random actions (uniformly)
                    actions = np.random.choice(self.env.num_actions, self.actions_to_sample, replace=False)

                    # Get next states for N actions, but not updating internal environment state
                    # e.g. Monte Carlo node expansion
                    for i, action in enumerate(actions):
                        self.next_states[i] = self.env.do_action(action, update_state=False)

                    # Get best Q values for next states
                    next_r, next_u, _, _ = self.model.predict_r_u_q_p(sess, self.next_states)

                    # Update Q values of N performed actions as:
                    # Q(s, a) <- R(s, a) + gamma * max a' [ Q(s', a') ]  -- Original DQN update
                    # U(s, a) <- gamma * max a' [ Q(s', a') ]
                    # Assuming that we are following the policy
                    u[actions] = self.discount_factor * np.max(next_r + next_u, axis=1)

                    # E-greedy policy
                    if np.random.rand() < self.epsilon:
                        action = np.random.choice(self.env.num_actions)
                    else:
                        # Choose best possible action
                        action = np.argmax(u + r)

                    episode_reward += r[action]

                    self.replay.append(state, u)

                    # Make an MDP step
                    state = self.env.do_action(action)
                else:
                    # For terminal state:
                    # Q(s, a) <- R(s,a)
                    # so, U(s, a) <- 0.0, so expectations of the next rewards are 0
                    self.replay.append(state, np.zeros(self.env.num_actions))

                if len(self.replay) > self.batch_size:
                    batch_states, batch_u = self.replay.sample(self.batch_size)
                    self.model.train_u(sess, batch_states, batch_u, average_episode_reward=self.mean_reward.value)

                if is_terminal:
                    self.mean_reward.add(episode_reward)
                    print('DQN BROAD: Episode={0}/{1} R={2:.3f} MeanR={3:.3f}'.format(
                        episode, num_episodes, episode_reward, self.mean_reward.value))
                    break
