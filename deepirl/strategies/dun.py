import numpy as np
import tensorflow as tf

from deepirl.models.base import RQModelBase
from deepirl.environments.base import Environment
from deepirl.strategies.base import Strategy
from deepirl.utils import IncrementalMean
from deepirl.utils.replay import DqnReplayMemory


class DunStrategy(Strategy):
    def __init__(self, env: Environment, model: RQModelBase,
                 memory_capacity=1000000,
                 mini_batch_size=32,
                 discount_factor=0.96,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_over_steps=1000000,
                 mean_reward_for_episodes=100,
                 transfer_target_steps=1000):
        super(DunStrategy, self).__init__(env, model)
        self.env = env
        self.model = model
        self.replay_memory = DqnReplayMemory(memory_capacity,
                                             state_shape=env.state_shape,
                                             state_dtype=np.float32,
                                             action_dtype=np.uint16)

        self.transfer_target_steps = transfer_target_steps
        self.mini_batch_size = mini_batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_over_steps = epsilon_decay_over_steps
        self.epsilon = epsilon_start
        self.discount_factor = discount_factor

        # Mean reward for last N episodes
        self.mean_reward = IncrementalMean(mean_reward_for_episodes)

    def run(self, sess: tf.Session, num_episodes, verbose=False, *args, **kwargs):
        for episode in range(num_episodes):
            episode_reward = 0.0
            state = self.env.reset()

            while True:
                r, _, q, _ = self.model.predict_r_u_q_p(sess, [state])
                r = r[0]
                q = q[0]

                # E-greedy policy
                if np.random.rand() < self.epsilon:
                    action = np.random.choice(self.env.num_actions)
                else:
                    action = np.argmax(q)

                new_state = self.env.do_action(action)
                reward = r[action]
                is_terminal = self.env.is_terminal()

                episode_reward += reward

                self.replay_memory.append(state, action, reward, is_terminal)

                if len(self.replay_memory) > self.mini_batch_size:
                    self.train_on_replay(sess)

                # Decay epsilon exponentially
                self.epsilon = (self.epsilon_start - self.epsilon_end) * np.exp(
                    -5 * self.model.dqn_step / self.epsilon_decay_over_steps) + self.epsilon_end

                if self.model.double and self.model.dqn_step % self.transfer_target_steps == 0:
                    print('DQN: Copying weights from Eval to Target')
                    self.model.update_target(sess)

                if is_terminal:
                    break
                else:
                    state = new_state

            self.mean_reward.add(episode_reward)
            if verbose:
                print('Episode {0}/{1}, R={2:.3f}  MeanR={3:.3f} i={4} eps={5:.4f}'
                      .format(episode, num_episodes, episode_reward,
                              self.mean_reward.value, self.model.dqn_step, self.epsilon))
        if verbose:
            print('Finished run!')
            print('\tMean reward for last {0} episodes: {1:.3f}'.format(self.mean_reward.size, self.mean_reward.value))
        return self.mean_reward.value

    def train_on_replay(self, sess: tf.Session):
        states, actions, rewards, next_states, is_terminal = self.replay_memory.sample(self.mini_batch_size)

        if self.model.double:
            # Predict wit one call
            r_all, u_all, u_target_all = self.model.predict_vars(sess, np.concatenate((states, next_states)), [
                self.model.rewards,
                self.model.u,
                self.model.u_target
            ])

            # Split for current and next states
            _, u, _ = r_all[:len(states)], u_all[:len(states)], u_target_all[:len(states)]
            r_next, u_next, u_target_next = r_all[len(states):], u_all[len(states):], u_target_all[len(states):]

            # Q(s, a) <- R(s, a) + gamma * Q`(s', argmax a' (Q(s', a')))
            # Q(s, a) <- R(s, a) + U(s, a)
            # a` = argmax a' [ Q(s', a') ]   -- next a
            # U(s, a) <- gamma * Q`(s', a`)
            # U(s, a) <- gamma * ( R(s, a`) + U(s, a`) )
            next_a = np.argmax(r_next + u_next, axis=1)
            u_targets = self.discount_factor * (r_next[range(self.mini_batch_size), next_a] +
                                                u_target_next[range(self.mini_batch_size), next_a])
        else:
            r_all, u_all, _, _ = self.model.predict_r_u_q_p(sess, np.concatenate((states, next_states)))
            _, u = r_all[:len(states)], u_all[:len(states)]
            r_next, u_next = r_all[len(states):], u_all[len(states):]

            # Q(s, a) <- R(s, a) + gamma * max a' [ Q(s', a') ]
            # Q(s, a) <- R(s, a) + U(s, a)
            # U(s, a) <- gamma * max a' [ Q(s', a') ]
            # U(s, a) <- gamma * max a' [ R(s', a') + U(s', a') ]
            u_targets = self.discount_factor * np.max(r_next + u_next, axis=1)

        # Next expected return for terminal states is 0
        u_targets[is_terminal] = 0.0

        u[range(self.mini_batch_size), actions] = u_targets
        return self.model.train_u(sess, states, u, average_episode_reward=self.mean_reward.value)
