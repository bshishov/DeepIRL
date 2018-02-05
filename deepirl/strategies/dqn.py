import numpy as np
import tensorflow as tf

from deepirl.environments.base import Environment
from deepirl.models.dqn import Model
from deepirl.utils.replay import DqnReplayMemory, GenericMemory
from deepirl.utils import IncrementalMean


class Strategy(object):
    def __init__(self, env: Environment, model: Model):
        self.env = env
        self.model = model

    def run(self, sess: tf.Session, num_episodes: int, *args, **kwargs):
        raise NotImplementedError


class DqnStrategy(Strategy):
    def __init__(self, env: Environment, model: Model,
                 memory_capacity=1000000,
                 mini_batch_size=32,
                 discount_factor=0.96,
                 epsilon_start=1.0,
                 epsilon_end=0.05,
                 epsilon_decay_over_steps=1000000,
                 mean_reward_for_episodes=100,
                 transfer_target_steps=1000):
        super(self.__class__, self).__init__(env, model)
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
                r, _, q, _ = self.model.predict(sess, [state])
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

                if self.model.dqn_step % self.transfer_target_steps == 0:
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
            r_all, u_all, _, _ = self.model.predict(sess, np.concatenate((states, next_states)))
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
        return self.model.train_dqn(sess, states, u, average_episode_reward=self.mean_reward.value)


class McStrategy(Strategy):
    """
        Updates Q values with the accumulated rewards over a whole episode
    """

    def __init__(self, env: Environment, model: Model,
                 memory_capacity=1000000,
                 discount_factor=0.96,
                 batch_size=64,
                 epsilon=0.5):
        super(self.__class__, self).__init__(env, model)
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
                loss = self.model.train_dqn(sess, batch_states, batch_u)

                print('MC: Episode: {0}/{1} Loss={2:.5f} R: {3:.3f}  Avg R: {4:.3f}'
                      .format(episode, num_episodes, loss, total_reward, self.mean_reward.value))

    def play_episode(self, sess: tf.Session, use_policy: bool = False):
        states = []
        u_values = []
        actions = []
        rewards = []

        last_state = self.env.reset()

        while True:
            predicted_r, predicted_u, _, predicted_policy = self.model.predict(sess, [last_state])

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


class DqnBroadStrategy(Strategy):
    def __init__(self, env: Environment, model: Model,
                 memory_capacity=100000,
                 discount_factor=0.96,
                 batch_size=32,
                 epsilon=0.5):
        super(self.__class__, self).__init__(env, model)
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
                predicted_rewards, predicted_u, _, predicted_policy = self.model.predict(sess, [state])
                r = predicted_rewards[0]
                u = predicted_u[0]
                policy = predicted_policy[0]

                if not is_terminal:
                    # Select N best action w.r.t. policy
                    # Selected actions are like receptive field
                    actions = np.random.choice(self.env.num_actions, self.actions_to_sample, p=policy)

                    # Get next states for N actions, but not updating current states
                    # e.g. Monte Carlo node expansion
                    for i, action in enumerate(actions):
                        self.next_states[i] = self.env.do_action(action, update_state=False)

                    # Get best Q values for next states
                    next_r, next_u, _, _ = self.model.predict(sess, self.next_states)

                    # Update Q values of N actions as:
                    # Q(s, a) <- R(s, a) + gamma * max a' [ Q(s', a') ]
                    # U(s, a) <- gamma * max a' [ Q(s', a') ]
                    # Assuming that we are following the policy
                    #u[actions] = self.discount_factor * np.max(next_r + next_u, axis=1)

                    # TODO: double check
                    # Maybe use softmax over next_r + next_u directly
                    u[actions] = self.discount_factor * np.sum(policy * (next_r + next_u), axis=1)


                    # E-greedy policy
                    if np.random.rand() < self.epsilon:
                        action = np.random.choice(actions)
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
                    # so, U(s, a) <- 0.0
                    self.replay.append(state, np.zeros(self.env.num_actions))

                if len(self.replay) > self.batch_size:
                    batch_states, batch_u = self.replay.sample(self.batch_size)
                    self.model.train_dqn(sess, batch_states, batch_u, average_episode_reward=self.mean_reward.value)

                if is_terminal:
                    self.mean_reward.add(episode_reward)
                    print('DQN BROAD: Episode={0}/{1} R={2:.3f} MeanR={3:.3f}'.format(
                        episode, num_episodes, episode_reward, self.mean_reward.value))
                    break
