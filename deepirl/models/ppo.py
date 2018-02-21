import numpy as np
import tensorflow as tf
import gym

import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler

from deepirl.models.policy import PolicyValueHead
from deepirl.utils.reward import expectations
from deepirl.utils.replay import GenericMemory
from deepirl.utils import IncrementalMean
from deepirl.utils.tf_log import TfLogger


class GymEnvWrapper(object):
    def __init__(self, key):
        self._env = gym.make(key)

        observation_examples = np.array([self._env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

    @property
    def state_shape(self):
        return (400,)

    @property
    def action_space(self):
        return self._env.action_space

    def _featurize_state(self, state):
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def step(self, *args, **kwargs):
        state, reward, is_done, meta = self._env.step(*args, **kwargs)
        return self._featurize_state(state), reward, is_done, meta

    def reset(self):
        state = self._env.reset()
        return self._featurize_state(state)


class ProximalPolicyOptimization(object):
    def __init__(self,
                 observation_shape: tuple,
                 actions_dim: int,
                 clip_epsilon: float = 0.3,
                 entropy_beta: float = 1e-4,
                 learning_rate: float = 3e-4,
                 value_loss_coeff: float = 1.0,
                 max_steps=5e5):
        with tf.variable_scope('Inputs'):
            self.observations = tf.placeholder(tf.float32, (None,) + observation_shape, name='Observations')
            self.actions = tf.placeholder(tf.float32, [None, actions_dim], name='Actions')
            self.target_values = tf.placeholder(tf.float32, [None], name='TargetValues')
            self.advantages = tf.placeholder(tf.float32, [None], name='Advantages')
            #self.old_log_prob = tf.placeholder(tf.float32, [None], name='OldLogProbabilities')

        self.global_step = tf.Variable(0, name="GlobalStep", trainable=False, dtype=tf.int32)

        self.policy = PolicyValueHead(self.observations, actions_dim, scope_name='PolicyValue')
        self.policy_old = PolicyValueHead(self.observations, actions_dim, scope_name='PolicyValueOld')

        with tf.variable_scope('Loss'):
            # Entropy
            kl_old_new_div = tf.distributions.kl_divergence(self.policy_old.dist, self.policy.dist)
            mean_kl = tf.reduce_mean(kl_old_new_div)
            entropy = tf.reduce_mean(self.policy.dist.entropy())
            deacy_entropy_beta = tf.train.polynomial_decay(entropy_beta, self.global_step, max_steps, 1e-5, power=1.0)
            loss_entropy = (-deacy_entropy_beta) * entropy

            # L^VF - Value function loss
            # loss_value = tf.reduce_mean(tf.squared_difference(self.policy.value, self.target_values))
            loss_value = tf.losses.huber_loss(self.target_values, self.policy.value)

            # L^CLIP - Clipped loss (surrogate objectives)
            decay_epsilon = tf.train.polynomial_decay(clip_epsilon, self.global_step, max_steps, 1e-2, power=1.0)

            dist_ratio = tf.exp(self.policy.dist.log_prob(self.actions) - self.policy_old.dist.log_prob(self.actions))

            surr_1 = dist_ratio * self.advantages
            surr_2 = tf.clip_by_value(dist_ratio, 1.0 - decay_epsilon, 1.0 + decay_epsilon) * self.advantages
            loss_clip = -tf.reduce_mean(tf.minimum(surr_1, surr_2))
            self.total_loss = loss_clip + loss_value * value_loss_coeff + loss_entropy

            self.losses = {
                'L_CLIP': loss_clip,
                'ValueLoss': loss_value,
                'PolicyEntropyLoss': loss_entropy,
                'KLPolicyOld': mean_kl,
                'PolicyEntropy': entropy
            }

        with tf.variable_scope('Optimization'):
            self.learning_rate = tf.train.polynomial_decay(learning_rate, self.global_step, max_steps, 1e-10, power=1.0)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-5)
            self.optimize_op = optimizer.minimize(self.total_loss,
                                                  var_list=self.policy.trainable_variables,
                                                  global_step=self.global_step)

        with tf.variable_scope('Transfer'):
            self.transfer_vars = [tf.assign(target, source) for target, source in
                                  zip(self.policy_old.trainable_variables, self.policy.trainable_variables)]

    def sample_actions_and_value(self, sess: tf.Session, observations):
        actions, values = sess.run([self.policy_old.sample_actions, self.policy_old.value],
                                   feed_dict={self.observations: observations})
        return actions, values

    def sample_single_action_and_value(self, sess, observation):
        actions, values = self.sample_actions_and_value(sess, [observation])
        return actions[0], values[0]

    def train(self, sess: tf.Session, observations, actions, values, advantages, verbose=True):
        loss_info_ops = list(self.losses.values())
        results = sess.run([self.total_loss, self.optimize_op] + loss_info_ops, feed_dict={
            self.observations: observations,
            self.actions: actions,
            self.target_values: values,
            self.advantages: advantages
        })
        loss = results[0]
        if verbose:
            print('Total Loss: {0:.4f}'.format(loss))
            loss_info = results[2:]
            for key, info in zip(list(self.losses.keys()), loss_info):
                print('\t{0}: {1:.4f}'.format(key, info))
        return loss

    def update_old_policy(self, sess: tf.Session):
        sess.run(self.transfer_vars)


class Runner(object):
    def __init__(self, env: gym.Env, ppo: ProximalPolicyOptimization, num_steps: int, logger: TfLogger = None):
        self.rollout = GenericMemory(num_steps, [
            #('observations', np.float32, env.state_shape),
            ('observations', np.float32, env.observation_space.shape),
            ('actions', np.float32, env.action_space.shape),
            ('rewards', np.float32, ()),
            ('values', np.float32, ()),
            ('next_is_terminal', np.bool, ())
        ])
        self.env = env
        self.ppo = ppo
        self.observation = env.reset()
        self.num_steps = num_steps
        self.running_reward = IncrementalMean(20)
        self.episode_reward = 0.0
        self.episode = 0
        self.logger = logger

    def run(self, sess: tf.Session):
        for step in range(self.num_steps):
            # if self.running_reward.value is not None and self.running_reward.value > 10.0:
            # self.env.render()
            action, value = self.ppo.sample_single_action_and_value(sess, self.observation)
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            new_observation, reward, is_terminal, _ = self.env.step(action)
            self.episode_reward += reward

            # Is terminal indicates whether new_observation if from terminal state
            self.rollout.append(self.observation, action, reward, value, is_terminal)
            if is_terminal:
                self.running_reward.add(self.episode_reward)
                if self.logger is not None:
                    self.logger.log_scalar(self.episode_reward, 'EpisodeReward')
                    self.logger.log_scalar(self.running_reward.value, 'MeanEpisodeReward')

                print('Episode {0} finished: R:{1:.3f} MeanR:{2:.3f}'
                      .format(self.episode, self.episode_reward, self.running_reward.value))

                self.observation = self.env.reset()
                self.episode += 1
                self.episode_reward = 0.0
            else:
                self.observation = new_observation

        returns, advantages = expectations(self.rollout['rewards'][:-1],
                                           self.rollout['values'][:-1],
                                           self.rollout['next_is_terminal'][:-1],
                                           bootstrap_value=self.rollout['values'][-1], lam=1.0)

        # Normalize advantages for better gradients
        advantages = (advantages - advantages.mean()) / advantages.std()
        return self.rollout['observations'][:-1], self.rollout['actions'][:-1], returns, advantages


def main():
    env_name = 'MountainCarContinuous-v0'
    # env_name = 'Hopper-v1'
    # env_name = 'BipedalWalker-v2'
    #env_name = 'LunarLanderContinuous-v2'

    #env = GymEnvWrapper(env_name)
    env = gym.make(env_name)

    actions_dim = env.action_space.shape[0]
    #observations_shape = env.state_shape
    observations_shape = env.observation_space.shape

    batch_size = 256
    epochs = 5
    rollout_len = 4096
    num_batches = rollout_len // batch_size
    ppo = ProximalPolicyOptimization(observations_shape, actions_dim)
    logger = TfLogger('D:/ppo_logs/',
                      run_name=env_name,
                      step_tensor=ppo.global_step,
                      every_n_steps=rollout_len,
                      write_each_update=False)
    runner = Runner(env, ppo, num_steps=rollout_len, logger=logger)

    logger.track_tensor(ppo.total_loss, 'TotalLoss', scope='Losses')

    for name, tensor in ppo.losses.items():
        logger.track_tensor(tensor, name=name, scope='Losses')

    with tf.train.MonitoredSession(hooks=[logger]) as sess:
        while True:
            observations, actions, returns, advantages = runner.run(sess)
            for e in range(epochs):
                indices = np.arange(len(observations))
                np.random.shuffle(indices)
                for i in range(num_batches):
                    batch_indices = indices[i * batch_size:i * batch_size + batch_size]
                    batch_s = observations[batch_indices]
                    batch_a = actions[batch_indices]
                    batch_v = returns[batch_indices]
                    batch_adv = advantages[batch_indices]

                    verbose = e == epochs - 1 and i == num_batches - 1
                    ppo.train(sess, batch_s, batch_a, batch_v, batch_adv, verbose=verbose)
            ppo.update_old_policy(sess)
            logger.write(step=runner.episode)


if __name__ == '__main__':
    main()
