import tensorflow as tf
import numpy as np


def log_sigmoid(a):
    """Equivalent to tf.log(tf.sigmoid(a))"""
    return -tf.nn.softplus(-a)


def logit_bernoulli_entropy(logits):
    """ Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
    ent = (1.-tf.nn.sigmoid(logits))*logits - log_sigmoid(logits)
    return ent


def discriminator_head(observations: tf.Tensor, actions: tf.Tensor, hidden_size: int = 64):
        weights_initializer = (lambda scale: tf.initializers.variance_scaling(scale=scale))

        inputs = tf.concat((observations, actions), axis=1, name='Inputs')
        dense_1 = tf.layers.dense(inputs, hidden_size, name='Dense1', activation=tf.nn.tanh,
                                  kernel_initializer=weights_initializer(1.4))
        dense_2 = tf.layers.dense(dense_1, hidden_size, name='Dense2', activation=tf.nn.tanh,
                                  kernel_initializer=weights_initializer(1.4))
        logits = tf.layers.dense(dense_2, 1, name='Logits', activation=None,
                                 kernel_initializer=weights_initializer(0.01))
        logits = tf.reshape(logits, [-1])
        return logits


class GailDiscriminator(object):
    def __init__(self, observation_shape: tuple, actions_dim: tuple, entropy_coeff: float=0.001):
        with tf.variable_scope('Discriminator'):
            self.step = tf.Variable(0, trainable=False, name='DiscriminatorStep')

            with tf.variable_scope('Inputs'):
                self.generator_observations = tf.placeholder(tf.float32, (None,) + observation_shape, name='GeneratorObservations')
                self.expert_observations = tf.placeholder(tf.float32, (None,) + observation_shape, name='ExpertObservations')
                self.generator_actions = tf.placeholder(tf.float32, (None, actions_dim), name='GeneratorActions')
                self.expert_actions = tf.placeholder(tf.float32, (None, actions_dim), name='ExpertActions')

            with tf.variable_scope('Head', reuse=False):
                expert_logits = discriminator_head(self.expert_observations, self.expert_actions)
            with tf.variable_scope('Head', reuse=True):
                generator_logits = discriminator_head(self.expert_observations, self.expert_actions)

            with tf.variable_scope('Losses'):
                generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                         labels=tf.zeros_like(generator_logits))
                generator_loss = tf.reduce_mean(generator_loss, name='GeneratorLoss')

                expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits,
                                                                      labels=tf.ones_like(expert_logits))
                expert_loss = tf.reduce_mean(expert_loss, name='ExpertLoss')

                logits = tf.concat([generator_logits, expert_logits], 0)
                entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
                entropy_loss = -entropy_coeff * entropy

                self.total_loss = generator_loss + expert_loss + entropy_loss

            with tf.variable_scope('Reward'):
                self.reward = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)

            with tf.variable_scope('Optimization'):
                var_list = tf.trainable_variables('Discriminator/Head')
                opt = tf.train.AdamOptimizer(1e-3)
                self.optimize_op = opt.minimize(self.total_loss, global_step=self.step, var_list=var_list)

    def get_rewards(self, sess: tf.Session, observations, actions):
        rewards = sess.run(self.reward, feed_dict={
            self.generator_observations: observations,
            self.generator_actions: actions
        })
        return rewards
