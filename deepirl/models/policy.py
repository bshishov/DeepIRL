import tensorflow as tf


class PolicyValueHead(object):
    def __init__(self, inputs: tf.Tensor, actions_dim: int, scope_name='PolicyValue'):
        weights_initializer = tf.contrib.layers.xavier_initializer

        with tf.variable_scope(scope_name, reuse=False):
            with tf.variable_scope('PolicyHead'):
                x = inputs
                x = tf.layers.dense(x, 64, name='fc1', activation=tf.nn.tanh,
                                    kernel_initializer=weights_initializer())
                x = tf.layers.dense(x, 64, name='fc2', activation=tf.nn.tanh,
                                    kernel_initializer=weights_initializer())
                x = tf.layers.dense(x, actions_dim, name='Mean', activation=None,
                                    kernel_initializer=weights_initializer())
                self.policy_mean = x

                # Std just from 1 variable
                log_std = tf.Variable(tf.zeros([actions_dim]) - 2.0, dtype=tf.float32, trainable=True, name='LogStd')
                self.policy_std = tf.nn.softplus(log_std) + 1e-5

                # Std as Dense from states
                """
                with tf.variable_scope('Std'):
                    log_std = tf.layers.dense(inputs, 1, name='fc1', activation=None,
                                              kernel_initializer=tf.zeros_initializer,
                                              use_bias=False)
                    log_std = tf.reshape(log_std, [-1])
                    self.policy_std = tf.nn.softplus(log_std) + 1e-5  # softplus(x) = log(exp(x) + 1)
                """

            with tf.variable_scope('ValueHead'):
                x = inputs
                x = tf.layers.dense(x, 64, name='fc1', activation=tf.nn.tanh,
                                    kernel_initializer=weights_initializer())
                x = tf.layers.dense(x, 64, name='fc2', activation=tf.nn.tanh,
                                    kernel_initializer=weights_initializer())
                x = tf.layers.dense(x, 1, name='fc3', activation=None, kernel_initializer=weights_initializer())
                self.value = tf.reshape(x, [-1])

            with tf.variable_scope('Distribution'):
                # Build a MultivariateNormal distribution with Mu from nn and std as trainable variable
                self.dist = tf.contrib.distributions.MultivariateNormalDiag(self.policy_mean, self.policy_std)
                self.sample_actions = self.dist.sample()

            self.trainable_variables = tf.trainable_variables(scope=scope_name)
