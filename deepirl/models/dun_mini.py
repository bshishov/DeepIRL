import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from deepirl.models.base import RQModelBase


def one_hot(a, depth):
    r = np.zeros((len(a), depth), dtype=np.float32)
    r[range(len(a)), a] = 1.0
    return r


class DunMiniModel(RQModelBase):
    def __init__(self, input_shape, outputs, r_learning_rate=5e-4, u_learning_rate=1e-4, l2_scale=1e-5):
        super(DunMiniModel, self).__init__(double=False)
        self._outputs = outputs

        with tf.variable_scope('Inputs'):
            self._states = tf.placeholder(shape=[None, ] + list(input_shape), dtype=tf.float32)
            self._expert_policy = tf.placeholder(shape=[None, outputs], dtype=tf.float32)
            self._target_u = tf.placeholder(shape=[None, outputs], dtype=tf.float32)

            states_flat = layers.flatten(self._states)

        with tf.variable_scope('R'):
            x = layers.fully_connected(states_flat, 1024, activation_fn=tf.nn.relu)

            # Last layer is Sigmoid activated, so R(s, a) is bounded by [0, 1]
            self._r = layers.fully_connected(x, self._outputs, activation_fn=tf.nn.sigmoid)

        with tf.variable_scope('U'):
            x = layers.fully_connected(states_flat, 1024, activation_fn=tf.nn.relu)

            # Last layer, since R(s, a) is [0, 1] bounded, then expectations are [0, +inf) bounded
            # So we could use ReLU
            self._u = layers.fully_connected(x, self._outputs, activation_fn=tf.nn.relu, biases_initializer=None)

        with tf.variable_scope('Policy'):
            # Q(s, a) = R(s, a) + U(s, a)
            self._q = tf.add(self._r, self._u)
            policy_logits = self._q

            # Boltzmann policy:  p(s, a) = e ^ Q(s, a) / sum a' [ e ^ Q(s, a') ]
            self._policy = tf.nn.softmax(policy_logits)

        with tf.variable_scope('Optimization'):
            with tf.variable_scope('R'):
                train_vars = [v for v in tf.trainable_variables(scope='R')]
                print('R training vars:\n\t{0}'.format('\n\t'.join([v.name for v in train_vars])))
                with tf.variable_scope('Loss'):
                    l2_vars = [v for v in train_vars if 'weights' in v.name]
                    print('R L2 vars:\n\t{0}'.format('\n\t'.join([v.name for v in l2_vars])))

                    loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in l2_vars]) * l2_scale
                    self._r_loss = tf.losses.softmax_cross_entropy(onehot_labels=self._expert_policy,
                                                                   logits=policy_logits) + loss_l2

                opt = tf.train.AdamOptimizer(learning_rate=r_learning_rate)
                self._r_train_op = opt.minimize(self._r_loss, var_list=train_vars, global_step=self._irl_step_tf)

            with tf.variable_scope('U'):
                train_vars = [v for v in tf.trainable_variables(scope='U')]
                print('U training vars:\n\t{0}'.format('\n\t'.join([v.name for v in train_vars])))

                with tf.variable_scope('Loss'):
                    l2_vars = [v for v in train_vars if 'weights' in v.name]
                    print('U L2 vars:\n\t{0}'.format('\n\t'.join([v.name for v in l2_vars])))
                    loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in l2_vars]) * l2_scale

                    # Huber loss
                    self._u_loss = tf.losses.huber_loss(self._target_u, self._u) + loss_l2

                opt = tf.train.AdamOptimizer(learning_rate=u_learning_rate)
                self._u_train_op = opt.minimize(self._u_loss, var_list=train_vars, global_step=self._dqn_step_tf)

    def train_r(self, sess, states, expert_actions, *args, **kwargs):
        feed_dict = {self._states: states, self._expert_policy: one_hot(expert_actions, self._outputs)}
        loss, _, self.irl_step = sess.run([self._r_loss, self._r_train_op, self._irl_step_tf], feed_dict=feed_dict)
        return loss

    def train_u(self, sess, states, target_u, *args, **kwargs):
        feed_dict = {self._states: states, self._target_u: target_u}
        loss, _, self.dqn_step = sess.run([self._u_loss, self._u_train_op, self._dqn_step_tf], feed_dict=feed_dict)
        return loss

    def predict_r_u_q_p(self, sess: tf.Session, states, *args, **kwargs):
        var_set = [self._r, self._u, self._q, self._policy]
        rewards, u, q, policy = sess.run(var_set, feed_dict={self._states: states})
        return rewards, u, q, policy
