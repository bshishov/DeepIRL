import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.layers as layers


def one_hot(a, depth=1024):
    r = np.zeros((len(a), depth), dtype=np.float32)
    r[range(len(a)), a] = 1.0
    return r


def simple_embedding(input_tensor: tuple, num_hidden):
    x = layers.flatten(input_tensor)
    x = layers.fully_connected(x, num_hidden)
    return x


class Model(object):
    def __init__(self, input_shape, outputs,
                 irl_lr=1e-3,
                 dqn_lr=1e-3,
                 l2_scale=1e-5,
                 ):
        self.dqn_step = 1
        self.irl_step = 1

        self._outputs = outputs

        self._dqn_step = tf.Variable(self.dqn_step, name='dqn_step', trainable=False)
        self._irl_step = tf.Variable(self.irl_step, name='irl_step', trainable=False)

        with tf.variable_scope('Inputs'):
            self.states = tf.placeholder(shape=[None, ] + list(input_shape), dtype=tf.float32)
            self.expert_policy = tf.placeholder(shape=[None, outputs], dtype=tf.float32)
            self.target_u = tf.placeholder(shape=[None, outputs], dtype=tf.float32)
            self._is_training = tf.placeholder(tf.bool, shape=(), name="IsTraining")

        self.rewards = self._build_reward_head(scope='Reward')

        with tf.variable_scope('DQN'):
            self.u = self._build_dqn_head('Eval')

        with tf.variable_scope('Policy'):
            self.q = tf.add(self.rewards, self.u)
            policy_logits = self.q
            self.policy = tf.nn.softmax(policy_logits)

        with tf.variable_scope('Optimization'):
            with tf.variable_scope('Reward'):
                train_vars = [v for v in tf.trainable_variables(scope='Reward')]
                print('Irl training vars:\n\t{0}'.format('\n\t'.join([v.name for v in train_vars])))
                with tf.variable_scope('Loss'):
                    l2_vars = [v for v in train_vars if 'weights' in v.name]
                    print('Irl L2 vars:\n\t{0}'.format('\n\t'.join([v.name for v in l2_vars])))

                    loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in l2_vars]) * l2_scale
                    self.irl_loss = tf.losses.softmax_cross_entropy(onehot_labels=self.expert_policy,
                                                                    logits=policy_logits) + loss_l2

                opt = tf.train.AdamOptimizer(learning_rate=irl_lr)
                self.irl_train_op = opt.minimize(self.irl_loss, var_list=train_vars, global_step=self._irl_step)

            with tf.variable_scope('DQN'):
                train_vars = [v for v in tf.trainable_variables(scope='DQN')]
                train_vars = [v for v in train_vars if 'Target' not in v.name]
                print('DQN training vars:\n\t{0}'.format('\n\t'.join([v.name for v in train_vars])))

                with tf.variable_scope('Loss'):
                    l2_vars = [v for v in train_vars if 'Outputs' not in v.name]
                    l2_vars = [v for v in l2_vars if 'Target' not in v.name]
                    l2_vars = [v for v in l2_vars if 'weights' in v.name]
                    print('DQN L2 vars:\n\t{0}'.format('\n\t'.join([v.name for v in l2_vars])))
                    loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in l2_vars]) * l2_scale

                    # Huber loss
                    self.dqn_loss = tf.losses.huber_loss(self.target_u, self.u) + loss_l2

                opt = tf.train.AdamOptimizer(learning_rate=dqn_lr)
                self.dqn_train_op = opt.minimize(self.dqn_loss, var_list=train_vars, global_step=self._dqn_step)

        with tf.variable_scope('Saver'):
            self.saver = tf.train.Saver()

    def _build_dqn_head(self, scope='DqnHead'):
        with tf.variable_scope(scope):
            embedding = simple_embedding(self.states, self._outputs)

            x = layers.fully_connected(embedding, self._outputs, activation_fn=tf.nn.relu)
            out = x

            with tf.variable_scope('Outputs'):
                u = layers.fully_connected(out, self._outputs, activation_fn=None, biases_initializer=None)
            return u

    def _build_reward_head(self, scope='Reward'):
        with tf.variable_scope(scope):
            embedding = simple_embedding(self.states, self._outputs)
            x = layers.fully_connected(embedding, self._outputs, activation_fn=tf.nn.relu)

            # Last layer is Sigmoid activated, so rewards for state are bounded by (0, +1)
            net = layers.fully_connected(x, self._outputs, activation_fn=tf.nn.sigmoid)

            # OR clipped ReLU could be used:
            # net = tf.clip_by_value(layers.fully_connected(net, outputs, activation_fn=tf.nn.relu), 0.0, MAX_R)

            return net

    def train_irl(self, sess, states, expert_actions):
        feed_dict = {self.states: states, self.expert_policy: one_hot(expert_actions, self._outputs)}
        loss, _ = sess.run([self.irl_loss, self.irl_train_op], feed_dict=feed_dict)
        return loss

    def train_dqn(self, sess, states, target_u, *args, **kwargs):
        feed_dict = {self.states: states, self.target_u: target_u, self._is_training: True}
        loss, _, self.dqn_step = sess.run([self.dqn_loss, self.dqn_train_op, self._dqn_step], feed_dict=feed_dict)
        return loss

    def predict(self, sess, states):
        var_set = [self.rewards, self.u, self.q, self.policy]
        rewards, u, q, policy = sess.run(var_set, feed_dict={self.states: states, self._is_training: False})
        return rewards, u, q, policy

    def predict_vars(self, sess, states, variables: list):
        return sess.run(variables, feed_dict={self.states: states, self._is_training: False})

    def update_target(self, sess):
        pass

    def save(self, sess, path):
        full_path = self.saver.save(sess, path)
        print('Model saved to {0}'.format(full_path))

    def load(self, sess, path):
        self.saver.restore(sess, path)
        self.dqn_step, self.irl_step = sess.run([self._dqn_step, self._irl_step])
        print('Model restored, DQN step: {0}, IRL step: {1}'.format(self.dqn_step, self.irl_step))

    def load_if_exists(self, sess, path):
        if os.path.exists(path + '.meta'):
            print('Model already exists, loading: {0}'.format(path))
            self.load(sess, path)
            return True
        return False

    def set_writer(self, writer: tf.summary.FileWriter):
        pass
