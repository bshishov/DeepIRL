import numpy as np
import os
import tensorflow as tf
import tensorflow.contrib.layers as layers


def one_hot(a, depth=1024):
    r = np.zeros((len(a), depth), dtype=np.float32)
    r[range(len(a)), a] = 1.0
    return r


def conv_residual_embedding(input_tensor: tuple, conv_filters: int, use_residual=True):
    # 32 x 32 x 2 -> 32 x 32 x conv_filters
    x = layers.conv2d(input_tensor, conv_filters, (3, 3), stride=1, padding='SAME', activation_fn=None)
    x = tf.contrib.layers.batch_norm(x)
    out1 = tf.nn.relu(x)

    # 32 x 32 x conv_filters -> 32 x 32 x conv_filters
    x = layers.conv2d(out1, conv_filters, (3, 3), stride=1, padding='SAME', activation_fn=None)
    x = tf.contrib.layers.batch_norm(x)
    x = tf.nn.relu(x)

    if use_residual:
        x = out1 + x  # residual connection
    return x


class DunModel(object):
    def __init__(self, input_shape, outputs,
                 conv_filters=16,
                 dueling=False,
                 use_dropout=True,
                 double=False,
                 residual=True,
                 irl_lr=1e-3,
                 dqn_lr=1e-3,
                 l2_scale=1e-5,
                 ):
        self.double = double
        self.dueling = dueling
        self.use_dropout = use_dropout
        self.dqn_step = 1
        self.irl_step = 1
        self.writer = None

        self._residual = residual
        self._conv_filters = conv_filters
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
            if self.double:
                self.u_target = self._build_dqn_head('Target')

                # Define transfer operation
                var_eval = [v for v in tf.trainable_variables('DQN/Eval')]
                var_target = [v for v in tf.trainable_variables('DQN/Target')]

                # Transfer Eval weights to Target
                self.dqn_transfer_op = [tf.assign(t, e) for t, e in zip(var_target, var_eval)]

        with tf.variable_scope('Policy'):
            self.q = tf.add(self.rewards, self.u)
            if self.double:
                with tf.variable_scope('Target'):
                    self.q_target = tf.add(self.rewards, self.u_target)
                    self.policy_target = tf.nn.softmax(self.q_target)

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

        with tf.variable_scope('Summaries'):
            with tf.variable_scope('Reward'):
                irl_summaries = []

                trainable_vars = [v for v in tf.trainable_variables('Reward')]
                trainable_weights = [v for v in trainable_vars if 'weights' in v.name]

                for v in trainable_weights:
                    irl_summaries.append(tf.summary.histogram(v.name, v))
                irl_summaries.append(tf.summary.scalar('IRL_Loss', self.irl_loss))

                self.irl_summaries = tf.summary.merge(irl_summaries, name='IRL_Summaries')

            with tf.variable_scope('DQN'):
                dqn_summaries = []

                trainable_vars = [v for v in tf.trainable_variables('DQN')]
                trainable_vars = [v for v in trainable_vars if 'Target' not in v.name]
                trainable_vars = [v for v in trainable_vars if 'weights' in v.name]

                for v in trainable_vars:
                    dqn_summaries.append(tf.summary.histogram(v.name, v))
                dqn_summaries.append(tf.summary.scalar('DQN_Loss', self.dqn_loss))

                self.dqn_summaries = tf.summary.merge(dqn_summaries, name='DQN_Summaries')

        with tf.variable_scope('Saver'):
            self.saver = tf.train.Saver()

    def _build_dqn_head(self, scope='DqnHead'):
        with tf.variable_scope(scope):
            embedding = conv_residual_embedding(self.states, self._conv_filters, use_residual=self._residual)

            x = layers.conv2d(embedding, 2, [1, 1], stride=1, activation_fn=None)
            x = layers.batch_norm(x)
            x = tf.nn.relu(x)
            x = layers.flatten(x)
            x = layers.fully_connected(x, self._outputs, activation_fn=tf.nn.relu)
            if self.use_dropout:
                x = layers.dropout(x, keep_prob=0.5, is_training=self._is_training)
            out = x

            with tf.variable_scope('Outputs'):
                if self.dueling:
                    # Dueling architecture, Dense layers with linear activation for V(s) value and A(s, a) advantage
                    advantage = layers.fully_connected(out, self._outputs, activation_fn=None, biases_initializer=None)
                    expected_value = layers.fully_connected(out, 1, activation_fn=None, biases_initializer=None)
                    u = expected_value + tf.subtract(advantage,
                                                     tf.reduce_mean(advantage, reduction_indices=1, keep_dims=True))
                else:
                    u = layers.fully_connected(out, self._outputs, activation_fn=None, biases_initializer=None)
            return u

    def _build_reward_head(self, scope='Reward'):
        with tf.variable_scope(scope):
            nn_base = conv_residual_embedding(self.states, self._conv_filters, use_residual=self._residual)

            # Reward head
            net = layers.conv2d(nn_base, 2, [1, 1], stride=1, activation_fn=None)
            net = layers.batch_norm(net)
            net = tf.nn.relu(net)
            net = layers.flatten(net)
            net = layers.fully_connected(net, self._outputs, activation_fn=tf.nn.relu)

            # Last layer is Sigmoid activated, so rewards for state are bounded by (0, +1)
            net = layers.fully_connected(net, self._outputs, activation_fn=tf.nn.sigmoid)

            # OR clipped ReLU could be used:
            # net = tf.clip_by_value(layers.fully_connected(net, outputs, activation_fn=tf.nn.relu), 0.0, MAX_R)

            return net

    def train_r(self, sess, states, expert_actions):
        feed_dict = {self.states: states, self.expert_policy: one_hot(expert_actions, self._outputs),
                     self._is_training: True}
        write_summaries = self.irl_step % 10 == 0
        if self.writer is not None and write_summaries:
            loss, _, summaries, self.irl_step = sess.run([self.irl_loss, self.irl_train_op,
                                                          self.irl_summaries, self._irl_step],
                                                         feed_dict=feed_dict)
            self.writer.add_summary(summaries, self.irl_step)
        else:
            loss, _, self.irl_step = sess.run([self.irl_loss, self.irl_train_op, self._irl_step], feed_dict=feed_dict)
        return loss

    def train_u(self, sess, states, target_u, average_episode_reward=None):
        feed_dict = {self.states: states, self.target_u: target_u, self._is_training: True}
        write_summaries = self.dqn_step % 1000 == 0
        if self.writer is not None and write_summaries:
            loss, _, summaries, self.dqn_step = sess.run([self.dqn_loss, self.dqn_train_op,
                                                          self.dqn_summaries, self._dqn_step],
                                                         feed_dict=feed_dict)

            self.writer.add_summary(summaries, self.dqn_step)
            if average_episode_reward is not None:
                summary = tf.Summary(value=[
                    tf.Summary.Value(tag="DQN/AverageEpisodeReward", simple_value=average_episode_reward),
                ])
                self.writer.add_summary(summary, self.dqn_step)
        else:
            loss, _, self.dqn_step = sess.run([self.dqn_loss, self.dqn_train_op, self._dqn_step], feed_dict=feed_dict)

        return loss

    def predict_r_u_q_p(self, sess, states, use_target_q=False):
        var_set = [self.rewards, self.u, self.q, self.policy]
        if use_target_q and self.double:
            var_set = [self.rewards, self.u_target, self.q_target, self.policy_target]
        rewards, u, q, policy = sess.run(var_set, feed_dict={self.states: states, self._is_training: False})
        return rewards, u, q, policy

    def predict_vars(self, sess, states, variables: list):
        return sess.run(variables, feed_dict={self.states: states, self._is_training: False})

    def update_target(self, sess):
        if self.double:
            sess.run(self.dqn_transfer_op)

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
        self.writer = writer
