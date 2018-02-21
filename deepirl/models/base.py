import tensorflow as tf
import os


class ModelBase(object):
    def __init__(self):
        self._summaries_writer = None
        self._saver = None

    def save(self, sess: tf.Session, path: str):
        if self._saver is None:
            with tf.variable_scope('Saver'):
                self._saver = tf.train.Saver()

        full_path = self._saver.save(sess, path)
        print('Model saved to {0}'.format(full_path))

    def load(self, sess: tf.Session, path: str):
        if self._saver is None:
            with tf.variable_scope('Saver'):
                self._saver = tf.train.Saver()

        self._saver.restore(sess, path)
        print('Model restored: {0}'.format(path))

    def load_if_exists(self, sess: tf.Session, path: str):
        if os.path.exists(path + '.meta'):
            print('Model already exists, loading: {0}'.format(path))
            self.load(sess, path)
            return True
        return False

    def set_writer(self, summaries_writer: tf.summary.FileWriter):
        self._summaries_writer = summaries_writer


class RQModelBase(ModelBase):
    def __init__(self, double: bool):
        super(RQModelBase, self).__init__()
        self.double = double
        self.dqn_step = 1
        self.irl_step = 1
        self._dqn_step_tf = tf.Variable(self.dqn_step, name='dqn_step', trainable=False)
        self._irl_step_tf = tf.Variable(self.irl_step, name='irl_step', trainable=False)

    def train_r(self, sess: tf.Session, states, expert_actions, *args, **kwargs):
        raise NotImplementedError

    def train_u(self, sess: tf.Session, states, target_u, *args, **kwargs):
        raise NotImplementedError

    def predict_r_u_q_p(self, sess: tf.Session, states, *args, **kwargs):
        raise NotImplementedError

    def update_target(self, sess: tf.Session):
        if self.double:
            pass
        else:
            raise RuntimeError("This model is not double")

    def load(self, sess: tf.Session, path: str):
        super(RQModelBase, self).load(sess, path)
        self.dqn_step, self.irl_step = sess.run([self._dqn_step_tf, self._irl_step_tf])
        print('DQN step: {0}, IRL step: {1}'.format(self.dqn_step, self.irl_step))

