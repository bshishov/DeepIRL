import tensorflow as tf
import numpy as np
import argparse

from deepirl.models.base import ModelBase
from deepirl.environments.eyetracking import EyeTrackingReplay
from deepirl.models.policy import PolicyValueHead

BATCH_SIZE = 256


class Model(ModelBase):
    def __init__(self,
                 input_shape: tuple,
                 output_dim: int,
                 residual_layers=2,
                 conv_filters=64,
                 learning_rate=0.001):
        super().__init__()
        self.inputs = tf.placeholder(tf.float32, shape=(None, ) + input_shape)
        self.target_outputs = tf.placeholder(tf.float32, shape=(None, output_dim))

        with tf.variable_scope('Embedding'):
            conv1 = tf.layers.conv2d(self.inputs, conv_filters, [3, 3], padding='same')
            batch_norm = tf.layers.batch_normalization(conv1)
            relu1 = tf.nn.relu(batch_norm)

        x = relu1
        for i in range(residual_layers):
            with tf.variable_scope('Residual{0}'.format(i)):
                conv1 = tf.layers.conv2d(x, conv_filters, [3, 3], name='Conv1', padding='same')
                batch_norm1 = tf.layers.batch_normalization(conv1, name='BatchNorm1')
                relu1 = tf.nn.relu(batch_norm1, name='Relu1')
                conv2 = tf.layers.conv2d(relu1, conv_filters, [3, 3], name='Conv2', padding='same')
                batch_norm2 = tf.layers.batch_normalization(conv2, name='BatchNorm2')
                relu2 = tf.nn.relu(batch_norm2 + x, name='Relu2')
                x = relu2
        embedding = tf.layers.flatten(x)

        self.policy = PolicyValueHead(embedding, output_dim)

        with tf.variable_scope('Losses'):
            self.entropy_loss = -0.01 * tf.reduce_mean(self.policy.dist.entropy())
            self.prob_loss = -tf.reduce_mean(self.policy.dist.log_prob(self.target_outputs))
            self.mode_loss_deterministic = tf.losses.mean_squared_error(self.target_outputs, self.policy.dist.mode())
            '''
            self.mode_loss_stochastic = tf.losses.mean_squared_error(self.target_outputs,
                                                                     self.policy.dist.sample(BATCH_SIZE))
            '''

        with tf.variable_scope('Optimization'):
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.optimize_prob = opt.minimize(self.prob_loss)
            self.optimize_mode_deterministic = opt.minimize(self.mode_loss_deterministic)
            '''            
            self.optimize_mode_stochastic = opt.minimize(self.mode_loss_stochastic)
            '''

    def train(self, sess: tf.Session, states: np.ndarray, actions: np.ndarray):
        opt_op = self.optimize_mode_deterministic
        loss_op = self.mode_loss_deterministic

        #opt_op = self.optimize_prob
        #loss_op = self.prob_loss

        loss, _, mode = sess.run([loss_op, opt_op, self.policy.dist.mode()], feed_dict={
            self.inputs: states,
            self.target_outputs: actions
        })
        return loss

    def get_prob(self, sess: tf.Session, state: np.ndarray, indices: np.ndarray):
        probabilities, mode = sess.run([self.policy.dist.prob(indices), self.policy.dist.mode()],
                                       feed_dict={ self.inputs: [state] })
        return probabilities, mode

    def stats(self, sess: tf.Session, state):
        mu, std, mode = sess.run([self.policy.dist.mean(), self.policy.dist.stddev(), self.policy.dist.mode()], feed_dict={
            self.inputs: [state]
        })
        print('Mu', mu)
        print('Std', std)
        return mu, std, mode


def train(sess: tf.Session, replay: EyeTrackingReplay, model: Model, epochs=100):
    for e in range(epochs):
        frames, x, y = replay.sample(BATCH_SIZE)
        actions = np.vstack((x, y)).T
        loss = model.train(sess, frames, actions)
        print('{0}/{1} Loss: {2:.3f}'.format(e, epochs, loss))


def play(sess: tf.Session, replay: EyeTrackingReplay, model: Model, channels=5):
    import deepirl.utils.vizualization as v
    import cv2

    drawer_size = 128
    wnd = v.Window(drawer_size * channels, drawer_size * 3)
    prob_drawer = v.ImageDrawer(v.Rect(0, drawer_size, drawer_size * 2, drawer_size * 2))
    ex_prob_drawer = v.ImageDrawer(v.Rect(drawer_size * 2, drawer_size, drawer_size * 2, drawer_size * 2))
    stats_drawer = v.StringDrawer(drawer_size * 4 + 10, drawer_size + 10)
    wnd.add_drawer(prob_drawer)
    wnd.add_drawer(ex_prob_drawer)
    wnd.add_drawer(stats_drawer)

    output_path = 'D:\\deepirl\\xray_bc.avi'
    frame_rate = 20
    frames = 0
    video_frames_to_write = 30 * frame_rate  # 30 secs
    writer = None
    if output_path:
        print('Recording video: {0}'.format(output_path))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (wnd.width, wnd.height))

    w, h = 64, 64
    ex_policy = np.zeros((h, w), dtype=np.float32)
    indices = np.zeros((h * w, 2), dtype=np.float32)
    for j, x in enumerate(np.linspace(-1, 1, w)):
        for i, y in enumerate(np.linspace(-1, 1, h)):
            indices[i * w + j, 0] = x
            indices[i * w + j, 1] = y

    drawers = []
    for i in range(channels):
        drawer = v.ImageDrawer(v.Rect(i * drawer_size, 0, drawer_size, drawer_size))
        drawers.append(drawer)
        wnd.add_drawer(drawer)

    while True:
        #frames, xs, ys = replay.sample(BATCH_SIZE)
        #for frame, x, y in zip(frames, xs, ys):
        for res in replay.iterate():
            frame = res['frame']
            x = res['x']
            y = res['y']
            for i, drawer in enumerate(drawers):
                drawer.set_value(frame[..., i])

            yy = np.clip((h - 1) * (y + 1.0) * 0.5, 0.0, h - 1)
            xx = np.clip((w - 1) * (x + 1.0) * 0.5, 0.0, w - 1)
            ex_policy[int(yy), int(xx)] += 1.0
            ex_policy *= 0.95
            probs, mode = model.get_prob(sess, frame, indices)
            prob_drawer.set_value(np.reshape(probs, newshape=(h, w)))
            ex_prob_drawer.set_value(ex_policy)

            pred_x = mode[0][0]
            pred_y = mode[0][1]

            mse = np.sqrt(np.square(pred_x - x) + np.square(pred_y - y))

            stats_drawer.text = 'Expert:\n X: {0:.3f}\n Y: {1:.3f}\n\nPredicted:\n X: {2:.3f}\n Y: {3:.3f}\n\nMSE: {4:.3f}'.format(
                x, y, pred_x, pred_y, mse
            )
            wnd.draw()

            frames += 1

            if writer is not None:
                if frames < video_frames_to_write:
                    writer.write(wnd.screen)
                if frames == video_frames_to_write:
                    print('Saving video to: {0}'.format(output_path))
                    writer.release()


def main(arguments):
    channels = 5

    replay = EyeTrackingReplay(width=32, height=32, channels=channels)
    replay.load(arguments.replay)

    with tf.device(arguments.device):
        model = Model((32, 32, channels), 2,
                      conv_filters=32,
                      residual_layers=1,
                      learning_rate=arguments.lr)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            model.load_if_exists(sess, arguments.model_path)

            if arguments.train > 0:
                train(sess, replay, model, arguments.train)
                model.save(sess, arguments.model_path)

            if arguments.play:
                play(sess, replay, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, help="Run training N times", default=10000)
    parser.add_argument("--replay", type=str, help="Path to expert replay", default='/netscratch/shishov/deepirl/replay/xray_expert_32x32x5.npz')
    parser.add_argument("--model_path", type=str, help="Path to save model to", default='/netscratch/shishov/deepirl/models/bc_xray')
    parser.add_argument("--device", type=str, help="Device to use", default='/device:GPU:0')
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.001)
    parser.add_argument("--play", action='store_true')
    main(parser.parse_args())
