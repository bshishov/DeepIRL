import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Dist(object):
    def __init__(self, batch_size):
        self.inputs = tf.placeholder(tf.float32, (batch_size, 2))

        dist_mean = tf.Variable(tf.zeros([2]), trainable=True, dtype=tf.float32)
        dist_log_std = tf.Variable(tf.zeros([2]), trainable=True, dtype=tf.float32)
        dist_std = tf.exp(dist_log_std)

        self.dist = tf.contrib.distributions.MultivariateNormalDiag(loc=dist_mean, scale_diag=dist_std)
        dist_mode = self.dist.mode()
        #self.loss = tf.losses.mean_squared_error(tf.reduce_mean(self.inputs, axis=0), dist_mode) - 0.1 * self.dist.entropy()
        #self.loss = tf.losses.mean_squared_error(self.inputs, self.dist.sample(batch_size)) - 0.1 * self.dist.entropy() + tf.reduce_mean(dist_std)
        self.loss = -self.dist.log_prob(self.inputs)

        opt = tf.train.AdamOptimizer(learning_rate=0.01)
        self.optimize = opt.minimize(self.loss)

    def train(self, sess: tf.Session, points: np.ndarray):
        loss, _ = sess.run([self.loss, self.optimize], feed_dict={
            self.inputs: points
        })
        return loss

    def plot(self, sess: tf.Session, w=128, h=128):
        indices = np.zeros((h * w, 2), dtype=np.float32)
        for j, x in enumerate(np.linspace(-5, 5, w)):
            for i, y in enumerate(np.linspace(-5, 5, h)):
                indices[i * w + j, 0] = x
                indices[i * w + j, 1] = y

        probabilities = sess.run(self.dist.prob(indices))
        prob_image = np.reshape(probabilities, newshape=(h, w))
        return prob_image

    def stats(self, sess: tf.Session):
        mu, std = sess.run([self.dist.mean(), self.dist.stddev()])
        print('mu: {}'.format(mu))
        print('stddev: {}'.format(std))

    def sample(self, sess: tf.Session, samples):
        return sess.run(self.dist.sample(samples))


def main():
    import deepirl.utils.vizualization as v
    wnd = v.Window(512, 512)
    image_drawer = v.ImageDrawer(v.Rect(0, 0, 512, 512))
    wnd.add_drawer(image_drawer)

    batch_size = 64
    d = Dist(batch_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        w = 128
        h = 128
        d.stats(sess)
        #d.plot(sess, w, h)

        for _ in range(10):
            for i in range(1000):
                print('Training: {}'.format(i))
                points = np.random.multivariate_normal(mean=[-0.5, 0.5], cov=[[1.7, 0.0], [0.0, 0.2]], size=batch_size)
                d.train(sess, points)
                if i % 100 == 0:
                    image_drawer.set_value(d.plot(sess, w, h))
                    wnd.draw()
            d.stats(sess)
            #d.plot(sess, w, h)

        d.stats(sess)
        d.plot(sess, w, h)


if __name__ == '__main__':
    main()