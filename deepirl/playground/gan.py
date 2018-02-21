import tensorflow as tf
import numpy as np
from scipy.misc import imsave


def get_pos(x: int, y: int, w: int, h: int):
    x = (float(x) - w // 2) / w
    y = (float(y) - h // 2) / h
    return [x, y, np.sqrt(np.square(x) + np.square(y))]


def build_discriminator(inputs: tf.Tensor, reuse=False, depths=[128, 256, 512, 3], training=False):
    outputs = inputs
    with tf.variable_scope('D', reuse=reuse):
        with tf.variable_scope('conv1'):
            outputs = tf.layers.conv2d(outputs, depths[0], [5, 5], strides=(2, 2), padding='SAME')
            outputs = tf.nn.leaky_relu(tf.layers.batch_normalization(outputs, training=training))
        with tf.variable_scope('conv2'):
            outputs = tf.layers.conv2d(outputs, depths[1], [5, 5], strides=(2, 2), padding='SAME')
            outputs = tf.nn.leaky_relu(tf.layers.batch_normalization(outputs, training=training))
        with tf.variable_scope('conv3'):
            outputs = tf.layers.conv2d(outputs, depths[2], [5, 5], strides=(2, 2), padding='SAME')
            outputs = tf.nn.leaky_relu(tf.layers.batch_normalization(outputs, training=training))
        with tf.variable_scope('conv4'):
            outputs = tf.layers.conv2d(outputs, depths[3], [5, 5], strides=(2, 2), padding='SAME')
            outputs = tf.nn.leaky_relu(tf.layers.batch_normalization(outputs, training=training))
        with tf.variable_scope('classify'):
            flatten = tf.layers.flatten(outputs)
            out = tf.layers.dense(flatten, 1)
            out = tf.reshape(out, [-1])
            #logits = tf.nn.sigmoid(out)
            logits = out
            return logits


def build_generator(inputs: tf.Tensor, reuse=False, depth=12):
    with tf.variable_scope('G', reuse=reuse):
        x = inputs
        for _ in range(depth):
            x = tf.contrib.layers.fully_connected(x, 32, activation_fn=tf.nn.tanh)
        output = tf.contrib.layers.fully_connected(x, 1, activation_fn=tf.nn.sigmoid)
        output = tf.reshape(output, [-1])
        return output


class GAN(object):
    def __init__(self, w: int, h: int, z_size: int=32, batch_size=64):
        self.w, self.h = w, h
        self.z_size = z_size

        with tf.variable_scope('Inputs'):
            with tf.variable_scope('G'):
                self.z = tf.placeholder(tf.float32, [None, z_size])
                self.pos = tf.placeholder(tf.float32, [None, 3])
                generator_input = tf.concat([self.z, self.pos], axis=1)
            with tf.variable_scope('D'):
                self.samples = tf.placeholder(tf.float32, [batch_size, h, w, 1])

        self.coords = self.create_coords(w, h)
        self.generated = build_generator(generator_input)
        generated_image = tf.reshape(self.generated, [batch_size, h, w, 1])
        d_samples_logits = build_discriminator(self.samples, reuse=False)
        d_generated_logits = build_discriminator(generated_image, reuse=True)

        with tf.variable_scope('Losses'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits
            true = tf.ones([batch_size], dtype=tf.float32) * 0.9
            d_sample_loss = tf.reduce_mean(cross_entropy(labels=true, logits=d_samples_logits))
            d_generated_loss = tf.reduce_mean(cross_entropy(labels=tf.zeros([batch_size], dtype=tf.int32),
                                                            logits=d_generated_logits))
            self.d_loss = d_sample_loss + d_generated_loss
            self.g_loss = tf.reduce_mean(cross_entropy(labels=tf.ones([batch_size], dtype=tf.int32),
                                                       logits=d_generated_logits))

        with tf.variable_scope('Optimization'):
            lr = 0.0005
            beta1 = 0.5
            g_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
            d_opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
            g_opt_op = g_opt.minimize(self.g_loss, var_list=tf.trainable_variables('G'))
            d_opt_op = d_opt.minimize(self.d_loss, var_list=tf.trainable_variables('D'))
            with tf.control_dependencies([g_opt_op, d_opt_op]):
                self.train_op = tf.no_op(name='train')

    def train(self, sess, input_samples):
        batch_size = len(input_samples)
        pos_batch = np.tile(self.coords, (batch_size, 1))
        z_batch = np.tile(np.random.standard_normal(self.z_size), (batch_size * self.w * self.h, 1))
        g_loss, d_loss, _ = sess.run([self.g_loss, self.d_loss, self.train_op], feed_dict={
            self.z: z_batch,
            self.pos: pos_batch,
            self.samples: input_samples
        })
        print(g_loss, d_loss)
        return g_loss, d_loss

    def generate_sample(self, sess: tf.Session, w=None, h=None):
        if w is None:
            w = self.w
        if h is None:
            h = self.h
        z_batch, pos_batch = self.create_input_for_generator(w, h)
        values = sess.run(self.generated, feed_dict={
            self.z: z_batch,
            self.pos: pos_batch
        })
        image = np.reshape(values, (h, w))
        return image

    def create_coords(self, w, h):
        pos_batch = np.zeros((w * h, 3))
        for i in range(h * w):
            x = float(i % w - w // 2) / w
            y = float(i // w - h // 2) / h
            pos_batch[i, 0] = x
            pos_batch[i, 1] = y
            pos_batch[i, 2] = np.sqrt(np.square(x) + np.square(y))
        return pos_batch

    def create_input_for_generator(self, w, h):
        z_batch = np.zeros((w * h, self.z_size), np.float32)
        z_batch[:] = np.random.standard_normal(self.z_size)
        return z_batch, self.create_coords(w, h)



class Generator(object):
    def __init__(self, z_num=10):
        self.z = tf.placeholder(tf.float32, [None, z_num])
        self.pos = tf.placeholder(tf.float32, [None, 3])

        inputs = tf.concat([self.z, self.pos], axis=1)

        x = inputs
        for i in range(12):
            x = tf.contrib.layers.fully_connected(x, 32, activation_fn=tf.nn.tanh,
                                                  weights_initializer=tf.initializers.random_normal(0, 0.6))

        self.color = tf.contrib.layers.fully_connected(x, 1, activation_fn=tf.nn.sigmoid,
                                                       weights_initializer=tf.initializers.random_normal(0, 0.6))

    def sample(self, sess: tf.Session, zs, positions):
        return sess.run(self.color, feed_dict={
            self.z: zs,
            self.pos: positions
        })


def main():
    import deepirl.utils.vizualization as v

    w = 1080
    h = 1920

    z_size = 10
    batch_size = 512
    image = np.zeros((h, w), dtype=np.float32)
    z = np.random.standard_normal((z_size,))
    z_batch = np.zeros((batch_size, z_size), np.float32)
    z_batch[:] = z
    pos_batch = np.zeros((batch_size, 3), np.float32)
    x_batch = np.zeros((batch_size,), np.uint16)
    y_batch = np.zeros((batch_size,), np.uint16)

    wnd = v.Window(w // 2, h // 2)
    image_drawer = v.ImageDrawer(v.Rect(0, 0, wnd.width, wnd.height))
    wnd.add_drawer(image_drawer)

    with tf.Session() as sess:
        g = Generator(z_num=z_size)
        sess.run(tf.global_variables_initializer())

        idx = 0
        for i in range(h):
            for j in range(w):
                pos_batch[idx] = get_pos(j, i, w, h)
                x_batch[idx] = i
                y_batch[idx] = j

                idx += 1
                if idx >= batch_size:
                    values = g.sample(sess, z_batch, pos_batch)
                    for ii, jj, val in zip(x_batch, y_batch, values):
                        image[ii, jj] = val
                    image_drawer.img = image
                    wnd.draw()
                    idx = 0

    imsave('D:/bgr.png', image)
    while True:
        wnd.draw()


def main2():
    import deepirl.utils.vizualization as v
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array

    batch_size = 64
    wnd = v.Window(400, 400)
    image_drawer = v.ImageDrawer(v.Rect(0, 0, wnd.width, wnd.height))
    wnd.add_drawer(image_drawer)

    with tf.Session() as sess:
        gan = GAN(28, 28, z_size=32, batch_size=batch_size)
        sess.run(tf.global_variables_initializer())
        i = 0
        while True:
            indices = np.random.choice(len(train_data), batch_size, replace=False)
            samples = np.reshape(train_data[indices], (-1, 28, 28, 1))
            gan.train(sess, samples)
            i += 1
            if i % 10 == 0:
                image_drawer.img = gan.generate_sample(sess, 64, 64)
                wnd.draw()


if __name__ == '__main__':
    main()
