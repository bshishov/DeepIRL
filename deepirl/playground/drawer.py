import tensorflow as tf
import numpy as np
import deepirl.utils.vizualization as v
import time

w = 512
h = 512
batch_size = 1024
noise_dim = 32
layers = 10
num_hidden = 64
stddev = 1.0
use_color = False
position_scale = 1.2
activation = tf.nn.tanh


def get_pos(x: int, y: int, w: int, h: int):
    x = position_scale * (float(x) - w // 2) / w
    y = position_scale * (float(y) - h // 2) / h
    return [x, y, np.sqrt(np.square(x) + np.square(y))]


inputs = tf.placeholder(tf.float32, (None, noise_dim + 3, ))
x = inputs
for i in range(layers):
    x = tf.layers.dense(x, num_hidden, activation=activation, kernel_initializer=tf.initializers.random_normal(stddev=stddev))
x = tf.exp(x)
if use_color:
    x = tf.layers.dense(x, 3, activation=tf.nn.sigmoid)
else:
    x = tf.reshape(tf.layers.dense(x, 1, activation=tf.nn.sigmoid), [-1])
out = x


if use_color:
    image = np.zeros((h, w, 3), dtype=np.float32)
else:
    image = np.zeros((h, w), dtype=np.float32)
wnd = v.Window(w, h)
image_drawer = v.ImageDrawer(v.Rect(0, 0, wnd.width, wnd.height))
wnd.add_drawer(image_drawer)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    while True:
        z = np.random.standard_normal((noise_dim,)) * 10.0
        z_batch = np.zeros((batch_size, noise_dim), np.float32)
        z_batch[:] = z
        pos_batch = np.zeros((batch_size, 3), np.float32)
        x_batch = np.zeros((batch_size,), np.uint16)
        y_batch = np.zeros((batch_size,), np.uint16)

        idx = 0
        for i in range(h):
            for j in range(w):
                pos_batch[idx] = get_pos(j, i, w, h)
                x_batch[idx] = i
                y_batch[idx] = j

                idx += 1
                if idx >= batch_size:
                    values = sess.run(out, feed_dict={inputs: np.concatenate((z_batch, pos_batch), axis=1)})
                    for ii, jj, val in zip(x_batch, y_batch, values):
                        image[ii, jj] = val
                    image_drawer.img = image
                    wnd.draw()
                    idx = 0

        wnd.draw()
        time.sleep(2.0)
