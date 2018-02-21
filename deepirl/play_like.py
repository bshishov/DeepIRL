import argparse
import time
import tensorflow as tf
import numpy as np

from deepirl.utils.config import instantiate
from deepirl.models.base import RQModelBase
from deepirl.environments.base import Environment
import deepirl.utils.vizualization as v


def explain(env: Environment, sess: tf.Session, model: RQModelBase):
    wnd = v.Window(120, 220)
    state_drawer = v.ImageDrawer(v.Rect(10, 10, 100, 200))

    wnd.add_drawer(state_drawer)

    state = env.reset()
    while True:
        r, u, q, p = model.predict_r_u_q_p(sess, [state])
        if np.random.rand() < 0.1:
            action = np.random.choice(env.num_actions)
        else:
            action = np.random.choice(env.num_actions, p=p[0])
            #action = np.argmax(q[0])
        state = env.do_action(action)
        state_drawer.img = state

        if env.is_terminal():
            state = env.reset()

        wnd.draw()


def main(arguments):
    env = instantiate(arguments.env)

    with tf.device(arguments.device):
        model = instantiate(arguments.model, input_shape=env.state_shape, outputs=env.num_actions)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            model.load_if_exists(sess, arguments.model_path)

            explain(env, sess, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Configurations
    parser.add_argument("--env", type=str, help="Path to environment configuration JSON file")
    parser.add_argument("--model", type=str, help="Path to model configuration JSON file")

    # Meta parameters
    parser.add_argument("--model_path", type=str, help="Path to save model to", default='/netscratch/shishov/eye_irl')
    parser.add_argument("--device", type=str, help="Device to use", default='/device:GPU:0')

    main(parser.parse_args())
