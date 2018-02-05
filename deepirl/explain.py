import argparse
import time
import tensorflow as tf
import numpy as np

from deepirl.utils.replay import StateActionReplay
from deepirl.utils.config import instantiate
from deepirl.models.dqn import Model
from deepirl.environments.base import Environment
import deepirl.utils.vizualization as v


def explain(env: Environment, sess: tf.Session, model: Model, replay: StateActionReplay, frame_rate: float = 30.0):
    wnd = v.Window(900, 700)
    w = 200
    h = 320
    padding = 20

    state1_drawer = v.ImgPlotDrawer(v.Rect(padding, padding, w, h),
                                    caption="State channel:0")
    state2_drawer = v.ImgPlotDrawer(v.Rect(state1_drawer.bounds.right + padding, padding, w, h),
                                    caption="State channel:1", color_map=v.ColorMap.HOT)
    expert_drawer = v.ImgPlotDrawer(v.Rect(state2_drawer.bounds.right + w + padding * 2, padding, w, h),
                                    caption="Expert behaviour", color_map=v.ColorMap.HOT)

    r_drawer = v.ImgPlotDrawer(v.Rect(padding, state1_drawer.bounds.bottom + padding, w, h),
                               caption="Reward R(s,a)", color_map=v.ColorMap.HOT)
    u_drawer = v.ImgPlotDrawer(v.Rect(r_drawer.bounds.right + padding, state1_drawer.bounds.bottom + padding, w, h),
                               caption="Expected return U(s,a)", color_map=v.ColorMap.HOT)
    q_drawer = v.ImgPlotDrawer(v.Rect(u_drawer.bounds.right + padding, state1_drawer.bounds.bottom + padding, w, h),
                               caption="Quality Q(s,a)=R(s,a)+U(s,a)", color_map=v.ColorMap.HOT)
    p_drawer = v.ImgPlotDrawer(v.Rect(q_drawer.bounds.right + padding, state1_drawer.bounds.bottom + padding, w, h),
                               caption="Policy p(s,a)=softmax(Q(s,a))", color_map=v.ColorMap.HOT)

    wnd.add_drawer(state1_drawer)
    wnd.add_drawer(state2_drawer)
    wnd.add_drawer(expert_drawer)
    wnd.add_drawer(r_drawer)
    wnd.add_drawer(u_drawer)
    wnd.add_drawer(q_drawer)
    wnd.add_drawer(p_drawer)

    while True:
        for state, action in replay.iterate():
            state1_drawer.set_value(state[..., 0])
            state2_drawer.set_value(state[..., 1])

            r, u, q, p = model.predict(sess, [state])

            expert = np.zeros_like(r[0])
            expert[action] = 1.0
            expert_drawer.set_value(expert.reshape(state.shape[:2]))

            r_drawer.set_value(r[0].reshape(state.shape[:2]))
            u_drawer.set_value(u[0].reshape(state.shape[:2]))
            q_drawer.set_value(q[0].reshape(state.shape[:2]))
            p_drawer.set_value(p[0].reshape(state.shape[:2]))

            wnd.draw()
            time.sleep(1.0 / frame_rate)


def main(arguments):
    env = instantiate(arguments.env)

    # Load expert trajectories
    expert_replay = StateActionReplay(100000, env.state_shape)
    expert_replay.load(arguments.expert_replay)

    with tf.device(arguments.device):
        model = instantiate(arguments.model, input_shape=env.state_shape, outputs=env.num_actions)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            model.load_if_exists(sess, arguments.model_path)

            explain(env, sess, model, expert_replay)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Configurations
    parser.add_argument("--env", type=str, help="Path to environment configuration JSON file")
    parser.add_argument("--model", type=str, help="Path to model configuration JSON file")

    # Expert data
    parser.add_argument("--expert_replay", type=str, help="Path to expert replay")

    # Meta parameters
    parser.add_argument("--model_path", type=str, help="Path to save model to", default='/netscratch/shishov/eye_irl')
    parser.add_argument("--device", type=str, help="Device to use", default='/device:GPU:0')

    main(parser.parse_args())
