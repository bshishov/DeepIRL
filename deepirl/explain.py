import argparse
import time
import tensorflow as tf
import numpy as np
import cv2

from deepirl.utils.replay import StateActionReplay
from deepirl.utils.config import instantiate
from deepirl.models.base import RQModelBase
from deepirl.environments.base import Environment
import deepirl.utils.vizualization as v


def square(a):
    if len(a.shape) >= 2:
        return a

    return np.reshape(a, (-1, int(np.sqrt(a.shape[0]))))


def explain(env: Environment, sess: tf.Session, model: RQModelBase, replay: StateActionReplay,
            frame_rate: float = 20.0, output_path: str=''):
    wnd = v.Window(900, 700)

    writer = None
    frames = 0
    video_frames_to_write = 60 * frame_rate  # one minute
    if output_path:
        print('Recording video: {0}'.format(output_path))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (wnd.width, wnd.height))

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

    for state, action in replay.iterate():
        state1_drawer.set_value(state[..., 0])
        state2_drawer.set_value(state[..., 1])

        r, u, q, p = model.predict_r_u_q_p(sess, [state])

        expert = np.zeros_like(p[0])
        expert[action] = 1.0
        expert_drawer.set_value(square(expert))

        r_drawer.set_value(square(r[0]))
        u_drawer.set_value(square(u[0]))
        q_drawer.set_value(square(q[0]))
        p_drawer.set_value(square(p[0]))

        wnd.draw()
        frames += 1

        if writer is not None:
            if frames < video_frames_to_write:
                writer.write(wnd.screen)
            if frames == video_frames_to_write:
                print('Saving video to: {0}'.format(output_path))
                writer.release()

        # No need to sleep. Computation take too much time already
        #time.sleep(1.0 / frame_rate)

    if writer is not None:
        writer.release()


def main(arguments):
    env = instantiate(arguments.env)

    # Load expert trajectories
    expert_replay = StateActionReplay(12000, env.state_shape)
    expert_replay.load(arguments.expert_replay)

    with tf.device(arguments.device):
        model = instantiate(arguments.model, input_shape=env.state_shape, outputs=env.num_actions)

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            model.load_if_exists(sess, arguments.model_path)

            explain(env, sess, model, expert_replay, output_path=arguments.output)


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

    parser.add_argument("--output", type=str, default='', help='Output recorded video path')

    main(parser.parse_args())
