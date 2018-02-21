import numpy as np

from deepirl.environments.base import Environment
from deepirl.utils.math import gaussian_2d, append_2d_array_shifted
from deepirl.utils.replay import GenericMemory, StateActionReplay


class EyeTrackingReplay(GenericMemory):
    def __init__(self,
                 capacity=11000,
                 width: int = 64,
                 height: int = 64,
                 channels=1,
                 dtype=np.float16):
        super(self.__class__, self).__init__(capacity=capacity, definitions=[
            ('frame', dtype, (height, width, channels)),
            ('x', dtype, ()),
            ('y', dtype, ())
        ])


class EyetrackingEnv(Environment):
    frame_dtype = np.float16

    def __init__(self,
                 path_to_frames: str,
                 width: int = 64,
                 height: int = 64,
                 actions_height: int = 32,
                 actions_width: int = 32,
                 sight_sigma: float = 0.05,
                 sight_per_fixation: float = 0.05,
                 forget_rate: float = 0.999,
                 ):
        self.replay = EyeTrackingReplay(width=width, height=height)
        self.replay.load(path_to_frames)

        # TODO: use more efficient frame buffer, don't load the whole frameset in the memory
        self._frames = self.replay.get_col('frame')
        self._height, self._width = self._frames.shape[1], self._frames.shape[2]
        self._actions_height, self._actions_width = actions_height, actions_width
        super(EyetrackingEnv, self).__init__((self._height, self._width, 2), self._actions_height * self._actions_width)
        self._frame = 0
        self._state = np.zeros(self.state_shape, dtype=self.frame_dtype)
        self._state[..., 0] = self._get_frame()

        self._centered_sight_map = gaussian_2d(self._width,
                                               self._height,
                                               x=0.5, y=0.5,
                                               sigma=sight_sigma) * sight_per_fixation
        self._centered_sight_map = np.ndarray.astype(self._centered_sight_map, dtype=self.frame_dtype)
        self._forget_rate = forget_rate

    def _get_frame(self, di: int = 0):
        return self._frames[self._frame + di].astype(self.frame_dtype) / 255.0

    def reset(self):
        self._frame = 0
        self._state[..., 0] = self._get_frame()
        self._state[..., 1] = 0.0
        return self._state.astype(dtype=np.float32)

    def is_terminal(self) -> bool:
        return self._frame >= (len(self.replay) - 1)

    def do_action(self, action: int, update_state=True):
        if update_state:
            new_state = self._state_transition(self._state, action).copy()
            self._frame += 1
            return new_state
        else:
            return self._state_transition(self._state.copy(), action)

    def action_from_x_y(self, x: float, y: float):
        return int(y * (self._actions_height - 1)) * self._actions_width + int(x * (self._actions_width - 1))

    def _state_transition(self, state, action):
        # If it is terminal state the do nothing
        if self._frame >= len(self._frames) - 1:
            return state

        # Action space to state space conversion
        # Action space to 2d {x, y} [0, 1]
        x = float(action % self._actions_width) / (self._actions_width - 1)
        y = float(action // self._actions_height) / (self._actions_height - 1)

        # Action space {x, y} [0, 1]  -> state space x [0, w], y [0, h]
        x = int(x * self._width)
        y = int(y * self._height)

        next_frame = self._get_frame(+1)
        diff = np.clip(np.mean(np.absolute(next_frame - state[..., 0])) / 255.0, 0., 1.)
        diff_decay = 1 - 1 / np.exp(4 * (1 - diff))
        state[..., 0] = next_frame

        if diff > 0.1:
            state[..., 1] = 0.0

        state[..., 1] = append_2d_array_shifted(state[..., 1],
                                                self._centered_sight_map,
                                                x - self._width // 2,
                                                y - self._height // 2) * self._forget_rate
        return state


def get_part(src: np.ndarray, x: float, y: float, w: int, h: int):
    src_h, src_w = src.shape[:2]
    assert w <= src_w and h <= src_h

    w_ratio = 0.5 * float(w) / src_w
    h_ratio = 0.5 * float(h) / src_h

    x = np.clip(x, w_ratio, 1.0 - w_ratio)
    y = np.clip(y, h_ratio, 1.0 - h_ratio)
    left = int(src_w * (x - w_ratio))
    right = left + w
    top = int(src_h * (y - h_ratio))
    bottom = top + h
    return src[top:bottom, left:right]


def frames_from_video(path: str,
                      gaze_points: dict,
                      width: int = 128,
                      height: int = 128,
                      output_path='',
                      draw: bool = True):
    import cv2
    import deepirl.utils.vizualization as v

    # LODS:
    #   target     src at x,y location
    #   0: 32x32     32 x 32
    #   2: 32x32     64 x 64
    #   4: 32x32    128 x 128
    #   8: 32x32    256 x 256
    #  -1: 32x32    original.width x original.height
    # 512x512 is not used since it is not very different from original
    # So we will have 5 channels
    lod_scales = [1, 2, 4, 8, -1]
    frame = np.zeros((height, width, len(lod_scales)), dtype=np.uint8)

    memory = EyeTrackingReplay(height=height, width=width, channels=len(lod_scales))
    cap = cv2.VideoCapture(path)

    # Check if camera opened successfully
    if not cap.isOpened():
        raise RuntimeError("Error opening video stream or file: {0}".format(path))

    drawer_width = 256
    wnd = v.Window(len(lod_scales) * drawer_width, 490)
    lod_drawers = []
    for i in range(len(lod_scales)):
        drawer = v.ImgPlotDrawer(v.Rect(i * drawer_width, 0, drawer_width, 480))
        wnd.add_drawer(drawer)
        lod_drawers.append(drawer)
    stats_drawer = v.StringDrawer(10, 440)
    wnd.add_drawer(stats_drawer)
    frame_i = 0

    left, top, right, bottom = 717, 226, 1423, 923
    # roi width is 706
    # roi height is 697
    roi = v.Rect(left, top, right - left, bottom - top)

    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame_raw = cap.read()

        if not ret:
            break

        if frame_i in gaze_points:
            x, y = gaze_points[frame_i]
            x = float(x - roi.left) / roi.width
            y = float(y - roi.top) / roi.height

            skip = False
            # Skip if x is out of area
            if x < 0.0 or x > 1.0:
                skip = True

            # Skip if y is out of area
            if y < 0.0 or y > 1.0:
                skip = True

            if not skip:
                cropped = frame_raw[roi.top:roi.bottom, roi.left:roi.right]
                cropped_bw = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

                for i, scale in enumerate(lod_scales):
                    if scale < 0:
                        part = cropped_bw
                    else:
                        part = get_part(cropped_bw, x, y, width * scale, height * scale)
                    if scale == 1:
                        frame[..., i] = part
                    else:
                        frame[..., i] = cv2.resize(part, (width, height), interpolation=cv2.INTER_AREA)

                memory.append(frame / 255.0, 2.0 * (x - 0.5), 2.0 * (y - 0.5))

                if draw:
                    for i, drawer in enumerate(lod_drawers):
                        drawer.set_value(frame[..., i])
                    stats_drawer.text = 'Frame {0}\nGaze X {1}\nGaze Y {2}'.format(frame_i, x, y)
                    wnd.draw()
                else:
                    print('Frame {0}\tGaze X {1:.3f}\tGaze Y {2:.3f}'.format(frame_i, x, y))
        frame_i += 1
    memory.save(output_path)


def load_from_csv(path):
    gaze_points = np.genfromtxt(path, delimiter=',', dtype=None)
    points = {}
    for x, y, frame, t, time in gaze_points:
        if x == 0 or y == 0:
            continue
        points[frame] = (x, y)
    return points


def generate_expert_trajectories(path_to_frames: str, output: str):
    env = EyetrackingEnv(path_to_frames)
    replay = env.replay

    trajectories = StateActionReplay(replay.filled, env.state_shape, state_dtype=env.frame_dtype)

    state = env.reset()
    while True:
        x = replay._memory[env._frame]['x']
        y = replay._memory[env._frame]['y']
        action = env.action_from_x_y(x, y)
        trajectories.append(state, action)
        state = env.do_action(action)

        if action > env._actions_width * env._actions_height:
            print('Frame: {0}\tx: {1:.3f}\ty: {2:.3f}\taction: {3}'.format(env._frame, x, y, action))

        if env.is_terminal():
            break

    print('Saving replay to: {0}'.format(output))
    trajectories.save(output)
    print('Saved')


def run_random(path):
    import deepirl.utils.vizualization as v
    env = EyetrackingEnv(path, actions_width=16, actions_height=16)

    wnd = v.Window(800, 600)
    state1_drawer = v.ImgPlotDrawer(v.Rect(10, 10, 200, 400))
    state2_drawer = v.ImgPlotDrawer(v.Rect(220, 10, 200, 400))
    wnd.add_drawer(state1_drawer)
    wnd.add_drawer(state2_drawer)

    state = env.reset()

    while not env.is_terminal():
        state1_drawer.set_value(state[..., 0])
        state2_drawer.set_value(state[..., 1])

        x = env.replay._memory[env._frame]['x']
        y = env.replay._memory[env._frame]['y']
        action = env.action_from_x_y(x, y)

        # action = np.random.choice(env.num_actions)
        # action = env.num_actions // 2
        state = env.do_action(action)

        wnd.draw()


if __name__ == '__main__':
    src_path = 'D:\\deepirl\\replay\\xray_expert_64x64.npz'
    replay_path = 'D:\\deepirl\\replay\\xray_expert_64x64_actions.npz'

    # run_random(src_path)
    #generate_expert_trajectories(src_path, replay_path)

    frames_from_video('F:\\DSC_0029.MOV',
                      load_from_csv('D:\\Doctors\\expert_gaze_points.csv'),
                      width=32,
                      height=32,
                      output_path='D:\\deepirl\\replay\\xray_expert_32x32x5.npz',
                      draw=False)
