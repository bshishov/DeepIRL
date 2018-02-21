import numpy as np
import copy

from deepirl.environments.base import Environment

TETROMINO_S = np.array([
    [0, 1, 1],
    [1, 1, 0]
], dtype=np.uint8)

TETROMINO_Z = np.array([
    [1, 1, 0],
    [0, 1, 1]
], dtype=np.uint8)

TETROMINO_T = np.array([
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

TETROMINO_I = np.array([
    [1, 1, 1, 1],
], dtype=np.uint8)

TETROMINO_J = np.array([
    [0, 1],
    [0, 1],
    [1, 1],
], dtype=np.uint8)

TETROMINO_L = np.array([
    [1, 0],
    [1, 0],
    [1, 1],
], dtype=np.uint8)

TETROMINO_O = np.array([
    [1, 1],
    [1, 1],
], dtype=np.uint8)


FIGURES = [TETROMINO_S, TETROMINO_Z, TETROMINO_T, TETROMINO_I, TETROMINO_O, TETROMINO_J, TETROMINO_L]
FIGURES_W = np.array([1, 1, 1, 0.25, 0.75, 0.75, 0.75])
FIGURES_P = FIGURES_W / np.sum(FIGURES_W)


ACTION_NO_OP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_ROTATE = 3


def rotate(tetromino: np.ndarray, ccw=True):
    if ccw:
        return np.rot90(tetromino)
    else:
        return np.rot90(tetromino, axes=(0, 1))


class Tetris(Environment):
    def __init__(self, width: int = 10, height: int = 20):
        super(Tetris, self).__init__((height, width), 4)
        self.score = 0
        self._state = np.zeros(self.state_shape, dtype=np.uint8)
        self._figure = None
        self._top = 0
        self._left = 0
        self._rotations = 0
        self._movements = 0

        self._spawn()
        self._is_terminal = False

    def _place(self, figure, left: int, top: int):
        h, w = figure.shape

        if left < 0:
            return False
        if left + w > self.state_shape[1]:
            return False
        if top < 0:
            return False
        if top + h > self.state_shape[0]:
            return False

        region = self._state[top:top+h, left:left + w]

        if np.sum(region[figure > 0]) > 0:
            return False

        region += figure
        return True

    def _remove(self, figure, left: int, top: int):
        h, w = figure.shape
        region = self._state[top:top + h, left:left + w]
        region -= figure
        #figure_mask = np.ma.masked_array(region, mask=figure)
        #figure_mask[:] = 0.0

    def _spawn(self):
        self._figure = np.random.choice(FIGURES, p=FIGURES_P)
        if np.random.rand() < 0.5:
            self._figure = rotate(self._figure)

        self._left = self.state_shape[1] // 2 - self._figure.shape[1] // 2
        self._top = 2
        res = self._place(self._figure, self._left, self._top)
        if not res:
            self._is_terminal = True

    def reset(self):
        self._state = np.zeros(self.state_shape, dtype=np.float32)
        self._is_terminal = False
        self._spawn()
        self.score = 0
        self._rotations = 0
        self._movements = 0
        return self._state

    def is_terminal(self) -> bool:
        return self._is_terminal

    def move_x(self, dx=1):
        self._remove(self._figure, self._left, self._top)
        if self._place(self._figure, self._left + dx, self._top):
            self._left += dx
        else:
            self._place(self._figure, self._left, self._top)

    def move_down(self):
        self._remove(self._figure, self._left, self._top)
        if self._place(self._figure, self._left, self._top + 1):
            self._top += 1
            return True
        else:
            self._place(self._figure, self._left, self._top)
            return False

    def rotate(self):
        self._remove(self._figure, self._left, self._top)
        if self._place(rotate(self._figure), self._left, self._top):
            self._figure = rotate(self._figure)
            return True
        self._place(self._figure, self._left, self._top)
        return False

    def do_action(self, action: int, update_state=True):
        if not update_state:
            env_copy = copy.deepcopy(self)
            return env_copy.do_action(action, update_state=True)

        if action == ACTION_ROTATE:
            if self._rotations < 2:
                self.rotate()
                self._rotations += 1
            else:
                action = ACTION_NO_OP
        if action == ACTION_LEFT:
            if self._movements < 2:
                self.move_x(-1)
                self._movements += 1
            else:
                action = ACTION_NO_OP
        if action == ACTION_RIGHT:
            if self._movements < 2:
                self.move_x(+1)
                self._movements += 1
            else:
                action = ACTION_NO_OP
        if action == ACTION_NO_OP:
            self._rotations = 0
            self._movements = 0
            res = self.move_down()
            if not res:
                row_sums = np.sum(self._state, axis=1, dtype=np.uint8)
                for row, row_sum in enumerate(row_sums):
                    if row_sum == self._state.shape[1]:
                        self._state[1:row + 1] = self._state[:row]
                self._spawn()
        return self._state


if __name__ == '__main__':
    import keyboard
    import time
    import sys
    import deepirl.utils.vizualization as v
    from deepirl.utils.replay import StateActionReplay

    env = Tetris()
    state = env.reset()
    replay = StateActionReplay(100000, env.state_shape)
    episode = 0
    record_episodes = 1

    wnd = v.Window(220, 420)
    state_drawer = v.ImageDrawer(v.Rect(10, 10, 200, 400), color_map=v.ColorMap.HOT)
    wnd.add_drawer(state_drawer)

    class Hack(object):
        action = None

    hack = Hack()

    def pressed(event: keyboard.KeyboardEvent, hack_ref):
        hack_ref.action = None
        if event.name == 'down':
            hack_ref.action = ACTION_NO_OP
        if event.name == 'left':
            hack_ref.action = ACTION_LEFT
        if event.name == 'right':
            hack_ref.action = ACTION_RIGHT
        if event.name == 'up':
            hack_ref.action = ACTION_ROTATE
        print(event.name, hack_ref.action)

    keyboard.on_press(lambda evt: pressed(evt, hack))

    while True:
        #hack.action = np.random.choice(range(3))
        if hack.action is not None:
            state = env.do_action(hack.action)
            state_drawer.img = state
            replay.append(state, hack.action)
            hack.action = None
        else:
            time.sleep(0.1)

        wnd.draw()

        if env.is_terminal():
            state = env.reset()
            episode += 1

            if episode >= record_episodes:
                print('Saving replay')
                replay.save('D:\\deepirl\\replay\\tetris_{0}x{1}_{2}'
                            .format(env.state_shape[1], env.state_shape[0], record_episodes))
                sys.exit(0)
