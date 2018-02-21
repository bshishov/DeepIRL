import numpy as np

from deepirl.environments.base import Environment


class Snake(Environment):
    def __init__(self, width=8, height=8):
        super(self.__class__, self).__init__((height, width), 3)
        self._state = np.zeros(self.state_shape, np.uint8)
        self._snake_pos = None
        self._food_pos = None
        self._is_terminal = False
        self.reset()

    def _redraw(self):
        self._state[:] = 0
        for pos in self._food_pos:
            self._state[pos] = 2
        for pos in self._snake_pos:
            self._state[pos] = 1

    def _place_food(self):
        free_cells_x, free_cells_y = np.where(self._state == 0)
        if len(free_cells_x) == 0:
            self._is_terminal = True
            return
        idx = np.random.randint(0, len(free_cells_x))
        self._food_pos.append((free_cells_y[idx], free_cells_x[idx]))

    def do_action(self, action: int, update_state=True):
        if self._is_terminal:
            return self._state

        # action == 0 - NO OP
        # action == 1 - LEFT
        # action == 2 - RIGHT
        head_y, head_x = self._snake_pos[-1]
        head1_y, head1_x = self._snake_pos[-2]
        dir_x = head_x - head1_x
        dir_y = head_y - head1_y

        # Left
        next_x = head_x
        next_y = head_y
        if action == 0:
            next_y = head_y + dir_y
            next_x = head_x + dir_x
        if action == 1:
            next_y = head_y + dir_x
            next_x = head_x + dir_y
        if action == 2:
            next_y = head_y - dir_x
            next_x = head_x - dir_y

        if next_x < 0 or next_x >= self._state.shape[1]:
            self._is_terminal = True
            return self._state
        if next_y < 0 or next_y >= self._state.shape[0]:
            self._is_terminal = True
            return self._state

        if self._state[next_y, next_x] == 1:
            self._is_terminal = True
            return self._state

        if (next_y, next_x) in self._food_pos:
            self._food_pos.remove((next_y, next_x))
            self._place_food()
        else:
            # If it is not a food, remove tail
            self._snake_pos.pop(0)

        self._snake_pos.append((next_y, next_x))
        self._redraw()

        return self._state

    def reset(self):
        self._snake_pos = []
        for i in range(3):
            self._snake_pos.append((i, 2))
        self._is_terminal = False
        self._food_pos = []
        self._place_food()
        self._redraw()
        return self._state

    def is_terminal(self):
        return self._is_terminal


def main(arguments):
    import deepirl.utils.vizualization as v
    import time
    from deepirl.utils.replay import StateActionReplay

    wnd = v.Window(420, 420)
    screen = v.ImageDrawer(v.Rect(10, 10, 400, 400))
    wnd.add_drawer(screen)

    env = Snake(width=10, height=10)
    state = env.reset()

    while True:
        screen.img = state
        action = np.random.choice(env.num_actions)
        state = env.do_action(action)
        wnd.draw()
        #time.sleep(0.1)

        if env.is_terminal():
            state = env.reset()

    #replay = StateActionReplay(10000, env.state_shape)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    main(parser.parse_args())
