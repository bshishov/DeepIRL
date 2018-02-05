import numpy as np
from deepirl.environments.base import Environment


SYMBOL_0 = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
]

SYMBOL_1 = [
    [0, 0, 1],
    [0, 1, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
]

SYMBOL_2 = [
    [1, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 0],
    [1, 1, 1],
]

SYMBOL_3 = [
    [1, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
]

SYMBOL_4 = [
    [1, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 0, 1],
    [0, 0, 1],
]

SYMBOL_5 = [
    [1, 1, 1],
    [1, 0, 0],
    [1, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
]

SYMBOL_6 = [
    [1, 1, 1],
    [1, 0, 0],
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
]

SYMBOL_7 = [
    [1, 1, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
]

SYMBOL_8 = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
]

SYMBOL_9 = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
]

SYMBOLS = [np.array(s, dtype=np.float32) for s in [SYMBOL_0, SYMBOL_1, SYMBOL_2, SYMBOL_3, SYMBOL_4,
                                                   SYMBOL_5, SYMBOL_6, SYMBOL_7, SYMBOL_8, SYMBOL_9]]


def try_place(canvas: np.ndarray,
              symbol: np.ndarray,
              position: tuple,
              fill_map: np.ndarray,
              padding: int = 1,
              symbol_brightness=1.0):
    x, y = position
    h, w = symbol.shape

    if x < padding or x + w > canvas.shape[1] - padding:
        return False

    if y < padding or y + h > canvas.shape[0] - padding:
        return False

    if np.sum(fill_map[y - padding:y + h + padding, x - padding:x + w + padding]) < 1.0:
        canvas[y:y + h, x:x + w] += symbol * symbol_brightness
        fill_map[y - padding:y + h + padding, x - padding:x + w + padding] = 1.0
        return True
    return False


def generate(width: int, height: int, symbols: list, background_noise: float, symbol_brightness: float, padding: int):
    shape = (height, width)
    if background_noise > 0.0:
        canvas = np.random.random(shape) * background_noise
    else:
        canvas = np.zeros(shape, dtype=np.float32)

    # Matrix indicating already filled area by symbols
    fill_map = np.zeros(shape, dtype=np.float32)

    positions = []
    for symbol in symbols:
        h, w = symbol.shape
        while True:
            x = np.random.randint(0, width - w)
            y = np.random.randint(0, height - h)
            position = (x, y)
            placed = try_place(canvas, symbol, position, fill_map, padding=padding, symbol_brightness=symbol_brightness)
            if placed:
                positions.append(position)
                break

    return canvas, positions


def gaussian_2d(width: int, height: int, position: tuple, sigma: float = 0.05):
    x, y = position

    x_axis, y_axis = np.meshgrid(
        np.linspace(-x, -x + 1, width),
        np.linspace(-y, -y + 1, height))
    d = np.sqrt(x_axis * x_axis + y_axis * y_axis)
    g = np.exp(-(d ** 2 / (2.0 * sigma ** 2)))
    return g


def generate_samples(positions,
                     fixations_overview: int=0,
                     fixations_spread: tuple=(3, 5),
                     fixations_min: int=9,
                     fixations_max: int=10):
    samples = []
    spread_x, spread_y = fixations_spread

    for i in range(fixations_overview):
        # random symbol position
        x, y = positions[np.random.randint(0, len(positions))]
        sample = (x + np.random.rand() * spread_x, y + np.random.rand() * spread_y)
        samples.append(sample)

    for x, y in positions:
        for i in range(np.random.randint(fixations_min, fixations_max)):
            sample = (x + np.random.rand() * spread_x, y + np.random.rand() * spread_y)
            samples.append(sample)

    return samples


class SymbolsEnvironment(Environment):
    def __init__(self,
                 width: int,
                 height: int,
                 symbols=SYMBOLS,
                 static_noise: float = 0.1,
                 dynamic_noise: float = 0.1,
                 forget_rate: float = 0.99,
                 same_image=False,
                 symbol_padding=1,
                 symbol_brightness: float = 1.0,
                 sight_sigma: float = 0.05,
                 sight_per_fixation: float = 1.0,
                 fixations_per_symbol=1,
                 seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.h, self.w = height, width
        super(self.__class__, self).__init__(state_shape=(self.h, self.w, 2), num_actions=self.w * self.h)
        self.symbols = symbols
        self.dynamic_noise = dynamic_noise
        self.forget_rate = forget_rate

        self._centered_sight_map = gaussian_2d(self.w, self.h, (0.5, 0.5), sigma=sight_sigma) * sight_per_fixation
        self._generator = lambda: generate(width, height,
                                           background_noise=static_noise,
                                           symbols=self.symbols,
                                           padding=symbol_padding,
                                           symbol_brightness=symbol_brightness)

        self.image, self.positions = self._generator()
        self.state = np.zeros(self.state_shape, dtype=np.float32)
        self.state[:, :, 0] = self.image
        self.step = 0
        self.same_image = same_image

        self.fixations_per_symbol = fixations_per_symbol
        self.max_actions = fixations_per_symbol * len(self.symbols)

    def reset(self):
        if not self.same_image:
            self.image, self.positions = self._generator()

        self.state[:, :, 0] = self.image
        self.state[:, :, 1] = 0.0
        self.step = 0
        return self.state.copy()

    def do_action(self, action: int, update_state=True):
        """
        :param action: Action index
        :param update_state: If it is True - internal environment state would be updated
        :return: new state
        """
        if update_state:
            self.step += 1
            return self._state_transition(self.state, action).copy()
        else:
            return self._state_transition(self.state.copy(), action)

    def position_to_action(self, x, y):
        return int(y) * self.w + int(x)

    def is_terminal(self):
        return self.step > self.max_actions

    def _state_transition(self, state, action):
        h, w = self.image.shape
        x = action % w
        y = action // w

        # MDP transition:
        # Add some noise to the original image
        state[:, :, 0] = self.image + np.random.random(self.image.shape) * self.dynamic_noise

        # Sight map
        # Draw a gaussian distributed area onto coverage map in the action location
        # self.raw[:, :, 1] += sight_map((x / w, y / h)) * sight_per_fixation  # NOT efficient

        # This is just copying the patch from precomputed 2D gaussian distribution
        # to the sight coverage. Just as optimization.

        half_w = w // 2
        half_h = h // 2

        # Sight map patch bounds, the patch is shifted by action
        left_1 = max(x - half_w, 0)
        right_1 = min(x + half_w, w)
        top_1 = max(y - half_h, 0)
        bottom_1 = min(y + half_h, h)

        # Gaussian dist patch bounds, the patch is shifted by action
        left_2 = max(half_w - x, 0)
        right_2 = min(w - x + half_w, w)
        top_2 = max(half_h - y, 0)
        bottom_2 = min(h - y + half_h, h)

        # Fill th sight coverage map according to bounds
        state[top_1:bottom_1, left_1:right_1, 1] += self._centered_sight_map[top_2:bottom_2, left_2:right_2]

        # Forgetting the old sight coverage a bit
        if self.forget_rate < 1.0:
            state[..., 1] *= self.forget_rate

        return state


def generate_expert_trajectories(env: SymbolsEnvironment,
                                 num_trajectories: int = 1000,
                                 fixation_spread_x: int = 3,
                                 fixation_spread_y: int = 5,
                                 capacity=100000):
    from deepirl.utils.replay import StateActionReplay
    replay = StateActionReplay(capacity, env.state_shape)
    for i in range(num_trajectories):
        env.reset()
        samples = []

        # N fixations per each symbol
        for x, y in env.positions:
            for fixation_i in range(env.fixations_per_symbol):
                sample_x = x + np.random.rand() * fixation_spread_x
                sample_y = y + np.random.rand() * fixation_spread_y
                samples.append(env.position_to_action(sample_x, sample_y))

        for action in samples:
            replay.append(env.state.copy(), action)
            env.do_action(action)
    return replay


def _test_explain():
    import time
    import deepirl.utils.vizualization as v
    wnd = v.Window(640, 480)
    state_drawer1 = v.ImageDrawer(v.Rect(10, 10, 200, 200))
    state_drawer2 = v.ImageDrawer(v.Rect(220, 10, 200, 200), color_map=v.ColorMap.HOT)
    wnd.add_drawer(state_drawer1)
    wnd.add_drawer(state_drawer2)

    env = SymbolsEnvironment(32, 32, fixations_per_symbol=10)
    replay = generate_expert_trajectories(env, num_trajectories=10)

    while True:
        for state, action in replay.iterate():
            state_drawer1.img = state[..., 0]
            state_drawer2.img = state[..., 1]
            wnd.draw()
            time.sleep(0.01)


if __name__ == '__main__':
    import argparse
    from deepirl.utils.config import instantiate

    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=str, help="Output file containing trajectories")
    parser.add_argument("--trajectories", type=int, help="Number of expert trajectories", default=1000)
    parser.add_argument("--env", type=str, help="Path to env configuration", default='config/envs/symbols.json')
    args = parser.parse_args()

    env = instantiate(args.env)

    print('Generating {0} trajectories'.format(args.trajectories))
    replay = generate_expert_trajectories(env, num_trajectories=args.trajectories)
    print('Saving...')
    replay.save(args.out)
    print('Saved to {0}'.format(args.out))
