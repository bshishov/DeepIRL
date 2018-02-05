import numpy as np

from deepirl.environments.base import Environment


class Snake(Environment):
    def __init__(self, width=8, height=8):
        super(self.__class__, self).__init__((height, width), 2)

    def do_action(self, action: int, update_state=True):
        pass

    def generate_expert_samples(self):
        pass

    def reset(self):
        pass


def main(arguments):
    from deepirl.utils.replay import StateActionReplay

    env = Snake()
    replay = StateActionReplay(10000, env.state_shape)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    main(parser.parse_args())
