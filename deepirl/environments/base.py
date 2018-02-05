class Environment(object):
    def __init__(self, state_shape: tuple, num_actions: int):
        self.state_shape = state_shape
        self.num_actions = num_actions

    def do_action(self, action: int, update_state=True):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def is_terminal(self) -> bool:
        raise NotImplementedError
