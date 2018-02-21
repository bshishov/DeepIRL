class Environment(object):
    def __init__(self,
                 state_shape: tuple,
                 num_actions: int,
                 discrete=True,
                 continuous_bounds=None):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.discrete = discrete
        self.continuous_bounds = continuous_bounds

    def do_action(self, action, update_state=True):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def is_terminal(self) -> bool:
        raise NotImplementedError
