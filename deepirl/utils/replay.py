import numpy as np


# Common memory columns
COL_STATE = 'state'
COL_ACTION = 'action'
COL_REWARD = 'reward'
COL_NEXT_STATE = 'next_state'
COL_IS_TERMINAL = 'is_terminal'
COL_PRIORITY = 'priority'
COL_WEIGHT = 'weight'


class ReplayBase(object):
    def append(self, *args):
        raise NotImplementedError

    def sample(self, batch_size: int) -> tuple:
        raise NotImplementedError

    def load(self, path: str):
        raise NotImplementedError

    def save(self, path: str):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class GenericMemory(ReplayBase):
    def __init__(self, capacity, definitions):
        self._memory = np.empty(capacity, dtype=definitions)
        self._capacity = capacity
        self._i = 0
        self._filled = 0
        self._definitions = definitions

    @property
    def capacity(self):
        return self._capacity

    @property
    def filled(self):
        return self._filled

    def append(self, *args):
        self._memory[self._i] = args
        self._i = (self._i + 1) % self._capacity
        self._filled = min(self._filled + 1, self._capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self._filled, min(self._filled, batch_size))
        batch = self._memory[indices]
        return (batch[col] for col, _, _ in self._definitions)

    def get_col(self, col_name):
        return self._memory[:self._filled][col_name]

    def __len__(self):
        return self._filled

    def save(self, path: str):
        # Save only the filled part to compressed array
        np.savez_compressed(path, memory=self._memory[:self._filled])

    def load(self, path: str):
        # Load only the filled part from compressed array
        data = np.load(path)

        # If it is a compressed array
        if not isinstance(data, np.ndarray):
            data = data['memory']

        self._filled = min(len(data), self._capacity)
        self._i = self._filled

        # If loaded data exceeds the capacity throw away exceeding part
        # TODO: reallocate memory ?
        self._memory[:self._filled] = data[:self._filled]

    def iterate(self):
        for i in range(self._filled):
            yield self._memory[i]

    def __getitem__(self, *args, **kwargs):
        return self._memory.__getitem__(*args, **kwargs)


class DqnReplayMemory(GenericMemory):
    def __init__(self, capacity, state_shape, action_shape=1, state_dtype=np.float32, action_dtype=np.uint16):
        super(self.__class__, self).__init__(capacity=capacity, definitions=[
            (COL_STATE, state_dtype, state_shape),
            (COL_ACTION, action_dtype, action_shape),
            (COL_REWARD, np.float32),
            (COL_IS_TERMINAL, np.bool)
        ])

    def sample(self, batch_size=32):
        indices = np.random.randint(0, self._filled - 1, min(self._filled - 1, batch_size))

        batch = self._memory[indices]
        batch_states = batch[COL_STATE]
        batch_actions = batch[COL_ACTION]
        batch_rewards = batch[COL_REWARD]
        batch_is_terminal = batch[COL_IS_TERMINAL]

        next_indices = indices + 1
        batch_next_states = self._memory[next_indices][COL_STATE]

        # States, actions, rewards, next_states, is_terminal
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_is_terminal


class StateActionReplay(GenericMemory):
    def __init__(self, capacity, state_shape, action_shape=1, state_dtype=np.float32, action_dtype=np.uint16):
        super(self.__class__, self).__init__(capacity=capacity, definitions=[
            (COL_STATE, state_dtype, state_shape),
            (COL_ACTION, action_dtype, action_shape),
        ])
