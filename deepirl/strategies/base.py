import tensorflow as tf

from deepirl.models.base import RQModelBase
from deepirl.environments.base import Environment


class Strategy(object):
    def __init__(self, env: Environment, model: RQModelBase):
        self.env = env
        self.model = model

    def run(self, sess: tf.Session, num_episodes: int, *args, **kwargs):
        raise NotImplementedError
