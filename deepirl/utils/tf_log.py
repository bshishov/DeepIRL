import os
import tensorflow as tf
import datetime
import numpy as np


def get_dependencies(tensor: tf.Tensor) -> set:
    dependencies = set()
    dependencies.update(tensor.op.inputs)
    for sub_op in tensor.op.inputs:
        dependencies.update(get_dependencies(sub_op))
    return dependencies


def get_placeholder_dependencies(tensor: tf.Tensor) -> list:
    return [tensor for tensor in get_dependencies(tensor) if tensor.op.type == 'Placeholder']


def get_dir_path(desired_path: str):
    # If desired path not exists - return it
    if not os.path.exists(desired_path):
        return desired_path

    # WHAT?!
    last_number_sep = desired_path.rfind('_')
    if last_number_sep < 0:
        return get_dir_path(desired_path + '_1')
    number_info = desired_path[last_number_sep + 1:]
    try:
        # If there is a number in the filename - increment
        int_val = int(number_info)
        new_path = '{0}_{1}'.format(desired_path[:last_number_sep], int_val + 1)
        return get_dir_path(new_path)
    except ValueError:
        return get_dir_path(desired_path + '_1')


class SummaryWrapper(object):
    def __init__(self):
        self._cached = None

    def get_cached_value(self) -> bytes:
        return self._cached

    def update_cached_value(self, value: bytes):
        self._cached = value


class TensorSummaryWrapper(SummaryWrapper):
    def __init__(self, summary_tensor):
        super().__init__()
        self.tensor = summary_tensor
        self.dependencies = get_placeholder_dependencies(self.tensor)


class SimpleSummaryWrapper(SummaryWrapper):
    def __init__(self, value=None):
        super().__init__()
        self.update_cached_value(value)


class TfLogger(tf.train.SessionRunHook):
    STEP_TENSOR_KEY = 'summaries_step'

    def __init__(self,
                 log_dir: str,
                 every_n_steps: int = 100,
                 write_each_update: bool = True,
                 step_tensor: tf.Variable = None,
                 run_name: str = None,
                 summaries_scope: str = 'Summaries',
                 datetime_fmt: str = '%d-%m-%Y_%H-%M'):
        super().__init__()
        if not run_name:
            run_name = datetime.datetime.now().strftime(datetime_fmt)
        self._every_n_steps = every_n_steps
        self._write_each_update = write_each_update

        self._output_dir = get_dir_path(os.path.join(log_dir, run_name))
        self._writer = None  # type: tf.summary.FileWriter
        self._step_tensor = step_tensor
        self._step = None
        self._last_update_step = None
        self._update_requested = False

        self._summary_wrappers = {}  # type: dict[str, SummaryWrapper]
        self._scope = summaries_scope

    def track_tensor(self, tensor: tf.Tensor, name: str = None, scope: str = None):
        if not name:
            name = tensor.name

        path = [self._scope, scope]
        scope_name = '/'.join([p for p in path if p])

        with tf.name_scope(scope_name + '/'):
            self._summary_wrappers[name] = TensorSummaryWrapper(tf.summary.scalar(name, tensor))

    def log_scalar(self, value: float, name: str, scope: str = None):
        path = [self._scope, scope, name]
        tag = '/'.join([p for p in path if p])

        serialized_summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self._summary_wrappers[name] = SimpleSummaryWrapper(serialized_summary)

    def begin(self):
        if self._output_dir and self._writer is None:
            self._writer = tf.summary.FileWriter(self._output_dir, tf.get_default_graph())

        if self._step_tensor is None:
            self._step_tensor = tf.train.get_or_create_global_step()

    def before_run(self, run_context: tf.train.SessionRunContext):
        requests = {self.STEP_TENSOR_KEY: self._step_tensor}

        if self._should_update():
            self._update_requested = True

        if self._update_requested:
            for key, summary in self._summary_wrappers.items():
                if not isinstance(summary, TensorSummaryWrapper):
                    continue

                dependencies_satisfied = True
                if run_context.original_args.feed_dict is not None:
                    for d in summary.dependencies:
                        if d not in run_context.original_args.feed_dict:
                            dependencies_satisfied = False
                            break
                else:
                    if len(summary.dependencies) > 0:
                        dependencies_satisfied = False

                if dependencies_satisfied:
                    requests[key] = summary.tensor

        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context: tf.train.SessionRunContext, run_values: tf.train.SessionRunValues):
        self._step = run_values.results[self.STEP_TENSOR_KEY]

        if self._update_requested:
            # Update summary cache if it is in results
            for key in self._summary_wrappers.keys():
                if key in run_values.results and isinstance(self._summary_wrappers[key], TensorSummaryWrapper):
                    data = run_values.results[key]
                    self._summary_wrappers[key].update_cached_value(data)
            self._last_update_step = self._step
            self._update_requested = False

            if self._write_each_update:
                self.write(self._step)

    def _should_update(self) -> bool:
        if self._step is None or self._last_update_step is None:
            return True
        return (self._step - self._last_update_step + 1) >= self._every_n_steps

    def write(self, step: int = None):
        if not self._writer:
            return

        if step is None:
            step = self._step

        for key, summary in self._summary_wrappers.items():
            cached_value = summary.get_cached_value()
            if cached_value is not None:
                self._writer.add_summary(summary.get_cached_value(), step)

    def end(self, session):
        if self._writer:
            self._writer.flush()

    def __del__(self):
        if self._writer:
            self._writer.close()


def test():
    logger = TfLogger('D:/test_logs/', every_n_steps=1)

    x = tf.placeholder(tf.float32, (10, 10))
    a = tf.get_variable('a', (10, 10), dtype=tf.float32, initializer=tf.initializers.random_normal())
    b = tf.get_variable('b', (10, 10), dtype=tf.float32, initializer=tf.initializers.random_normal())
    y = a * x + b

    x_mean = tf.reduce_mean(x)
    y_mean = tf.reduce_mean(y)

    logger.track_tensor(x_mean, name='XMean')
    logger.track_tensor(y_mean, name='YMean')

    with tf.train.MonitoredSession(hooks=[logger]) as sess:
        for i in range(100):
            xx = np.random.random((10, 10))
            logger.log_scalar(i, 'Iteration')

            # Run something that is not be able to update summaries
            aa = sess.run(a)

            # Run something that updates summaries
            yy = sess.run(y, feed_dict={x: xx})

            print(i)
'''
    logger = Logger('some_path')
    
    logger.track_tensor(some_tensor)
    logger.log_scalar(1.2)    
    logger.write()
'''

if __name__ == '__main__':
    test()
