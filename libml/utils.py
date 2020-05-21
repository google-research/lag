# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities."""

import glob
import io
import os
import re
import time

import matplotlib as mpl
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def setup_tf():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.logging.set_verbosity(tf.logging.ERROR)
    if not get_available_gpus():
        raise SystemError('You need at least 1 GPU.')


class HookReport(tf.train.SessionRunHook):
    """Custom reporting hook.

    Register your tensor scalars with HookReport.log_tensor(my_tensor, 'my_name').
    This hook will report their average values over report period argument
    provided to the constructed. The values are printed in the order the tensors
    were registered.

    Attributes:
      step: int, the current global step.
    """
    _REPORT_KEY = 'report'
    _ENABLE = True
    _TENSOR_NAMES = {}

    def __init__(self, period, batch_size):
        self.step = 0
        self._period = period // batch_size
        self._batch_size = batch_size
        self._sums = np.array([])
        self._count = 0
        self._step_ratio = 0
        self._start = time.time()

    @classmethod
    def disable(cls):
        class controlled_execution(object):
            def __enter__(self):
                cls._ENABLE = False
                return self

            def __exit__(self, type, value, traceback):
                cls._ENABLE = True

        return controlled_execution()

    def begin(self):
        self._count = 0
        self._start = time.time()

    def before_run(self, run_context):
        del run_context
        fetches = tf.get_collection(self._REPORT_KEY)
        fetches = fetches + [tf.train.get_global_step()]
        return tf.train.SessionRunArgs(fetches)

    def after_run(self, run_context, run_values):
        del run_context
        results = run_values.results
        # Note: sometimes the returned step is incorrect (off by one) for some
        # unknown reason.
        self.step = results[-1] + 1
        self._count += 1

        if not self._sums.size:
            self._sums = np.array(results[:-1], 'd')
        else:
            self._sums += np.array(results[:-1], 'd')

        if self.step // self._period != self._step_ratio:
            fetches = tf.get_collection(self._REPORT_KEY)
            stats = '  '.join('%s=% .2f' % (self._TENSOR_NAMES[tensor],
                                            value / self._count)
                              for tensor, value in zip(fetches, self._sums))
            stop = time.time()
            tf.logging.info('kimg=%d  %s  [%.2f img/s]' %
                            ((self.step * self._batch_size) >> 10, stats,
                             self._batch_size * self._count / (
                                     stop - self._start)))
            self._step_ratio = self.step // self._period
            self._start = stop
            self._sums *= 0
            self._count = 0

    def end(self, session=None):
        del session

    @classmethod
    def log_tensor(cls, tensor, name):
        """Adds a tensor to be reported by the hook.

        Args:
          tensor: `tensor scalar`, a value to report.
          name: string, the name to give the value in the report.

        Returns:
          None.
        """
        if cls._ENABLE:
            cls._TENSOR_NAMES[tensor] = name
            tf.add_to_collection(cls._REPORT_KEY, tensor)
            tf.summary.scalar(name, tensor)


class Shuffler:
    """Shuffles data in a buffer. This allows to shuffle data post augmentation for smoother gradient descent."""

    def __init__(self, buffer_size=4096):
        self.buffer_size = buffer_size
        self.fill = 0
        self.buffers = None

    def shuffle(self, *args):
        if self.buffers is None:
            self.buffers = [np.zeros((self.buffer_size,) + x.shape[1:], x.dtype) for x in args]
        batch = args[0].shape[0]
        if self.fill == self.buffer_size:
            sel = np.random.choice(self.buffer_size, batch, replace=False)
            new_args = [buffer[sel] for buffer in self.buffers]
            for buffer, x in zip(self.buffers, args):
                buffer[sel] = x
            return new_args
        elif self.fill < self.buffer_size:
            for buffer, x in zip(self.buffers, args):
                buffer[self.fill: self.fill + batch] = x
            self.fill += batch
            return args
        else:
            assert 0


# A dict where you can use a.b for a['b']
class ClassDict(dict):

    def __init__(self, *args, **kwargs):
        super(ClassDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def ilog2(x):
    """Integer log2."""
    return int(np.ceil(np.log2(x)))


def to_hwc(image):
    """Converts a CHW image to HWC."""
    return image.transpose(1, 2, 0)


def images_to_grid(images):
    """Converts a grid of images (NNCHW 5D tensor) to a single image in HWC.

    Args:
      images: 5D tensor (count_y, count_x, colors, height, width), grid of images.

    Returns:
      a 3D tensor image of shape (count_y * height, count_x * width, colors).
    """
    ny, nx, c, h, w = images.shape
    images = images.transpose(0, 3, 1, 4, 2)
    images = images.reshape([ny * h, nx * w, c])
    return images


def save_images(image, output_dir, cur_nimg, name=None):
    """Saves images to disk.

    Saves a file called 'name.png' containing the latest samples from the
     generator and a file called 'name_123.png' where 123 is the KiB of trained
     images.

    Args:
      image: 3D numpy array (height, width, colors), the image to save.
      output_dir: string, the directory where to save the image.
      cur_nimg: int, current number of images seen by training.

    Returns:
      None
    """
    if name:
        names = [name]
    else:
        names = ('name.png', 'name_%06d.png' % (cur_nimg >> 10))
    for name in names:
        with tf.gfile.Open(os.path.join(output_dir, name), 'wb') as f:
            f.write(image)


def to_png(x):
    """Convert a 3D tensor to png.

    Args:
      x: Tensor, 01C formatted input image.

    Returns:
      Tensor, 1D string representing the image in png format.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess_temp:
            x = tf.constant(x, dtype=tf.float32)
            y = tf.image.encode_png(
                tf.cast(
                    tf.clip_by_value(tf.round(127.5 + 127.5 * x), 0, 255),
                    tf.uint8),
                compression=9)
            return sess_temp.run(y)


def from_png(x):
    """Convert a png to a 3D tensor.

    Args:
      x: png image bytestring.

    Returns:
      Tensor, 3D image.
    """
    with tf.Graph().as_default():
        with tf.Session() as sess_temp:
            x = tf.constant(x)
            y = tf.image.decode_png(x)
            return sess_temp.run(y)


def find_latest_checkpoint(dir, glob_term='model.ckpt-*.meta'):
    """Replacement for tf.train.latest_checkpoint.

    It does not rely on the "checkpoint" file which sometimes contains
    absolute path and is generally hard to work with when sharing files
    between users / computers.
    """
    r_step = re.compile('.*model\.ckpt-(?P<step>\d+)\.meta')
    matches = glob.glob(os.path.join(dir, glob_term))
    matches = [(int(r_step.match(x).group('step')), x) for x in matches]
    ckpt_file = max(matches)[1][:-5]
    return ckpt_file


def get_latest_global_step(dir):
    """Loads the global step from the latest checkpoint in directory.
  
    Args:
      dir: string, path to the checkpoint directory.
  
    Returns:
      int, the global step of the latest checkpoint or 0 if none was found.
    """
    try:
        checkpoint_reader = tf.train.NewCheckpointReader(find_latest_checkpoint(dir))
        return checkpoint_reader.get_tensor(tf.GraphKeys.GLOBAL_STEP)
    except:  # pylint: disable=bare-except
        return 0


def get_latest_global_step_in_subdir(dir):
    """Loads the global step from the latest checkpoint in sub-directories.

    Args:
      dir: string, parent of the checkpoint directories.

    Returns:
      int, the global step of the latest checkpoint or 0 if none was found.
    """
    sub_dirs = (x for x in glob.glob(os.path.join(dir, '*')) if os.path.isdir(x))
    step = 0
    for x in sub_dirs:
        step = max(step, get_latest_global_step(x))
    return step


def getter_ema(ema, getter, name, *args, **kwargs):
    """Exponential moving average getter for variable scopes.

    Args:
        ema: ExponentialMovingAverage object, where to get variable moving averages.
        getter: default variable scope getter.
        name: variable name.
        *args: extra args passed to default getter.
        **kwargs: extra args passed to default getter.

    Returns:
        If found the moving average variable, otherwise the default variable.
    """
    var = getter(name, *args, **kwargs)
    ema_var = ema.average(var)
    return ema_var if ema_var else var


def model_vars(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def average_gradients(tower_grads):
    # Adapted from:
    #  https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. For each tower, a list of its gradients.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    if len(tower_grads) <= 1:
        return tower_grads[0]

    average_grads = []
    for grads_and_vars in zip(*tower_grads):
        grad = tf.reduce_mean([gv[0] for gv in grads_and_vars], 0)
        average_grads.append((grad, grads_and_vars[0][1]))
    return average_grads


def para_list(fn, *args):
    """Run on multiple GPUs in parallel and return list of results."""
    gpus = len(get_available_gpus())
    if gpus <= 1:
        return zip(*[fn(*args)])
    splitted = [tf.split(x, gpus) for x in args]
    outputs = []
    for gpu, x in enumerate(zip(*splitted)):
        with tf.name_scope('tower%d' % gpu):
            with tf.device(tf.train.replica_device_setter(
                    worker_device='/gpu:%d' % gpu, ps_device='/cpu:0', ps_tasks=1)):
                outputs.append(fn(*x))
    return zip(*outputs)


def para_mean(fn, *args):
    """Run on multiple GPUs in parallel and return means."""
    gpus = len(get_available_gpus())
    if gpus <= 1:
        return fn(*args)
    splitted = [tf.split(x, gpus) for x in args]
    outputs = []
    for gpu, x in enumerate(zip(*splitted)):
        with tf.name_scope('tower%d' % gpu):
            with tf.device(tf.train.replica_device_setter(
                    worker_device='/gpu:%d' % gpu, ps_device='/cpu:0', ps_tasks=1)):
                outputs.append(fn(*x))
    if isinstance(outputs[0], (tuple, list)):
        return [tf.reduce_mean(x, 0) for x in zip(*outputs)]
    return tf.reduce_mean(outputs, 0)


def para_cat(fn, *args):
    """Run on multiple GPUs in parallel and return concatenated outputs."""
    gpus = len(get_available_gpus())
    if gpus <= 1:
        return fn(*args)
    splitted = [tf.split(x, gpus) for x in args]
    outputs = []
    for gpu, x in enumerate(zip(*splitted)):
        with tf.name_scope('tower%d' % gpu):
            with tf.device(tf.train.replica_device_setter(
                    worker_device='/gpu:%d' % gpu, ps_device='/cpu:0', ps_tasks=1)):
                outputs.append(fn(*x))
    if isinstance(outputs[0], (tuple, list)):
        return [tf.concat(x, axis=0) for x in zip(*outputs)]
    return tf.concat(outputs, axis=0)


def plot_points(data, width=768):
    """Plot points in a 2D plane [-2, 2]. Return an RGB image."""
    fig, ax = plt.subplots(figsize=(width / 100.,) * 2)
    ax.set_aspect('equal')
    ax.set_ylim((-2, 2))
    ax.set_xlim((-2, 2))
    sns.kdeplot(data[:, 0], data[:, 1], cmap='Blues', shade=True, shade_lowest=False, ax=ax)
    ax.scatter(data[:, 0], data[:, 1], linewidth=1, marker='+', c=None, color='m')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    png = buf.getvalue()
    buf.close()
    return from_png(png)


def smart_shape(x):
    s = x.shape
    st = tf.shape(x)
    return [s[i] if s[i].value is not None else st[i] for i in range(4)]
