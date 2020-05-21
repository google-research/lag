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

"""Custom neural network layers.

Low-level primitives such as custom convolution with custom initialization.
"""
import math

import numpy as np
import tensorflow as tf

NCHW, NHWC = 'NCHW', 'NHWC'
DATA_FORMAT_ORDER = {
    'channels_first': NCHW,
    'channels_last': NHWC
}


def smart_shape(x):
    s, t = x.shape, tf.shape(x)
    return [t[i] if s[i].value is None else s[i] for i in range(len(s))]


def to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])


def to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])


def torus_pad(x, w, order=NCHW):
    if w < 1:
        return x
    if order == NCHW:
        y = tf.concat([x[:, :, -w:], x, x[:, :, :w]], axis=2)
        y = tf.concat([y[:, :, :, -w:], y, y[:, :, :, :w]], axis=3)
    else:
        y = tf.concat([x[:, -w:], x, x[:, :w]], axis=1)
        y = tf.concat([y[:, :, -w:], y, y[:, :, :w]], axis=2)
    return y


def downscale2d(x, n=2, order=NCHW):
    """Box downscaling.

    Args:
    x: 4D tensor.
    n: integer scale.
    order: NCHW or NHWC.

    Returns:
    4D tensor down scaled by a factor n.
    """
    if n <= 1:
        return x
    if order == NCHW:
        return tf.nn.avg_pool(x, [1, 1, n, n], [1, 1, n, n], 'VALID', 'NCHW')
    else:
        return tf.nn.avg_pool(x, [1, n, n, 1], [1, n, n, 1], 'VALID', 'NHWC')


def upscale2d(x, n=2, order=NCHW):
    """Box upscaling (also called nearest neighbors).

    Args:
    x: 4D tensor in NCHW format.
    n: integer scale (must be a power of 2).

    Returns:
    4D tensor up scaled by a factor n.
    """
    if n == 1:
        return x
    s, ts = x.shape, tf.shape(x)
    if order == NCHW:
        x = tf.reshape(x, [-1, s[1], ts[2], 1, ts[3], 1])
        x = tf.tile(x, [1, 1, 1, n, 1, n])
        x = tf.reshape(x, [-1, s[1], ts[2] * n, ts[3] * n])
    else:
        x = tf.reshape(x, [-1, ts[1], 1, ts[2], 1, s[3]])
        x = tf.tile(x, [1, 1, n, 1, n, 1])
        x = tf.reshape(x, [-1, ts[1] * n, ts[2] * n, s[3]])
    return x


def remove_details2d(x, n=2):
    """Remove box details by upscaling a downscaled image.

    Args:
    x: 4D tensor in NCHW format.
    n: integer scale (must be a power of 2).

    Returns:
    4D tensor image with removed details of size nxn.
    """
    if n == 1:
        return x
    return upscale2d(downscale2d(x, n), n)


def bicubic_downscale2d(x, n=2, order=NCHW):
    """Downscale x by a factor of n, using dense bicubic weights.

    Args:
    x: 4D tensor in NCHW format.
    n: integer scale (must be a power of 2).

    Returns:
    4D tensor down scaled by a factor n.
    """

    def kernel_weight(x):
        """https://clouard.users.greyc.fr/Pantheon/experiments/rescaling/index-en.html#bicubic"""
        x = abs(x)
        if x <= 1:
            return 1.5 * x ** 3 - 2.5 * x ** 2 + 1
        elif 1 < x < 2:
            return - 0.5 * x ** 3 + 2.5 * x ** 2 - 4 * x + 2
        else:
            return 0

    def kernel():
        k1d = np.array([kernel_weight((x + 0.5) / n) for x in range(-2 * n, 2 * n)])
        k1d /= k1d.sum()
        k2d = np.outer(k1d, k1d.T).astype('f')
        return tf.constant(k2d.reshape((4 * n, 4 * n, 1, 1)))

    if order == NHWC:
        x = to_nchw(x)
    y = tf.pad(x, [[0, 0], [0, 0], [2 * n - 1, 2 * n], [2 * n - 1, 2 * n]], mode='REFLECT')
    s, ts = y.shape, tf.shape(y)
    y = tf.reshape(y, [ts[0] * s[1], 1, ts[2], ts[3]])
    y = tf.nn.conv2d(y, filter=kernel(), strides=[1, 1, n, n], padding='VALID', data_format='NCHW')
    y = tf.reshape(y, [ts[0], s[1], tf.shape(y)[2], tf.shape(y)[3]])
    return y if order == NCHW else to_nhwc(y)


def space_to_channels(x, n=2, order=NCHW):
    """Reshape image tensor by moving space to channels.

    Args:
    x: 4D tensor in NCHW format.
    n: integer scale (must be a power of 2).

    Returns:
    Reshaped 4D tensor image of shape (N, C * n**2, H // n, W // n).
    """
    s, ts = x.shape, tf.shape(x)
    if order == NCHW:
        x = tf.reshape(x, [-1, s[1], ts[2] // n, n, ts[3] // n, n])
        x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
        x = tf.reshape(x, [-1, s[1] * (n ** 2), ts[2] // n, ts[3] // n])
    else:
        x = tf.reshape(x, [-1, ts[1] // n, n, ts[2] // n, n, s[3]])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, ts[1] // n, ts[2] // n, s[3] * (n ** 2)])
    return x


def channels_to_space(x, n=2, order=NCHW):
    """Reshape image tensor by moving channels to space.

    Args:
    x: 4D tensor in NCHW format.
    n: integer scale (must be a power of 2).

    Returns:
    Reshaped 4D tensor image of shape (N, C // n**2, H * n, W * n).
    """
    s, ts = x.shape, tf.shape(x)
    if order == NCHW:
        x = tf.reshape(x, [-1, s[1] // (n ** 2), n, n, ts[2], ts[3]])
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, [-1, s[1] // (n ** 2), ts[2] * n, ts[3] * n])
    elif order == NHWC:
        x = tf.reshape(x, [-1, ts[1], ts[2], n, n, s[3] // (n ** 2)])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [-1, ts[1] * n, ts[2] * n, s[3] // (n ** 2)])
    else:
        assert 0, 'Only supporting NCHW and NHWC.'
    return x


class HeNormalInitializer(tf.initializers.random_normal):
    def __init__(self, slope, dtype=tf.float32):
        self.slope = slope
        self.dtype = dtype

    def get_config(self):
        return dict(slope=self.slope, dtype=self.dtype.name)

    def __call__(self, shape, dtype=None, partition_info=None):
        del partition_info
        if dtype is None:
            dtype = self.dtype
        std = np.sqrt(2) * tf.rsqrt((1. + self.slope ** 2) *
                                    tf.cast(tf.reduce_prod(shape[:-1]),
                                            tf.float32))
        return tf.random_normal(shape, stddev=std, dtype=dtype)


def blend_resolution(lores, hires, alpha):
    """Blend two images.

    Args:
        lores: 4D tensor in NCHW, low resolution image.
        hires: 4D tensor in NCHW, high resolution image.
        alpha: scalar tensor in [0, 1], 0 produces the low resolution, 1 the high one.

    Returns:
        4D tensor in NCHW of blended images.
    """
    return lores + alpha * (hires - lores)


class SingleUpdate:
    COLLECTION = 'SINGLE_UPDATE'

    @classmethod
    def get_update(cls, variable):
        for v, u in tf.get_collection(cls.COLLECTION):
            if v == variable:
                return u
        return None

    @classmethod
    def register_update(cls, variable, update):
        assert cls.get_update(variable) is None
        tf.add_to_collection(cls.COLLECTION, (variable, update))
        return update


class Conv2DSpectralNorm(tf.layers.Conv2D):
    def build(self, input_shape):
        was_built = self.built
        tf.layers.Conv2D.build(self, input_shape)
        self.built = was_built
        shape = self.kernel.shape.as_list()
        self.u = self.add_variable(name='u', shape=[1, shape[-1]], dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(),
                                   trainable=False)
        self.built = True

    def call(self, inputs):
        shape = self.kernel.shape.as_list()
        kernel = self.kernel
        if self.data_format == 'channels_first':
            kernel = tf.transpose(kernel, [0, 2, 3, 1])
        kernel = tf.reshape(kernel, [-1, shape[-1]])
        u = self.u
        v_ = tf.nn.l2_normalize(tf.matmul(u, kernel, transpose_b=True))
        u_ = tf.nn.l2_normalize(tf.matmul(v_, kernel))
        sigma = tf.squeeze(tf.matmul(tf.matmul(v_, kernel), u_, transpose_b=True))
        if SingleUpdate.get_update(u) is None:
            self.add_update(SingleUpdate.register_update(u, tf.assign(u, u_)))
        outputs = self._convolution_op(inputs, self.kernel / sigma)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=DATA_FORMAT_ORDER[self.data_format])
        if self.activation is None:
            return outputs
        return self.activation(outputs)


def conv2d_spectral_norm(x, filters, kernel_size, strides=1, padding='same',
                         activation=None, data_format='channels_last', **kwargs):
    layer = Conv2DSpectralNorm(filters, kernel_size, strides, padding,
                               activation=activation,
                               data_format=data_format, **kwargs)
    return layer.apply(x)


class DenseSpectralNorm(tf.layers.Dense):
    """Spectral Norm version of tf.layers.Dense."""

    def build(self, input_shape):
        self.u = self.add_variable(name='u', shape=[1, self.units], dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(),
                                   trainable=False)
        return tf.layers.Dense.build(self, input_shape)

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        u = self.u
        v_ = tf.nn.l2_normalize(tf.matmul(u, self.kernel, transpose_b=True))
        u_ = tf.nn.l2_normalize(tf.matmul(v_, self.kernel))
        sigma = tf.squeeze(tf.matmul(tf.matmul(v_, self.kernel), u_, transpose_b=True))
        if SingleUpdate.get_update(u) is None:
            self.add_update(SingleUpdate.register_update(u, tf.assign(u, u_)))
        outputs = tf.matmul(inputs, self.kernel / sigma)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


def dense_spectral_norm(inputs, units, activation=None, **kwargs):
    """Spectral Norm version of tf.layers.dense."""
    layer = DenseSpectralNorm(units, activation, **kwargs)
    return layer.apply(inputs)


class DenseSpectralNormCustom(tf.layers.Dense):
    """Spectral Norm version of tf.layers.Dense."""

    def build(self, input_shape):
        shape = [input_shape[-1], self.units]
        self.u = self.add_variable(name='u', shape=[1, shape[0]], dtype=tf.float32, trainable=False)
        self.v = self.add_variable(name='v', shape=[shape[1], 1], dtype=tf.float32, trainable=False)
        return tf.layers.Dense.build(self, input_shape)

    def call(self, inputs):
        inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
        u, v = self.u, self.v
        v_ = tf.nn.l2_normalize(tf.reshape(tf.matmul(u, self.kernel), v.shape))
        u_ = tf.nn.l2_normalize(tf.reshape(tf.matmul(self.kernel, v), u.shape))
        sigma = tf.matmul(tf.matmul(u, self.kernel), v)[0, 0]
        if SingleUpdate.get_update(u) is None:
            self.add_update(SingleUpdate.register_update(u, tf.assign(u, u_)))
            self.add_update(SingleUpdate.register_update(v, tf.assign(v, v_)))
        outputs = tf.matmul(inputs, self.kernel / sigma)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


def dense_spectral_norm_custom(inputs, units, activation=None, **kwargs):
    """Spectral Norm version of tf.layers.dense."""
    layer = DenseSpectralNormCustom(units, activation, **kwargs)
    return layer.apply(inputs)


def kaiming_scale(shape, activation):
    activation_slope = {
        tf.nn.relu: 0,
        tf.nn.leaky_relu: 0.2
    }
    slope = activation_slope.get(activation, 1)
    fanin = np.prod(shape[:-1])
    return np.sqrt(2. / ((1 + slope ** 2) * fanin))


class DenseScaled(tf.layers.Dense):
    def call(self, inputs):
        scale = kaiming_scale(self.kernel.get_shape().as_list(), self.activation)
        if hasattr(self, 'gain'):
            scale *= self.gain
        outputs = tf.matmul(inputs, self.kernel * scale)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is None:
            return outputs
        return self.activation(outputs)

    def set_gain(self, gain):
        self.gain = gain


class Conv2DScaled(tf.layers.Conv2D):
    def call(self, inputs):
        scale = kaiming_scale(self.kernel.get_shape().as_list(), self.activation)
        if hasattr(self, 'gain'):
            scale *= self.gain
        outputs = self._convolution_op(inputs, self.kernel * scale)
        assert self.rank == 2

        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias, DATA_FORMAT_ORDER[self.data_format])
        if self.activation is None:
            return outputs
        return self.activation(outputs)

    def set_gain(self, gain):
        self.gain = gain


def conv2d_scaled(x, filters, kernel_size, strides=1, padding='same',
                  activation=None, gain=1, data_format='channels_first', **kwargs):
    layer = Conv2DScaled(filters, kernel_size, strides, padding,
                         activation=activation,
                         data_format=data_format,
                         kernel_initializer=tf.initializers.random_normal(stddev=1.), **kwargs)
    layer.set_gain(gain)
    return layer.apply(x)


def dense_scaled(x, filters, activation=tf.nn.leaky_relu, gain=1, **kwargs):
    layer = DenseScaled(filters,
                        activation=activation,
                        kernel_initializer=tf.initializers.random_normal(stddev=1.),
                        **kwargs)
    layer.set_gain(gain)
    return layer.apply(x)


def channel_norm(x):
    """Channel normalization.

    Args:
      x: nD tensor with channels in dimension 1.

    Returns:
      nD tensor with normalized channels.
    """
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), [1], keepdims=True) + 1e-8)


def minibatch_mean_stddev(x):
    """Computes the standard deviation average.

    This is used by the discriminator as a form of batch discrimination.

    Args:
      x: nD tensor for which to compute standard deviation average.

    Returns:
      a scalar, the mean standard deviation of variable x.
    """
    mean = tf.reduce_mean(x, 0, keepdims=True)
    vals = tf.sqrt(tf.reduce_mean(tf.squared_difference(x, mean), 0) + 1e-8)
    vals = tf.reduce_mean(vals)
    return vals


def scalar_concat(x, scalar):
    """Concatenate a scalar to a 4D tensor as an extra channel.

    Args:
      x: 4D image tensor in NCHW format.
      scalar: a scalar to concatenate to the tensor.

    Returns:
      a 4D tensor with one extra channel containing the value scalar at
       every position.
    """
    s = tf.shape(x)
    return tf.concat([x, tf.ones([s[0], 1, s[2], s[3]]) * scalar], axis=1)


class ClassBiasScale(tf.layers.Layer):
    """For a class c, return x*gamma[c] + beta[c]"""

    def __init__(self, nclass, name=None, trainable=True, **kwargs):
        super(ClassBiasScale, self).__init__(
            name=name, trainable=trainable, **kwargs)
        self.nclass = nclass
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        self.beta = self.add_variable(name='beta', shape=[self.nclass, input_shape[1]], dtype=tf.float32,
                                      initializer=tf.initializers.zeros, trainable=True)
        self.gamma = self.add_variable(name='gamma', shape=[self.nclass, input_shape[1]], dtype=tf.float32,
                                       initializer=tf.initializers.zeros, trainable=True)
        self.built = True

    def call(self, inputs, labels):
        ndims = len(inputs.get_shape())
        with tf.colocate_with(self.beta):
            beta = tf.gather(self.beta, labels)
        with tf.colocate_with(self.gamma):
            gamma = tf.gather(self.gamma, labels)
        gamma = tf.nn.sigmoid(gamma)
        reshape = [tf.shape(inputs)[0], inputs.shape[1]] + [1] * (ndims - 2)
        return inputs * tf.reshape(gamma, reshape) + tf.reshape(beta, reshape)

    def compute_output_shape(self, input_shape):
        return input_shape


def conv2d_mono(x, kernel, order=NCHW):
    """2D convolution using the same filter for every channel.

    :param x: 4D input tensor of the images.
    :param kernel: 2D input tensor of the convolution to apply.
    :param order: enum {NCHW, NHWC}, the format of the input tensor.
    :return: a 4D output tensor resulting from the convolution.
    """
    y = x if order == NCHW else tf.transpose(x, [0, 3, 1, 2])
    s = smart_shape(y)
    y = tf.reshape(y, [s[0] * s[1], 1, s[2], s[3]])
    y = tf.nn.conv2d(y, kernel[:, :, None, None], [1] * 4, 'VALID', data_format=NCHW)
    t = smart_shape(y)
    y = tf.reshape(y, [s[0], s[1], t[2], t[3]])
    return y if order == NCHW else tf.transpose(y, [0, 2, 3, 1])


def class_bias_scale(inputs, labels, nclass):
    """For a class c, return x*gamma[c] + beta[c]"""
    layer = ClassBiasScale(nclass)
    return layer.apply(inputs, labels)


def blur_kernel_area(radius):
    """Compute an area blurring kernel.

    :param radius: float in [0, inf[, the ratio of the area.
    :return: a 2D convolution kernel.
    """
    radius = max(radius, 1e-8)
    cr = 1 + round(math.ceil(radius))
    m = np.ones((cr, cr), 'f')
    m[-1] *= (radius + 2 - cr)
    m[:, -1] *= (radius + 2 - cr)
    m = np.concatenate([m[::-1], m[1:]], axis=0)
    m = np.concatenate([m[:, ::-1], m[:, 1:]], axis=1)
    return m / m.sum()


def blur_apply(x, kernel, order=NCHW):
    h, w = kernel.shape[0], kernel.shape[1]
    if order == NCHW:
        x = tf.pad(x, [[0] * 2, [0] * 2, [h // 2] * 2, [w // 2] * 2], 'REFLECT')
    else:
        x = tf.pad(x, [[0] * 2, [h // 2] * 2, [w // 2] * 2, [0] * 2], 'REFLECT')
    return conv2d_mono(x, kernel, order)
