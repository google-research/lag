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

# Inspired from https://github.com/machrisaa/tensorflow-vgg/blob/master/vgg19.py
# Download vgg19.npy from https://github.com/machrisaa/tensorflow-vgg

import os

import numpy as np
import tensorflow as tf

from libml.data import DATA_DIR

_VGG19_NPY = os.path.join(DATA_DIR, 'Models/vgg19.npy')
_VGG19_URL = 'https://github.com/machrisaa/tensorflow-vgg'


class Vgg19:
    BGR_MEAN = [103.939, 116.779, 123.68]

    def __init__(self):
        if not os.path.exists(_VGG19_NPY):
            raise FileNotFoundError(
                'You must downloaded vgg19.npy from %s and save it to %s' % (_VGG19_URL, _VGG19_NPY))
        self.data_dict = np.load(_VGG19_NPY, encoding='latin1', allow_pickle=True).item()

    def build(self, layer, inputs, channels_last=True):
        # inputs in [-1, 1]
        ops = [(self._conv2d, 'conv1_1'), (self._conv2d, 'conv1_2'), (self._max_pool, 'pool1'),
               (self._conv2d, 'conv2_1'), (self._conv2d, 'conv2_2'), (self._max_pool, 'pool2'),
               (self._conv2d, 'conv3_1'), (self._conv2d, 'conv3_2'),
               (self._conv2d, 'conv3_3'), (self._conv2d, 'conv3_4'), (self._max_pool, 'pool3'),
               (self._conv2d, 'conv4_1'), (self._conv2d, 'conv4_2'),
               (self._conv2d, 'conv4_3'), (self._conv2d, 'conv4_4'), (self._max_pool, 'pool4'),
               (self._conv2d, 'conv5_1'), (self._conv2d, 'conv5_2'),
               (self._conv2d, 'conv5_3'), (self._conv2d, 'conv5_4'), (self._max_pool, 'pool5')]

        rgb_scaled = (1 + inputs) * 127.5  # To [0, 255]
        if channels_last:
            red, green, blue = [rgb_scaled[:, :, :, x] for x in range(3)]
        else:
            red, green, blue = [rgb_scaled[:, x, :, :] for x in range(3)]

        # Starts with BGR data with channels_last.
        x = tf.stack([blue - self.BGR_MEAN[0], green - self.BGR_MEAN[1], red - self.BGR_MEAN[2]], axis=3)
        for func, name in ops:
            x = func(x, name)
            if name == layer:
                if channels_last:
                    return x
                return tf.transpose(x, [0, 3, 1, 2])
        raise NameError('No such layer "%s"' % layer)

    def _max_pool(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _conv2d(self, x, name):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            filt = tf.constant(self.data_dict[name][0], name='filter')
            conv_biases = tf.constant(self.data_dict[name][1], name='biases')

            conv = tf.nn.conv2d(x, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu
