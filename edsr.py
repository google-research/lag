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

"""EDSR reimplementation.

Enhanced Deep Residual Networks for Single Image Super-Resolution
https://arxiv.org/abs/1707.02921

Note: I simplified the model (with the same accuracy) by removing the
rescaling layers, only is needed in the end since all transformations
are linear.
"""

import os

import tensorflow as tf
from absl import app
from absl import flags
from easydict import EasyDict

from libml import layers, utils, data
from libml.train_sr import SRES

FLAGS = flags.FLAGS


class EDSR(SRES):

    def model(self, dataset, scale, repeat, filters, **kwargs):
        del kwargs
        x = tf.placeholder(tf.float32, [None, dataset.colors, dataset.height, dataset.width], 'x')
        y = tf.placeholder(tf.float32, [None, dataset.colors, None, None], 'y')

        conv_args = dict(data_format='channels_first', padding='same')

        def sres(x0, scope='sres'):
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                y = tf.layers.conv2d(x0, filters, 3, **conv_args)
                for x in range(repeat):
                    dy = tf.layers.conv2d(y, filters, 3, activation=tf.nn.relu, **conv_args)
                    y += tf.layers.conv2d(dy, filters, 3, **conv_args) / repeat
                y = tf.layers.conv2d(y, x0.shape[1] * scale ** 2, 3, **conv_args)
                y = layers.channels_to_space(y, scale)
                return y + layers.upscale2d(x0, scale)

        loss = tf.losses.mean_squared_error(x, sres(self.downscale(x)))
        utils.HookReport.log_tensor(tf.sqrt(loss) * 127.5, 'rmse')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss, global_step=tf.train.get_global_step())

        return EasyDict(x=x, y=y, sres_op=sres(y), eval_op=sres(self.downscale(x)), train_op=train_op)


def main(argv):
    del argv  # Unused.
    dataset = data.get_dataset(FLAGS.dataset)
    model = EDSR(
        os.path.join(FLAGS.train_dir, dataset.name),
        lr=FLAGS.lr,
        batch=FLAGS.batch,
        scale=FLAGS.scale,
        downscaler=FLAGS.downscaler,
        filters=FLAGS.filters,
        repeat=FLAGS.repeat)
    model.train(dataset)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_integer('filters', 256, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 16, 'Depth of residual network.')
    FLAGS.set_default('batch', 16)
    FLAGS.set_default('lr', 0.0001)
    FLAGS.set_default('total_kimg', 1 << 14)
    app.run(main)
