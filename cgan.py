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

"""cGAN reimplementation.

cGANs with Projection Discriminator.
https://openreview.net/forum?id=ByS1VpgRZ
https://github.com/pfnet-research/sngan_projection
"""

import os

import tensorflow as tf
from absl import app
from absl import flags
from easydict import EasyDict

from libml import data, layers, utils
from libml.train_sr import SRES

FLAGS = flags.FLAGS


def g_resnet_block(x0, filters, train, noise=0, upsample=False):
    conv_args = dict(padding='same', kernel_initializer=tf.glorot_uniform_initializer())
    s = utils.smart_shape(x0)
    h = x = x0
    h = tf.layers.batch_normalization(h, training=train)
    h = tf.nn.relu(h)
    if noise:
        z = tf.random_normal([s[0], s[1], s[2], noise])
        h = tf.concat([h, z], axis=3)
    if upsample:
        h = layers.upscale2d(h, 2, layers.NHWC)
        x = layers.upscale2d(x, 2, layers.NHWC)
    h = tf.layers.conv2d(h, filters, 3, **conv_args)
    h = tf.layers.batch_normalization(h, training=train)
    h = tf.nn.relu(h)
    h = tf.layers.conv2d(h, filters, 3, **conv_args)
    if s[3] != filters:
        x = tf.layers.conv2d(x, filters, 1, **conv_args)
    return h + x


def d_resnet_block(x0, filters, downsample=False):
    conv_args = dict(padding='same', kernel_initializer=tf.glorot_uniform_initializer())
    s = utils.smart_shape(x0)
    h = x = x0
    h = tf.nn.relu(h)
    h = layers.conv2d_spectral_norm(h, filters, 3, **conv_args)
    h = tf.nn.relu(h)
    h = layers.conv2d_spectral_norm(h, filters, 3, **conv_args)
    if downsample:
        h = layers.downscale2d(h, 2, layers.NHWC)
        x = layers.downscale2d(x, 2, layers.NHWC)
    if s[3] != filters:
        x = tf.layers.conv2d(x, filters, 1, **conv_args)
    return h + x


def d_optimized_resnet_block(x0, filters):
    conv_args = dict(padding='same', kernel_initializer=tf.glorot_uniform_initializer())
    h = layers.conv2d_spectral_norm(x0, filters, 3, activation=tf.nn.relu, **conv_args)
    h = layers.conv2d_spectral_norm(h, filters, 3, **conv_args)
    h = layers.downscale2d(h, 2, layers.NHWC)
    x = layers.downscale2d(x0, 2, layers.NHWC)
    x = layers.conv2d_spectral_norm(x, filters, 1, **conv_args)
    return x + h


class cGAN(SRES):

    def sres(self, x, noise, filters, blocks, train):
        conv_args = dict(padding='same', kernel_initializer=tf.glorot_uniform_initializer())
        colors = utils.smart_shape(x)[3]

        with tf.variable_scope('sres', reuse=tf.AUTO_REUSE):
            h = x
            for block in range(blocks):
                h = g_resnet_block(h, filters << self.log_scale, train)
            for block in range(self.log_scale - 1, -1, -1):
                h = g_resnet_block(h, filters << block, train, noise=noise, upsample=True)
                h = g_resnet_block(h, filters << block, train)
            h = tf.layers.batch_normalization(h, training=train)
            h = tf.nn.relu(h)
            h = tf.layers.conv2d(h, colors, 3, activation=tf.nn.tanh, **conv_args)
            return h

    def disc(self, x, x_lr, resolution, filters):
        conv_args = dict(padding='same', kernel_initializer=tf.glorot_uniform_initializer())
        lr_h, lr_w, lr_c = [tf.cast(v, tf.float32) for v in utils.smart_shape(x_lr)[1:]]
        colors = utils.smart_shape(x)[3]
        log_res = utils.ilog2(resolution)

        with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
            h = x
            h = d_optimized_resnet_block(h, filters)
            h = d_resnet_block(h, filters)
            for block in range(1, self.log_scale - 1):
                h = d_resnet_block(h, filters << block, downsample=True)
                h = d_resnet_block(h, filters << block)
            h = d_resnet_block(h, filters << (self.log_scale - 1), downsample=True)
            h = d_resnet_block(h, filters << (self.log_scale - 1))
            lr_disc = layers.conv2d_spectral_norm(h, colors, 3, **conv_args) * x_lr
            lr_disc = tf.reduce_sum(lr_disc, [1, 2, 3]) * tf.rsqrt(lr_h * lr_w * lr_c)
            lr_disc = tf.reshape(lr_disc, [-1, 1])
            for block in range(self.log_scale, log_res - 2):
                h = d_resnet_block(h, filters << block, downsample=True)
            h = d_resnet_block(h, filters << block)
            h = tf.reduce_sum(h, [1, 2]) * (1 / 4.)
            hr_disc = layers.dense_spectral_norm(h, 1, kernel_initializer=tf.glorot_uniform_initializer())
            return lr_disc + hr_disc

    def train_step(self, data, ops):
        for _ in range(5):
            x = next(data)
            self.sess.run(ops.train_d, feed_dict={ops.x: x['x']})
        x = next(data)
        self.sess.run(ops.train_g, feed_dict={ops.x: x['x']})

    def model(self, dataset, scale, blocks, filters, noise, decay_start, decay_stop, lr_decay, **kwargs):
        del kwargs
        x = tf.placeholder(tf.float32, [None, dataset.colors, dataset.height, dataset.width], 'x')
        y = tf.placeholder(tf.float32, [None, dataset.colors, None, None], 'y')

        cur_lr = tf.cond(tf.train.get_global_step() < decay_start,
                         lambda: FLAGS.lr,
                         lambda: tf.train.exponential_decay(FLAGS.lr, tf.train.get_global_step() - decay_start,
                                                            decay_stop - decay_start, lr_decay))

        def tower(real):
            real = layers.to_nhwc(real)
            lores = self.downscale(real, order=layers.NHWC)
            fake = self.sres(lores, noise, filters, blocks, train=True)
            disc_real = self.disc(real, lores, dataset.width, filters)
            disc_fake = self.disc(fake, lores, dataset.width, filters)

            loss_dreal = tf.reduce_mean(tf.nn.relu(1 - disc_real))
            loss_dfake = tf.reduce_mean(tf.nn.relu(1 + disc_fake))
            loss_gfake = -tf.reduce_mean(disc_fake)
            mse_ema = tf.losses.mean_squared_error(fake, real)

            return loss_gfake, loss_dreal, loss_dfake, mse_ema

        loss_gfake, loss_dreal, loss_dfake, mse_ema = utils.para_mean(tower, x)
        loss_disc = loss_dreal + loss_dfake
        loss_gen = loss_gfake

        utils.HookReport.log_tensor(cur_lr, 'lr')
        utils.HookReport.log_tensor(loss_dreal, 'dreal')
        utils.HookReport.log_tensor(loss_dfake, 'dfake')
        utils.HookReport.log_tensor(loss_gfake, 'gfake')
        utils.HookReport.log_tensor(tf.sqrt(mse_ema) * 127.5, 'rmse_ema')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_d = tf.train.AdamOptimizer(cur_lr, 0, 0.9).minimize(
                loss_disc, var_list=utils.model_vars('disc'),
                colocate_gradients_with_ops=True)
            train_g = tf.train.AdamOptimizer(cur_lr, 0, 0.9).minimize(
                loss_gen, var_list=utils.model_vars('sres'),
                colocate_gradients_with_ops=True,
                global_step=tf.train.get_global_step())

        def sres_op(y):
            return self.sres(layers.to_nhwc(y), noise, filters, blocks, train=False)

        return EasyDict(x=x, y=y, train_d=train_d, train_g=train_g,
                        sres_op=layers.to_nchw(sres_op(y)),
                        eval_op=layers.to_nchw(sres_op(self.downscale(x))))


def main(argv):
    del argv  # Unused.
    dataset = data.get_dataset(FLAGS.dataset)
    decay_start = (FLAGS.total_kimg << 9) // FLAGS.batch
    decay_stop = (FLAGS.total_kimg << 10) // FLAGS.batch
    model = cGAN(
        os.path.join(FLAGS.train_dir, dataset.name),
        scale=FLAGS.scale,
        downscaler=FLAGS.downscaler,
        blocks=FLAGS.blocks,
        filters=FLAGS.filters,
        noise=FLAGS.noise,
        decay_start=decay_start,
        decay_stop=decay_stop,
        lr_decay=FLAGS.lr_decay)
    if FLAGS.reset:
        model.reset_files()
    model.train(dataset)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_integer('blocks', 4, 'Number of residual blocks in generator.')
    flags.DEFINE_integer('filters', 64, 'Filter size of first convolution.')
    flags.DEFINE_integer('noise', 128, 'Number of noise dimensions.')
    flags.DEFINE_float('lr_decay', 0.1, 'Amount of learning rate decay during last training phase.')
    flags.DEFINE_bool('reset', False, 'Retrain from the start.')
    FLAGS.set_default('batch', 16)
    FLAGS.set_default('lr', 0.0002)
    FLAGS.set_default('total_kimg', 1 << 14)
    app.run(main)
