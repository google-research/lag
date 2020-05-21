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

"""SRGAN reimplementation.

Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
https://arxiv.org/abs/1609.04802

This is a modified version inspired by github re-implementations that seem to work better than the original.
"""

import os

import tensorflow as tf
from absl import app
from absl import flags
from easydict import EasyDict

from libml import data, layers, utils, vgg
from libml.train_sr import SRES

FLAGS = flags.FLAGS


class SRGAN2(SRES):
    def model(self, dataset, scale, blocks, filters, decay_start, decay_stop, lr_decay,
              adv_weight, pcp_weight, layer_name, **kwargs):
        del kwargs
        x = tf.placeholder(tf.float32, [None, dataset.colors, dataset.height, dataset.width], 'x')
        y = tf.placeholder(tf.float32, [None, dataset.colors, None, None], 'y')

        log_scale = utils.ilog2(scale)
        cur_lr = tf.train.exponential_decay(FLAGS.lr, tf.train.get_global_step() - decay_start,
                                            decay_stop - decay_start, lr_decay)
        utils.HookReport.log_tensor(cur_lr, 'lr')

        def sres(x0, train):
            conv_args = dict(padding='same', data_format='channels_first',
                             kernel_initializer=tf.random_normal_initializer(stddev=0.02))

            with tf.variable_scope("sres", reuse=tf.AUTO_REUSE) as vs:
                x1 = x = tf.layers.conv2d(x0, filters, 3, activation=tf.nn.relu, **conv_args)

                # Residuals
                for i in range(blocks):
                    xb = tf.layers.conv2d(x, filters, 3, **conv_args)
                    xb = tf.layers.batch_normalization(xb, axis=1, training=train)
                    xb = tf.nn.relu(xb)
                    xb = tf.layers.conv2d(xb, filters, 3, **conv_args)
                    xb = tf.layers.batch_normalization(xb, axis=1, training=train)
                    x += xb

                x = tf.layers.conv2d(x, filters, 3, **conv_args)
                x = tf.layers.batch_normalization(x, axis=1, training=train)
                x += x1

                # Upsampling
                for _ in range(log_scale):
                    x = tf.layers.conv2d(x, filters * 4, 3, activation=tf.nn.relu, **conv_args)
                    x = layers.channels_to_space(x)

                x = tf.layers.conv2d(x, x0.shape[1], 1, activation=tf.nn.tanh, **conv_args)
                return x

        def disc(x):
            conv_args = dict(padding='same', data_format='channels_first',
                             kernel_initializer=tf.random_normal_initializer(stddev=0.02))

            with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
                y = tf.layers.conv2d(x, filters, 4, strides=2, activation=tf.nn.leaky_relu, **conv_args)
                y = tf.layers.conv2d(y, filters * 2, 4, strides=2, **conv_args)
                y = tf.nn.leaky_relu(tf.layers.batch_normalization(y, axis=1, training=True))
                y = tf.layers.conv2d(y, filters * 4, 4, strides=2, **conv_args)
                y = tf.nn.leaky_relu(tf.layers.batch_normalization(y, axis=1, training=True))
                y = tf.layers.conv2d(y, filters * 8, 4, strides=2, **conv_args)
                y = tf.nn.leaky_relu(tf.layers.batch_normalization(y, axis=1, training=True))
                if dataset.width > 32:
                    y = tf.layers.conv2d(y, filters * 16, 4, strides=2, **conv_args)
                    y = tf.nn.leaky_relu(tf.layers.batch_normalization(y, axis=1, training=True))
                if dataset.width > 64:
                    y = tf.layers.conv2d(y, filters * 32, 4, strides=2, **conv_args)
                    y = tf.nn.leaky_relu(tf.layers.batch_normalization(y, axis=1, training=True))
                    y = tf.layers.conv2d(y, filters * 16, 1, **conv_args)
                    y = tf.nn.leaky_relu(tf.layers.batch_normalization(y, axis=1, training=True))
                if dataset.width > 32:
                    y = tf.layers.conv2d(y, filters * 8, 1, **conv_args)
                    y = tf.nn.leaky_relu(tf.layers.batch_normalization(y, axis=1, training=True))
                y7 = y
                y = tf.layers.conv2d(y, filters * 2, 1, **conv_args)
                y = tf.nn.leaky_relu(tf.layers.batch_normalization(y, axis=1, training=True))
                y = tf.layers.conv2d(y, filters * 2, 3, **conv_args)
                y = tf.nn.leaky_relu(tf.layers.batch_normalization(y, axis=1, training=True))
                y = tf.layers.conv2d(y, filters * 8, 3, **conv_args)
                y8 = tf.nn.leaky_relu(y7 + tf.layers.batch_normalization(y, axis=1, training=True))
                logits = tf.layers.conv2d(y8, 1, 3, **conv_args)
                return tf.reshape(logits, [-1, 1])

        def tower(real):
            lores = self.downscale(real)
            fake = sres(lores, True)
            disc_real = disc(real)
            disc_fake = disc(fake)

            with tf.variable_scope('VGG', reuse=tf.AUTO_REUSE):
                vgg19 = vgg.Vgg19()
                real_embed = vgg19.build(layer_name, real, channels_last=False)
                fake_embed = vgg19.build(layer_name, fake, channels_last=False)

            loss_gmse = tf.losses.mean_squared_error(fake, real)
            loss_gpcp = tf.losses.mean_squared_error(real_embed, fake_embed)
            loss_ggan = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake))
            loss_dreal = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real))
            loss_dfake = tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake))
            return (loss_gmse, loss_gpcp,
                    tf.reduce_mean(loss_ggan), tf.reduce_mean(loss_dreal), tf.reduce_mean(loss_dfake))

        loss_gmse, loss_gpcp, loss_ggan, loss_dreal, loss_dfake = utils.para_mean(tower, x)
        loss_disc = loss_dreal + loss_dfake
        loss_gen = (loss_gmse
                    + pcp_weight * loss_gpcp
                    + adv_weight * loss_ggan)

        utils.HookReport.log_tensor(loss_dreal, 'dreal')
        utils.HookReport.log_tensor(loss_dfake, 'dfake')
        utils.HookReport.log_tensor(loss_gmse, 'gmse')
        utils.HookReport.log_tensor(pcp_weight * loss_gpcp, 'gpcp')
        utils.HookReport.log_tensor(adv_weight * loss_ggan, 'ggan')
        utils.HookReport.log_tensor(loss_gen, 'gen')
        utils.HookReport.log_tensor(tf.sqrt(loss_gmse) * 127.5, 'rmse')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_d = tf.train.AdamOptimizer(cur_lr, 0.9).minimize(
                loss_disc, var_list=utils.model_vars('disc'),
                colocate_gradients_with_ops=True)
            train_g = tf.train.AdamOptimizer(cur_lr, 0.9).minimize(
                loss_gen, var_list=utils.model_vars('sres'),
                colocate_gradients_with_ops=True,
                global_step=tf.train.get_global_step())

        return EasyDict(x=x, y=y, sres_op=sres(y, False), eval_op=sres(self.downscale(x), False),
                        train_op=tf.group(train_d, train_g))


def main(argv):
    del argv  # Unused.
    dataset = data.get_dataset(FLAGS.dataset)
    decay_start = (FLAGS.total_kimg << 9) // FLAGS.batch
    decay_stop = (FLAGS.total_kimg << 10) // FLAGS.batch
    model = SRGAN2(
        os.path.join(FLAGS.train_dir, dataset.name),
        scale=FLAGS.scale,
        downscaler=FLAGS.downscaler,
        filters=FLAGS.filters,
        blocks=FLAGS.blocks,
        decay_start=decay_start,
        decay_stop=decay_stop,
        lr_decay=FLAGS.lr_decay,
        adv_weight=FLAGS.adv_weight,
        pcp_weight=FLAGS.pcp_weight,
        layer_name=FLAGS.layer_name)
    if FLAGS.reset:
        model.reset_files()
    model.train(dataset)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('adv_weight', 0.01, 'Amount of adversarial loss.')
    flags.DEFINE_integer('filters', 64, 'Filter size of first convolution.')
    flags.DEFINE_string('layer_name', 'conv2_2', 'VGG layer to use in perceptual loss.')
    flags.DEFINE_float('lr_decay', 0.1, 'Amount of learning rate decay during last training phase.')
    flags.DEFINE_float('pcp_weight', 2e-6, 'Amount of perceptual loss.')
    flags.DEFINE_integer('blocks', 16, 'Number of residual layers in residual networks.')
    flags.DEFINE_bool('reset', False, 'Retrain from the start.')
    FLAGS.set_default('batch', 16)
    FLAGS.set_default('lr', 0.0001)
    FLAGS.set_default('total_kimg', 1 << 14)
    app.run(main)
