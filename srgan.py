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

For sake of simplicity, I replaced the PReLU with a LeakyReLU.
"""

import os

import tensorflow as tf
from absl import app
from absl import flags
from easydict import EasyDict

from libml import data, layers, utils, vgg
from libml.train_sr import SRES

FLAGS = flags.FLAGS


class SRGAN(SRES):

    def sres(self, x, colors, filters, blocks, train):
        conv_args = dict(padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.02))

        with tf.variable_scope('sres', reuse=tf.AUTO_REUSE):
            y0 = y = tf.layers.conv2d(x, filters, 9, activation=tf.nn.leaky_relu, **conv_args)
            for block in range(blocks):
                dy = tf.layers.conv2d(y, filters, 3, **conv_args)
                dy = tf.nn.leaky_relu(tf.layers.batch_normalization(dy, training=train))
                dy = tf.layers.conv2d(dy, filters, 3, **conv_args)
                dy = tf.layers.batch_normalization(dy, training=train)
                y += dy
            y = tf.layers.conv2d(dy, filters, 3, **conv_args)
            y = tf.layers.batch_normalization(y, training=train)
            y += y0
            for scale in range(self.log_scale):
                y = tf.layers.conv2d(y, filters * 4, 3, activation=tf.nn.leaky_relu, **conv_args)
                y = layers.channels_to_space(y, order=layers.NHWC)
            return tf.layers.conv2d(y, colors, 9, activation=tf.nn.tanh, **conv_args)

    def disc(self, x, resolution, filters):
        conv_args = dict(padding='same', kernel_initializer=tf.random_normal_initializer(stddev=0.02))
        log_res = utils.ilog2(resolution)

        def f(stage):
            return min(filters << stage, 512)

        with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
            y = x
            for r in range(log_res - 2):
                y = tf.layers.conv2d(y, f(r), 3, **conv_args)
                if r > 0:
                    y = tf.layers.batch_normalization(y, training=True)
                y = tf.nn.leaky_relu(y)
                y = tf.layers.conv2d(y, f(r), 3, strides=2, **conv_args)
                y = tf.nn.leaky_relu(tf.layers.batch_normalization(y, training=True))

            # single image = 4 x 4 x (filters << (log(resolution) - 3))
            y = tf.layers.dense(y, 1024, activation=tf.nn.leaky_relu)
            y = tf.layers.dense(y, 1)
            return y

    def model(self, dataset, scale, blocks, filters, adv_weight, pcp_weight, layer_name,
              decay_start, decay_stop, lr_decay, **kwargs):
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
            fake = self.sres(lores, dataset.colors, filters, blocks, train=True)
            disc_real = self.disc(real, dataset.width, filters)
            disc_fake = self.disc(fake, dataset.width, filters)

            with tf.variable_scope('VGG', reuse=tf.AUTO_REUSE):
                vgg19 = vgg.Vgg19()
                real_embed = vgg19.build(layer_name, real, channels_last=True) / 1000
                fake_embed = vgg19.build(layer_name, fake, channels_last=True) / 1000

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

        utils.HookReport.log_tensor(cur_lr, 'lr')
        utils.HookReport.log_tensor(loss_dreal, 'dreal')
        utils.HookReport.log_tensor(loss_dfake, 'dfake')
        utils.HookReport.log_tensor(tf.sqrt(loss_gpcp / loss_gmse), 'grat')
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

        def sres_op(y):
            return self.sres(layers.to_nhwc(y), dataset.colors, filters, blocks, train=False)

        return EasyDict(x=x, y=y, train_op=tf.group(train_d, train_g),
                        sres_op=layers.to_nchw(sres_op(y)),
                        eval_op=layers.to_nchw(sres_op(self.downscale(x))))


def main(argv):
    del argv  # Unused.
    dataset = data.get_dataset(FLAGS.dataset)
    decay_start = (FLAGS.total_kimg << 9) // FLAGS.batch
    decay_stop = (FLAGS.total_kimg << 10) // FLAGS.batch
    model = SRGAN(
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
    flags.DEFINE_float('adv_weight', 0.001, 'Amount of adversarial loss.')
    flags.DEFINE_integer('filters', 64, 'Filter size of first convolution.')
    flags.DEFINE_string('layer_name', 'conv2_2', 'VGG layer to use in perceptual loss.')
    flags.DEFINE_float('lr_decay', 0.1, 'Amount of learning rate decay during last training phase.')
    flags.DEFINE_float('pcp_weight', 1, 'Amount of perceptual loss.')
    flags.DEFINE_integer('blocks', 16, 'Number of residual blocks in generator.')
    flags.DEFINE_bool('reset', False, 'Retrain from the start.')
    FLAGS.set_default('batch', 16)
    FLAGS.set_default('lr', 0.0001)
    FLAGS.set_default('total_kimg', 1 << 14)
    app.run(main)
