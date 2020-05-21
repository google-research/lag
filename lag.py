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

"""Latent-Adversarial Generator.
"""

import functools
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from easydict import EasyDict

from libml import data, layers, utils
from libml.layers import conv2d_scaled
from libml.train import TrainSchedule
from libml.train_sr import SRESPro

FLAGS = flags.FLAGS


class LAG(SRESPro):

    def stage_scopes(self, stage):
        return (['global_step']
                + ['opt_%d/' % x for x in range(stage + 1)]
                + ['sres/stage_%d/' % x for x in range(stage + 1)]
                + ['disc/stage_%d/' % x for x in range(stage + 1)])

    def sres(self, x0, colors, lod, lod_min, lod_start, lod_stop, blocks, lfilters, ema=None):
        getter = functools.partial(utils.getter_ema, ema) if ema else None
        scope_args = dict(custom_getter=getter, reuse=tf.AUTO_REUSE)
        lrelu_args = dict(activation=tf.nn.leaky_relu)
        relu_args = dict(activation=tf.nn.relu)

        with tf.variable_scope('sres', **scope_args):
            with tf.variable_scope('stage_0', **scope_args):
                y = conv2d_scaled(x0, lfilters[0], 3)
                for x in range(blocks):
                    dy = conv2d_scaled(y, lfilters[0], 3, **relu_args)
                    y += conv2d_scaled(dy, lfilters[0], 3) / blocks
            rgb = []
            for stage in range(lod_min, lod_stop + 1):
                with tf.variable_scope('stage_%d' % stage, **scope_args):
                    y = layers.upscale2d(y)
                    y = conv2d_scaled(y, lfilters[stage], 3, **lrelu_args)
                    y = conv2d_scaled(y, lfilters[stage], 3, **lrelu_args)
                    with tf.variable_scope('to_rgb', **scope_args):
                        rgb.append(conv2d_scaled(y, colors, 3))
            im = rgb.pop(0)
            for stage in range(lod_min + 1, lod_start + 1):
                im = layers.upscale2d(im) + rgb.pop(0)
            if lod_start == lod_stop:
                return im
            return layers.upscale2d(im) + (lod - lod_start) * rgb[-1]

    def disc(self, x0, x0_lores_delta, lod, lod_min, lod_start, lod_stop, blocks, lfilters):
        leaky_relu = dict(activation=tf.nn.leaky_relu)

        def from_rgb(x, stage):
            with tf.variable_scope('from_rgb', reuse=tf.AUTO_REUSE):
                return conv2d_scaled(x, lfilters[stage], 3, **leaky_relu)

        with tf.variable_scope('disc', reuse=tf.AUTO_REUSE):
            y = None
            for stage in range(lod_stop, lod_min - 1, -1):
                with tf.variable_scope('stage_%d' % stage, reuse=tf.AUTO_REUSE):
                    if stage == lod_stop:
                        y = from_rgb(x0, stage)
                    elif stage == lod_start:
                        y0 = from_rgb(layers.downscale2d(x0), stage)
                        y = y0 + (lod - lod_start) * y
                    else:
                        y += from_rgb(layers.downscale2d(x0, 1 << (lod_stop - stage)), stage)
                    y = conv2d_scaled(y, lfilters[stage], 3, **leaky_relu)
                    y = layers.space_to_channels(y)
                    y = conv2d_scaled(y, lfilters[stage - 1], 3, **leaky_relu)
            y = tf.concat([y, x0_lores_delta], axis=1)
            with tf.variable_scope('stage_0', reuse=tf.AUTO_REUSE):
                for x in range(blocks):
                    y = conv2d_scaled(y, lfilters[0], 3, **leaky_relu)
                center = np.ones(lfilters[0], 'f')
                center[::2] = -1
                center = tf.constant(center, shape=[1, lfilters[0], 1, 1])
        return y * center

    def model(self, dataset, lod_min, lod_max, lod_start, lod_stop, scale, blocks, filters, filters_min,
              wass_target, weight_avg, mse_weight, noise_dim, ttur, total_steps, **kwargs):
        assert lod_min == 1
        del kwargs
        x = tf.placeholder(tf.float32, [None, dataset.colors, dataset.height, dataset.width], 'x')
        y = tf.placeholder(tf.float32, [None, dataset.colors, None, None], 'y')
        noise = tf.placeholder(tf.float32, [], 'noise')
        lod = tf.placeholder(tf.float32, [], 'lod')
        lfilters = [max(filters_min, filters >> stage) for stage in range(lod_max + 1)]
        disc = functools.partial(self.disc, lod=lod, lod_min=lod_min, lod_start=lod_start, lod_stop=lod_stop,
                                 blocks=blocks, lfilters=lfilters)
        sres = functools.partial(self.sres, lod=lod, lod_min=lod_min, lod_start=lod_start, lod_stop=lod_stop,
                                 blocks=blocks, lfilters=lfilters, colors=dataset.colors)
        ema = tf.train.ExponentialMovingAverage(decay=weight_avg) if weight_avg > 0 else None

        def pad_shape(x):
            return [tf.shape(x)[0], noise_dim, tf.shape(x)[2], tf.shape(x)[3]]

        def straight_through_round(x, r=127.5 / 4):
            xr = tf.round(x * r) / r
            return tf.stop_gradient(xr - x) + x

        def sres_op(y, noise):
            eps = tf.random_normal(pad_shape(y), stddev=noise)
            sres_op = sres(tf.concat([y, eps], axis=1), ema=ema)
            sres_op = layers.upscale2d(sres_op, 1 << (lod_max - lod_stop))
            return sres_op

        def tower(x):
            lores = self.downscale(x)
            real = layers.downscale2d(x, 1 << (lod_max - lod_stop))
            if lod_start != lod_stop:
                real = layers.blend_resolution(layers.remove_details2d(real), real, lod - lod_start)

            eps = tf.random_normal(pad_shape(lores))
            fake = sres(tf.concat([lores, tf.zeros_like(eps)], axis=1))
            fake_eps = sres(tf.concat([lores, eps], axis=1))
            lores_fake = self.downscale(layers.upscale2d(fake, 1 << (lod_max - lod_stop)))
            lores_fake_eps = self.downscale(layers.upscale2d(fake_eps, 1 << (lod_max - lod_stop)))
            latent_real = disc(real, straight_through_round(tf.abs(lores - lores)))
            latent_fake = disc(fake, straight_through_round(tf.abs(lores - lores_fake)))
            latent_fake_eps = disc(fake_eps, straight_through_round(tf.abs(lores - lores_fake_eps)))

            # Gradient penalty.
            mix = tf.random_uniform([tf.shape(real)[0], 1, 1, 1], 0., 1.)
            mixed = real + mix * (fake_eps - real)
            mixed = layers.upscale2d(mixed, 1 << (lod_max - lod_stop))
            mixed_round = straight_through_round(tf.abs(lores - self.downscale(mixed)))
            mixdown = layers.downscale2d(mixed, 1 << (lod_max - lod_stop))
            grad = tf.gradients(tf.reduce_sum(tf.reduce_mean(disc(mixdown, mixed_round), 1)), [mixed])[0]
            grad_norm = tf.sqrt(tf.reduce_mean(tf.square(grad), axis=[1, 2, 3]) + 1e-8)

            loss_dreal = -tf.reduce_mean(latent_real)
            loss_dfake = tf.reduce_mean(latent_fake_eps)
            loss_gfake = -tf.reduce_mean(latent_fake_eps)
            loss_gmse = tf.losses.mean_squared_error(latent_real, latent_fake)
            loss_gp = 10 * tf.reduce_mean(tf.square(grad_norm - wass_target)) * wass_target ** -2
            mse_ema = tf.losses.mean_squared_error(sres(tf.concat([lores, tf.zeros_like(eps)], axis=1), ema=ema), real)

            return loss_gmse, loss_gfake, loss_dreal, loss_dfake, loss_gp, mse_ema

        loss_gmse, loss_gfake, loss_dreal, loss_dfake, loss_gp, mse_ema = utils.para_mean(tower, x)
        loss_disc = loss_dreal + loss_dfake + loss_gp
        loss_gen = loss_gfake + mse_weight * loss_gmse

        utils.HookReport.log_tensor(loss_dreal, 'dreal')
        utils.HookReport.log_tensor(loss_dfake, 'dfake')
        utils.HookReport.log_tensor(loss_gp, 'gp')
        utils.HookReport.log_tensor(loss_gfake, 'gfake')
        utils.HookReport.log_tensor(loss_gmse, 'gmse')
        utils.HookReport.log_tensor(tf.sqrt(mse_ema) * 127.5, 'rmse_ema')
        utils.HookReport.log_tensor(lod, 'lod')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_d, train_g = [], []
            global_arg = dict(global_step=tf.train.get_global_step())
            for stage in range(lod_stop + 1):
                g_arg = global_arg if stage == 0 else {}
                with tf.variable_scope('opt_%d' % stage):
                    train_d.append(tf.train.AdamOptimizer(FLAGS.lr, 0, 0.99).minimize(
                        loss_disc * ttur, var_list=utils.model_vars('disc/stage_%d' % stage),
                        colocate_gradients_with_ops=True))
                    train_g.append(tf.train.AdamOptimizer(FLAGS.lr, 0, 0.99).minimize(
                        loss_gen, var_list=utils.model_vars('sres/stage_%d' % stage),
                        colocate_gradients_with_ops=True, **g_arg))

        if ema is not None:
            ema_op = ema.apply(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'sres'))
            train_op = tf.group(*train_d, *train_g, ema_op)
        else:
            train_op = tf.group(*train_d, *train_g)

        return EasyDict(x=x, y=y, noise=noise, lod=lod, train_op=train_op,
                        downscale_op=self.downscale(x),
                        upscale_op=layers.upscale2d(y, self.scale, order=layers.NCHW),
                        sres_op=sres_op(y, noise),
                        eval_op=sres_op(self.downscale(x), 0))


def main(argv):
    del argv  # Unused.
    dataset = data.get_dataset(FLAGS.dataset)
    schedule = TrainSchedule(2, FLAGS.scale, FLAGS.transition_kimg, FLAGS.training_kimg, FLAGS.total_kimg)
    if FLAGS.memtest:
        schedule.schedule = schedule.schedule[-2:]

    model = LAG(
        os.path.join(FLAGS.train_dir, dataset.name),
        lr=FLAGS.lr,
        batch=FLAGS.batch,
        lod_min=1,
        scale=FLAGS.scale,
        downscaler=FLAGS.downscaler,

        blocks=FLAGS.blocks,
        filters=FLAGS.filters,
        filters_min=FLAGS.filters_min,
        mse_weight=FLAGS.mse_weight,
        noise_dim=FLAGS.noise_dim,
        transition_kimg=FLAGS.transition_kimg,
        training_kimg=FLAGS.training_kimg,
        ttur=FLAGS.ttur,
        wass_target=FLAGS.wass_target,
        weight_avg=FLAGS.weight_avg)
    if FLAGS.reset:
        model.reset_files()
    model.train(dataset, schedule)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_integer('blocks', 8, 'Number of residual layers in residual networks.')
    flags.DEFINE_integer('filters', 256, 'Filter size of first convolution.')
    flags.DEFINE_integer('filters_min', 64, 'Minimum filter size of convolution.')
    flags.DEFINE_integer('noise_dim', 64, 'Number of noise dimensions to concat to lores.')
    flags.DEFINE_integer('transition_kimg', 2048, 'Number of images during transition (in kimg).')
    flags.DEFINE_integer('training_kimg', 2048, 'Number of images during between transitions (in kimg).')
    flags.DEFINE_integer('ttur', 4, 'How much faster D is trained.')
    flags.DEFINE_float('wass_target', 1, 'Wasserstein gradient penalty target value.')
    flags.DEFINE_float('weight_avg', 0.999, 'Weight averaging.')
    flags.DEFINE_float('mse_weight', 10, 'Amount of mean square error loss for G.')
    flags.DEFINE_bool('reset', False, 'Retrain from the start.')
    flags.DEFINE_bool('memtest', False, 'Test if the parameters fit in memory (start at last stage).')
    FLAGS.set_default('batch', 16)
    FLAGS.set_default('lr', 0.001)
    FLAGS.set_default('total_kimg', 0)
    app.run(main)
