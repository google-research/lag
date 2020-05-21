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

"""Super-resolution Training model.

This an import only file to provide training helpers.
"""
import functools
import os

import numpy as np
import tensorflow as tf
from absl import flags

from libml import utils, layers
from libml.data import as_iterator
from libml.train import Model, FLAGS, ModelPro

flags.DEFINE_integer('scale', 4, 'Scale by which to increase resolution.')
flags.DEFINE_string('downscaler', 'average', 'Downscaling method [average, bicubic].')


class EvalSessionPro:
    def __init__(self, model, checkpoint_dir, **params):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.train.get_or_create_global_step()
            self.ops = model(**params)
            ckpt = utils.find_latest_checkpoint(checkpoint_dir, 'stage*/model.ckpt-*.meta')
            self.sess = tf.train.SingularMonitoredSession(checkpoint_filename_with_path=ckpt)


class SRES(Model):
    """Super-Resolution base class."""

    def __init__(self, train_dir, scale, downscaler, **kwargs):
        self.scale = scale
        self.downscaler = downscaler
        Model.__init__(self, train_dir, scale=scale, downscaler=downscaler, **kwargs)

    def experiment_name(self, **kwargs):
        args = [x + str(y) for x, y in sorted(kwargs.items()) if x not in {'scale', 'downscaler'}]
        return os.path.join('%s%dX' % (self.downscaler, self.scale), '_'.join([self.__class__.__name__] + args))

    @property
    def log_scale(self):
        return utils.ilog2(self.scale)

    def downscale(self, x, scale=None, order=layers.NCHW):
        scale = scale or self.scale
        if scale <= 1:
            return x
        if self.downscaler == 'average':
            return layers.downscale2d(x, scale, order)
        elif self.downscaler == 'bicubic':
            return layers.bicubic_downscale2d(x, scale, order)
        else:
            raise ValueError('Unknown downscaler "%s"' % self.downscaler)

    def train_step(self, data, ops):
        x = next(data)
        self.sess.run(ops.train_op, feed_dict={ops.x: x['x']})

    def make_samples(self, dataset, input_op, sres_op, batch=1, width=8, height=16, feed_extra=None):
        if 'test_hires' not in self.tmp:
            with dataset.graph.as_default():
                it = iter(as_iterator(dataset.test.batch(width * height).take(1).repeat(), dataset.sess))
                self.tmp.test_hires = next(it)['x']

        hires = self.tmp.test_hires.copy()
        with tf.Graph().as_default(), tf.Session() as sess_new:
            lores = sess_new.run(self.downscale(hires))
            pixelated = sess_new.run(layers.upscale2d(lores, self.scale))

        images = np.concatenate(
            [
                self.tf_sess.run(sres_op, feed_dict={
                    input_op: lores[x:x + batch], **(feed_extra or {})})
                for x in range(0, lores.shape[0], batch)
            ], axis=0)
        images = images.clip(-1, 1)
        images = np.concatenate([hires, pixelated, images], axis=3)
        images = utils.images_to_grid(images.reshape((height, width) + images.shape[1:]))
        return images

    def add_summaries(self, dataset, ops, feed_extra=None, **kwargs):
        del kwargs
        feed_extra = feed_extra.copy() if feed_extra else {}
        if 'noise' in ops:
            feed_extra[ops.noise] = 0

        def gen_images():
            samples = self.make_samples(dataset, ops.y, ops.sres_op, FLAGS.batch, feed_extra=feed_extra)
            # Prevent summary scaling, force offset/ratio = 0/1
            samples[-1, -1] = (-1, 0, 1)
            return samples

        samples = tf.py_func(gen_images, [], [tf.float32])
        tf.summary.image('samples', samples)

    def model(self, latent, **kwargs):
        raise NotImplementedError


class SRESPro(ModelPro, SRES):
    """Progressive Super-Resolution Setup."""

    def eval_mode(self, dataset):
        assert self.eval is None
        log_scale = utils.ilog2(self.scale)
        model = functools.partial(self.model, dataset=dataset, total_steps=1,
                                  lod_start=log_scale, lod_stop=log_scale, lod_max=log_scale)
        self.eval = EvalSessionPro(model, self.checkpoint_dir, **self.params)
        print('Eval model %s at global_step %d' % (self.__class__.__name__,
                                                   self.eval.sess.run(self.eval.global_step)))
        return self.eval

    def train_step(self, data, lod, ops):
        x = next(data)
        self.sess.run(ops.train_op, feed_dict={ops.x: x['x'], ops.lod: lod})

    def add_summaries(self, dataset, ops, lod_fn, **kwargs):
        del kwargs

        def gen_images():
            feed_extra = {ops.lod: lod_fn()}
            if 'noise' in ops:
                feed_extra[ops.noise] = 0
            samples = self.make_samples(dataset, ops.y, ops.sres_op, FLAGS.batch, feed_extra=feed_extra)
            # Prevent summary scaling, force offset/ratio = 0/1
            samples[-1, -1] = (-1, 0, 1)
            return samples

        samples = tf.py_func(gen_images, [], [tf.float32])
        tf.summary.image('samples', samples)
