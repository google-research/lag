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

"""Training setup."""
import json
import os.path
import shutil

import tensorflow as tf
from absl import flags
from easydict import EasyDict
from tensorflow.core.protobuf import rewriter_config_pb2

from libml import utils
from libml.data import as_iterator

FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', './experiments',
                    'Folder where to save training data.')
flags.DEFINE_float('lr', 0.0001, 'Learning rate.')
flags.DEFINE_integer('batch', 64, 'Batch size.')
flags.DEFINE_string('dataset', 'cifar10', 'Data to train on.')
flags.DEFINE_integer('save_kimg', 64, 'Training duration in samples.')
flags.DEFINE_integer('total_kimg', 1 << 14, 'Training duration in samples.')
flags.DEFINE_integer('report_kimg', 64, 'Training duration in samples.')
flags.DEFINE_bool('log_device_placement', False, 'For debugging purpose.')

FLAGS = flags.FLAGS


class EvalSession:
    def __init__(self, model, checkpoint_dir, **params):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.train.get_or_create_global_step()
            self.ops = model(**params)
            self.sess = tf.Session()
            saver = tf.train.Saver()
            ckpt = utils.find_latest_checkpoint(checkpoint_dir)
            saver.restore(self.sess, ckpt)


class Model:
    def __init__(self, train_dir, **kwargs):
        self.train_dir = os.path.join(train_dir, self.experiment_name(**kwargs))
        self.params = kwargs
        self.eval = None
        self.sess = None
        self.nimg_cur = 0
        self.tmp = EasyDict()
        self.create_initial_files()
        print('Model', self.__class__.__name__)
        print('-' * 80)
        for k, v in sorted(kwargs.items()):
            print('%-32s %s' % (k, v))
        print('-' * 80)

    def create_initial_files(self):
        for dir in (self.checkpoint_dir, self.summary_dir, self.arg_dir):
            if not os.path.exists(dir):
                os.makedirs(dir)
        self.save_args()

    def reset_files(self):
        shutil.rmtree(self.train_dir)
        self.create_initial_files()

    def save_args(self, **extra_params):
        with open(os.path.join(self.arg_dir, 'args.json'), 'w') as f:
            json.dump({**self.params, **extra_params}, f, sort_keys=True, indent=4)

    @classmethod
    def load(cls, train_dir):
        with open(os.path.join(train_dir, 'args/args.json'), 'r') as f:
            params = json.load(f)
        instance = cls(train_dir=train_dir, **params)
        instance.train_dir = train_dir
        return instance

    def experiment_name(self, **kwargs):
        args = [x + str(y) for x, y in sorted(kwargs.items())]
        return '_'.join([self.__class__.__name__] + args)

    @property
    def arg_dir(self):
        return os.path.join(self.train_dir, 'args')

    @property
    def checkpoint_dir(self):
        return os.path.join(self.train_dir, 'tf')

    @property
    def summary_dir(self):
        return os.path.join(self.checkpoint_dir, 'summaries')

    @property
    def tf_sess(self):
        return self.sess._tf_sess()

    def eval_mode(self, **kwargs):
        assert self.eval is None
        self.eval = EvalSession(self.model, self.checkpoint_dir, **self.params, **kwargs)
        print('Eval model %s at global_step %d' % (self.__class__.__name__,
                                                   self.eval.sess.run(self.eval.global_step)))
        return self.eval

    def add_summaries(self, dataset, ops, **kwargs):
        pass  # No default image summaries

    def train_step(self, data, ops):
        x = next(data)
        x, label = x['x'], x['label']
        self.sess.run(ops.train_op, feed_dict={ops.x: x, ops.label: label})

    def train(self, dataset):
        batch = FLAGS.batch

        with dataset.graph.as_default():
            train_data = dataset.train.batch(batch)
            train_data = train_data.prefetch(16)
            train_data = iter(as_iterator(train_data, dataset.sess))

        with tf.Graph().as_default():
            global_step = tf.train.get_or_create_global_step()
            ops = self.model(dataset=dataset,
                             total_steps=(FLAGS.total_kimg << 10) // batch,
                             **self.params)
            self.add_summaries(dataset, ops, **self.params)
            stop_hook = tf.train.StopAtStepHook(
                last_step=1 + (FLAGS.total_kimg << 10) // batch)
            report_hook = utils.HookReport(FLAGS.report_kimg << 10, batch)
            config = tf.ConfigProto()
            if len(utils.get_available_gpus()) > 1:
                config.allow_soft_placement = True
            if FLAGS.log_device_placement:
                config.log_device_placement = True
            config.gpu_options.allow_growth = True

            with tf.train.MonitoredTrainingSession(
                    checkpoint_dir=self.checkpoint_dir,
                    config=config,
                    hooks=[stop_hook],
                    chief_only_hooks=[report_hook],
                    save_checkpoint_secs=600,
                    save_summaries_steps=(FLAGS.save_kimg << 10) // batch) as sess:
                self.sess = sess
                self.nimg_cur = batch * self.tf_sess.run(global_step)
                while not sess.should_stop():
                    self.train_step(train_data, ops)
                    self.nimg_cur = batch * self.tf_sess.run(global_step)


class TrainPhase:
    def __init__(self, nimg_start, nimg_stop, lod_start, lod_stop):
        assert 0 <= lod_stop - lod_start <= 1
        assert nimg_start < nimg_stop
        self.nimg_start = nimg_start
        self.nimg_stop = nimg_stop
        self.lod_start = lod_start
        self.lod_stop = lod_stop

    def lod(self, nimg_cur):
        if self.lod_start == self.lod_stop:
            return self.lod_stop
        return self.lod_start + min(1, max(0, (nimg_cur - self.nimg_start) / float(self.nimg_stop - self.nimg_start)))


class TrainSchedule:
    def __init__(self, resolution_start, resolution_stop, transition_kimg, training_kimg, stop_kimg):
        self.transition_nimg = transition_kimg << 10
        self.training_nimg = training_kimg << 10
        self.lod_start = utils.ilog2(resolution_start)
        self.lod_stop = utils.ilog2(resolution_stop)
        self.schedule = []
        nimg_cur = 0
        for lod in range(self.lod_start, self.lod_stop):
            if training_kimg:
                self.schedule.append(TrainPhase(nimg_cur, nimg_cur + self.training_nimg, lod, lod))
                nimg_cur += self.training_nimg
            if transition_kimg:
                self.schedule.append(TrainPhase(nimg_cur, nimg_cur + self.transition_nimg, lod, lod + 1))
                nimg_cur += self.transition_nimg
        stop_nimg = nimg_cur + self.training_nimg if stop_kimg == 0 else stop_kimg << 10
        if stop_nimg > nimg_cur:
            self.schedule.append(TrainPhase(nimg_cur, stop_nimg, self.lod_stop, self.lod_stop))
        self.schedule[0].nimg_start = 0

    @property
    def lod_min(self):
        return self.schedule[0].lod_start

    @property
    def lod_max(self):
        return self.schedule[-1].lod_stop

    @property
    def total_nimg(self):
        return self.schedule[-1].nimg_stop

    def phase_index(self, nimg_cur):
        for pos, x in enumerate(self.schedule):
            if nimg_cur < x.nimg_stop:
                return pos
        return pos


class ModelPro(Model):
    """Progressive Growing Setup."""

    def train_step(self, data, lod, ops):
        x = next(data)
        x, label = x['x'], x['label']
        self.sess.run(ops.train_op, feed_dict={ops.x: x, ops.label: label, ops.lod: lod})

    def stage_scopes(self, stage):
        """Return all scopes up to `stage`."""
        raise NotImplementedError

    def train(self, dataset, schedule):
        assert isinstance(schedule, TrainSchedule)
        batch = FLAGS.batch
        resume_step = utils.get_latest_global_step_in_subdir(self.checkpoint_dir)
        phase_start = schedule.phase_index(resume_step * batch)
        checkpoint_dir = lambda stage: os.path.join(self.checkpoint_dir, 'stage_%d' % stage)

        for phase in schedule.schedule[phase_start:]:
            print('Resume step %d  Phase %dK:%dK  LOD %d:%d' %
                  (resume_step,
                   phase.nimg_start >> 10, phase.nimg_stop >> 10,
                   phase.lod_start, phase.lod_stop))
            assert isinstance(phase, TrainPhase)

            def lod_fn():
                return phase.lod(self.nimg_cur)

            with dataset.graph.as_default():
                train_data = dataset.train.batch(batch)
                train_data = train_data.prefetch(64)
                train_data = iter(as_iterator(train_data, dataset.sess))

            with tf.Graph().as_default():
                global_step = tf.train.get_or_create_global_step()
                ops = self.model(dataset=dataset,
                                 lod_start=phase.lod_start,
                                 lod_stop=phase.lod_stop,
                                 lod_max=schedule.lod_max,
                                 total_steps=schedule.total_nimg // batch,
                                 **self.params)
                self.add_summaries(dataset, ops, lod_fn, **self.params)
                stop_hook = tf.train.StopAtStepHook(last_step=phase.nimg_stop // batch)
                report_hook = utils.HookReport(FLAGS.report_kimg << 10, batch)
                config = tf.ConfigProto()
                if len(utils.get_available_gpus()) > 1:
                    config.allow_soft_placement = True
                if FLAGS.log_device_placement:
                    config.log_device_placement = True
                config.gpu_options.allow_growth = True
                config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF

                # When growing the model, load the previously trained layer weights.
                stage_step_last = utils.get_latest_global_step(checkpoint_dir(phase.lod_stop - 1))
                stage_step = utils.get_latest_global_step(checkpoint_dir(phase.lod_stop))
                if stage_step_last and not stage_step:
                    last_checkpoint = utils.find_latest_checkpoint(checkpoint_dir(phase.lod_stop - 1))
                    tf.train.init_from_checkpoint(last_checkpoint,
                                                  {x: x for x in self.stage_scopes(phase.lod_stop - 1)})

                with tf.train.MonitoredTrainingSession(
                        checkpoint_dir=checkpoint_dir(phase.lod_stop),
                        config=config,
                        hooks=[stop_hook],
                        chief_only_hooks=[report_hook],
                        save_checkpoint_secs=600,
                        save_summaries_steps=(FLAGS.save_kimg << 10) // batch) as sess:
                    self.sess = sess
                    self.nimg_cur = batch * self.tf_sess.run(global_step)
                    while not sess.should_stop():
                        self.train_step(train_data, lod_fn(), ops)
                        resume_step = self.tf_sess.run(global_step)
                        self.nimg_cur = batch * resume_step

    def add_summaries(self, dataset, ops, lod_fn, **kwargs):
        pass  # No default image summaries
