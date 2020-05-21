# !/usr/bin/env python
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
"""Script to measure time taken to load batches from a dataset.
"""

import tensorflow as tf
from absl import app
from absl import flags
from tqdm import trange

from libml import data

flags.DEFINE_integer('batch', 64, 'Batch size.')
flags.DEFINE_integer('samples', 1 << 16, 'Number of samples to load.')
flags.DEFINE_string('dataset', 'cifar10', 'Dataset to use.')

FLAGS = flags.FLAGS


def main(argv):
    del argv
    nbatch = FLAGS.samples // FLAGS.batch
    dataset = data.get_dataset(FLAGS.dataset)
    train_data = dataset.train.batch(FLAGS.batch)
    train_data = train_data.prefetch(32)
    train_data = train_data.make_one_shot_iterator().get_next()
    with tf.train.MonitoredSession() as sess:
        for _ in trange(nbatch, leave=True):
            sess.run(train_data)


if __name__ == '__main__':
    app.run(main)
