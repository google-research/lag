#!/usr/bin/env python

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

"""Script to download all datasets and create .tfrecord files.
"""

import collections
import functools
import gzip
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from urllib import request

import lmdb
import numpy as np
import scipy.io
import tensorflow as tf
from google_drive_downloader import GoogleDriveDownloader as gdd
from tqdm import trange, tqdm

from libml.data import DATA_DIR

URLS = {
    'svhn': 'http://ufldl.stanford.edu/housenumbers/{}_32x32.mat',
    'cifar10': 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz',
    'celeba': dict(images='0B7EVK8r0v71pZjFTYXZWM3FlRnM',
                   identities='1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS',
                   attributes='0B7EVK8r0v71pblRyaVFSWGxPY0U',
                   test_split='0B7EVK8r0v71pY0NSMzRuSXJEVkk'),
    'mnist': 'https://storage.googleapis.com/cvdf-datasets/mnist/{}.gz',
    'superres': {
        'Set14': 'https://drive.google.com/open?id=1nihY72G3ZGVxAIMVVTKOAvFbMYq2GVvg',
        'BSDS200': 'https://drive.google.com/open?id=1hIYAIODXT6GksNuk0EHiwgnVxZlDHUAI',
        'T91': 'https://drive.google.com/open?id=1dfsToAYgecVARKjw2wtQS5tsn6pzG6pr',
        'Set5': 'https://drive.google.com/open?id=1_JSiLOaNtOmoNSG2_s0bSgL6D4dJyp_8',
        'General100': 'https://drive.google.com/open?id=1Y4R8STXyPcOOykutbAJpMsH5O5n2NdFF',
        'urban100': 'https://drive.google.com/open?id=1XaY-tnBP_z21WKgOCeXBa9r-KJyBMbgZ',
        'BSDS100': 'https://drive.google.com/open?id=1EWEsfsgElkNvOcJwZLDe2TeDIMhr6SpH',
        'historical': 'https://drive.google.com/open?id=17Rq-4gm1_rJX3KB2jolcqMGWiSSmQIWz',
        'manga109': 'https://drive.google.com/open?id=15cAVM4BJtSGpduLufqDqfQV75m-Pfepi',
    }
}


def _encode_png(images):
    raw = []
    with tf.Session() as sess, tf.device('cpu:0'):
        image_x = tf.placeholder(tf.uint8, [None, None, None], 'image_x')
        to_png = tf.image.encode_png(image_x)
        for x in trange(images.shape[0], desc='PNG Encoding', leave=False):
            raw.append(sess.run(to_png, feed_dict={image_x: images[x]}))
    return raw


def _load_svhn():
    splits = collections.OrderedDict()
    for split in ['train', 'test', 'extra']:
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['svhn'].format(split), f.name)
            data_dict = scipy.io.loadmat(f.name)
        dataset = {}
        dataset['images'] = np.transpose(data_dict['X'], [3, 0, 1, 2])
        dataset['images'] = _encode_png(dataset['images'])
        dataset['labels'] = data_dict['y'].reshape((-1))
        # SVHN raw data uses labels from 1 to 10; use 0 to 9 instead.
        dataset['labels'] -= 1
        splits[split] = dataset
    return splits


def _load_cifar10():
    def unflatten(images):
        return np.transpose(images.reshape((images.shape[0], 3, 32, 32)),
                            [0, 2, 3, 1])

    with tempfile.NamedTemporaryFile() as f:
        request.urlretrieve(URLS['cifar10'], f.name)
        tar = tarfile.open(fileobj=f)
        train_data_batches, train_data_labels = [], []
        for batch in range(1, 6):
            data_dict = scipy.io.loadmat(tar.extractfile(
                'cifar-10-batches-mat/data_batch_{}.mat'.format(batch)))
            train_data_batches.append(data_dict['data'])
            train_data_labels.append(data_dict['labels'].flatten())
        train_set = {'images': np.concatenate(train_data_batches, axis=0),
                     'labels': np.concatenate(train_data_labels, axis=0)}
        data_dict = scipy.io.loadmat(tar.extractfile(
            'cifar-10-batches-mat/test_batch.mat'))
        test_set = {'images': data_dict['data'],
                    'labels': data_dict['labels'].flatten()}
    train_set['images'] = _encode_png(unflatten(train_set['images']))
    test_set['images'] = _encode_png(unflatten(test_set['images']))
    return dict(train=train_set, test=test_set)


def _load_celeba():
    with tempfile.NamedTemporaryFile() as f_images, tempfile.NamedTemporaryFile() as f_identities, tempfile.NamedTemporaryFile() as f_attributes, tempfile.NamedTemporaryFile() as f_split:
        gdd.download_file_from_google_drive(file_id=URLS['celeba']['identities'], dest_path=f_identities.name,
                                            overwrite=True)
        gdd.download_file_from_google_drive(file_id=URLS['celeba']['attributes'], dest_path=f_attributes.name,
                                            overwrite=True)
        gdd.download_file_from_google_drive(file_id=URLS['celeba']['test_split'], dest_path=f_split.name,
                                            overwrite=True)
        in_test = {}
        for x in f_split:
            name, group = x.decode('ascii').split()
            in_test[name] = int(group) == 2
        attributes = {}
        f_attributes.readline()  # Skip count
        attribute_names = f_attributes.readline().decode('ascii').split()
        for x in f_attributes:
            name, str_attrs = x.decode('ascii').split(maxsplit=1)
            attrs = [(1 + int(v)) // 2 for v in str_attrs.split()]
            attributes[name] = attrs
        identities = {}
        for x in f_identities:
            name, identity = x.decode('ascii').split()
            identities[name] = int(identity) - 1
        gdd.download_file_from_google_drive(file_id=URLS['celeba']['images'], dest_path=f_images.name, overwrite=True)
        if b'Quota exceeded' in f_images.read(1024):
            raise FileNotFoundError('Quota exceeded: File images.zip for CelebA could not be downloaded from'
                                    ' Google drive. Try again later.')
        f_images.seek(0)
        zip_f = zipfile.ZipFile(f_images)
        images = {}
        for image_filename in tqdm(sorted(zip_f.namelist()), 'Decompressing', leave=False):
            if os.path.splitext(image_filename)[1] == '.jpg':
                with zip_f.open(image_filename) as image_f:
                    images[os.path.basename(image_filename)] = image_f.read()

        train_keys = sorted([x for x in images.keys() if not in_test[x]])
        test_keys = sorted([x for x in images.keys() if in_test[x]])

        train_set = dict(images=[images[x] for x in train_keys],
                         labels=np.array([identities[x] for x in train_keys], 'i'),
                         attrs=np.array([attributes[x] for x in train_keys], 'f'))
        test_set = dict(images=[images[x] for x in test_keys],
                        labels=np.array([identities[x] for x in test_keys], 'i'),
                        attrs=np.array([attributes[x] for x in test_keys], 'f'))
        readme = ('# Attribute names:\n'
                  + ' '.join(attribute_names)
                  + '\n' + '# Identity min max:\n'
                  + '%s %s\n' % (min(identities.values()), max(identities.values())))
        return dict(train=train_set, test=test_set, readme=readme)


def _load_mnist():
    def _read32(data):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(data.read(4), dtype=dt)[0]

    image_filename = '{}-images-idx3-ubyte'
    label_filename = '{}-labels-idx1-ubyte'
    split_files = [('train', 'train'), ('test', 't10k')]
    splits = {}
    for split, split_file in split_files:
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(
                URLS['mnist'].format(image_filename.format(split_file)),
                f.name)
            with gzip.GzipFile(fileobj=f, mode='r') as data:
                assert _read32(data) == 2051
                n_images = _read32(data)
                row = _read32(data)
                col = _read32(data)
                images = np.frombuffer(data.read(n_images * row * col), dtype=np.uint8)
                images = images.reshape((n_images, row, col, 1))
        with tempfile.NamedTemporaryFile() as f:
            request.urlretrieve(URLS['mnist'].format(label_filename.format(split_file)), f.name)
            with gzip.GzipFile(fileobj=f, mode='r') as data:
                assert _read32(data) == 2049
                n_labels = _read32(data)
                labels = np.frombuffer(data.read(n_labels), dtype=np.uint8)
        splits[split] = {'images': _encode_png(images), 'labels': labels}
    return splits


def download(url, filename):
    if subprocess.call(['curl', url, '-o', filename]) != 0:
        raise request.HTTPError('Download failed.')


def _load_lsun(category):
    data = {}
    for subset in ('train', 'val'):
        url = 'http://dl.yf.io/lsun/scenes/{category}_{subset}_lmdb.zip'.format(**locals())
        with tempfile.NamedTemporaryFile(delete=False) as f:
            download(url, f.name)
            data[subset] = f.name
    return data


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _compute_checksum(filename):
    m = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 30), b''):
            m.update(chunk)
    return m.hexdigest()


def _save_as_tfrecord(data, filename):
    assert len(data['images']) == len(data['labels'])
    filename = os.path.join(DATA_DIR, filename + '.tfrecord')
    print('Saving dataset:', filename)
    with tf.python_io.TFRecordWriter(filename) as writer:
        for x in trange(len(data['images']), desc='Building records'):
            feat = dict(image=_bytes_feature(data['images'][x]),
                        label=_int64_feature(data['labels'][x]))
            if 'attrs' in data:
                feat['attrs'] = _float_feature(data['attrs'][x])
            record = tf.train.Example(features=tf.train.Features(feature=feat))
            writer.write(record.SerializeToString())
    checksum = _compute_checksum(filename)
    print('Saved:', filename, 'with checksum:', checksum)


def _lmdbzip_to_tfrecord(zip_file, filename):
    filename = os.path.join(DATA_DIR, filename + '.tfrecord')
    zip_f = zipfile.ZipFile(zip_file)
    zip_f.extractall('/tmp')
    lmdb_path = os.path.join('/tmp', zip_f.filelist[0].filename)
    print('Saving dataset:', filename)
    env = lmdb.open(lmdb_path, readahead=True, max_readers=1, readonly=True)
    with env.begin(write=False) as txn, tf.python_io.TFRecordWriter(filename) as writer:
        with txn.cursor() as cursor:
            it = cursor.iternext()
            num_images = env.stat()['entries']
            for _ in trange(num_images, desc='Building records'):
                key, val = next(it)
                feat = dict(label=_int64_feature(0),
                            image=_bytes_feature(val))
                record = tf.train.Example(features=tf.train.Features(feature=feat))
                writer.write(record.SerializeToString())
    checksum = _compute_checksum(filename)
    print('Saved:', filename, 'with checksum:', checksum)
    for zip_entry in zip_f.filelist:
        shutil.rmtree(os.path.join('/tmp', zip_entry.filename), ignore_errors=True)
    os.unlink(zip_file)


def _is_installed(name, checksums):
    for subset, checksum in checksums.items():
        filename = os.path.join(DATA_DIR, '%s-%s.tfrecord' % (name, subset))
        if not os.path.exists(filename):
            return False
        if checksum is not None and _compute_checksum(filename) != checksum:
            print(_compute_checksum(filename), checksum)
            return False
    return True


def _load_superres(dataset):
    folder = os.path.join('superres', dataset)
    with tempfile.NamedTemporaryFile() as f_images:
        url = URLS['superres'][dataset].replace('https://drive.google.com/open?id=', '')
        gdd.download_file_from_google_drive(file_id=url, dest_path=f_images.name, overwrite=True)
        zip_f = zipfile.ZipFile(f_images)
        print(dataset, len(sorted(zip_f.namelist())))
        for x in sorted(zip_f.namelist()):
            print(x)
        files = {}
        for image_filename in tqdm(sorted(zip_f.namelist()), 'Decompressing', leave=False):
            if image_filename.endswith('/'):
                continue
            with zip_f.open(image_filename) as image_f:
                files[os.path.join(folder, os.path.basename(image_filename))] = image_f.read()
    return dict(files=files)


def _save_files(files, *args, **kwargs):
    del args, kwargs
    for folder in frozenset(os.path.dirname(x) for x in files):
        os.makedirs(os.path.join(DATA_DIR, folder), exist_ok=True)
    for filename, contents in files.items():
        with open(os.path.join(DATA_DIR, filename), 'wb') as f:
            f.write(contents)


def _is_installed_folder(name, folder):
    return os.path.exists(os.path.join(DATA_DIR, name, folder))


CONFIGS = dict(
    celeba=dict(loader=_load_celeba,
                checksums=dict(train=None, test=None)),
    cifar10=dict(loader=_load_cifar10,
                 checksums=dict(train=None, test=None)),
    mnist=dict(loader=_load_mnist,
               checksums=dict(train=None, test=None)),
    svhn=dict(loader=_load_svhn,
              checksums=dict(train=None, test=None, extra=None)),
    lsun_bedroom=dict(loader=functools.partial(_load_lsun, 'bedroom'),
                      saver=_lmdbzip_to_tfrecord,
                      checksums=dict(train=None, val=None)),
    lsun_bridge=dict(loader=functools.partial(_load_lsun, 'bridge'),
                     saver=_lmdbzip_to_tfrecord,
                     checksums=dict(train=None, val=None)),
    lsun_church_outdoor=dict(loader=functools.partial(_load_lsun, 'church_outdoor'),
                             saver=_lmdbzip_to_tfrecord,
                             checksums=dict(train=None, val=None)),
    lsun_classroom=dict(loader=functools.partial(_load_lsun, 'classroom'),
                        saver=_lmdbzip_to_tfrecord,
                        checksums=dict(train=None, val=None)),
    lsun_conference_room=dict(loader=functools.partial(_load_lsun, 'conference_room'),
                              saver=_lmdbzip_to_tfrecord,
                              checksums=dict(train=None, val=None)),
    lsun_dining_room=dict(loader=functools.partial(_load_lsun, 'dining_room'),
                          saver=_lmdbzip_to_tfrecord,
                          checksums=dict(train=None, val=None)),
    lsun_kitchen=dict(loader=functools.partial(_load_lsun, 'kitchen'),
                      saver=_lmdbzip_to_tfrecord,
                      checksums=dict(train=None, val=None)),
    lsun_living_room=dict(loader=functools.partial(_load_lsun, 'living_room'),
                          saver=_lmdbzip_to_tfrecord,
                          checksums=dict(train=None, val=None)),
    lsun_restaurant=dict(loader=functools.partial(_load_lsun, 'restaurant'),
                         saver=_lmdbzip_to_tfrecord,
                         checksums=dict(train=None, val=None)),
    lsun_tower=dict(loader=functools.partial(_load_lsun, 'tower'),
                    saver=_lmdbzip_to_tfrecord,
                    checksums=dict(train=None, val=None)),
)
CONFIGS.update({
    'superres_' + x.lower(): dict(loader=functools.partial(_load_superres, x),
                                  saver=_save_files,
                                  is_installed=functools.partial(_is_installed_folder, 'superres', x))
    for x in URLS['superres']
})

if __name__ == '__main__':
    if len(sys.argv[1:]):
        subset = set(sys.argv[1:])
    else:
        subset = set(CONFIGS.keys())
    try:
        os.makedirs(DATA_DIR)
    except OSError:
        pass
    for name, config in CONFIGS.items():
        if name not in subset:
            continue
        if 'is_installed' in config:
            if config['is_installed']():
                print('Skipping already installed:', name)
                continue
        elif _is_installed(name, config['checksums']):
            print('Skipping already installed:', name)
            continue
        print('Preparing', name)
        datas = config['loader']()
        saver = config.get('saver', _save_as_tfrecord)
        for sub_name, data in datas.items():
            if sub_name == 'readme':
                filename = os.path.join(DATA_DIR, '%s-%s.txt' % (name, sub_name))
                with open(filename, 'w') as f:
                    f.write(data)
            else:
                saver(data, '%s-%s' % (name, sub_name))
