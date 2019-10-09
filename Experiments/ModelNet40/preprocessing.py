#!/usr/bin/env python3
# coding: utf-8

import sys
import os
import glob
from load_MN40 import ModelNet40DatasetCache, compute_mean_std

# data_path = '../../data/shrec17/'

if len(sys.argv) > 4:
    data_path = sys.argv[1]
    experiment = sys.argv[2]
    nside = int(sys.argv[3])
    augmentation = int(sys.argv[4])
    dl = bool(sys.argv[5])
    # verbose = False
else:
    nside = 32
    augmentation = 3
    data_path = '../../data/mn40/'
    experiment = 'deepsphere_rot'  # in ['rot', 'notr', 'Z']
    dl = True


def _check_exists(path):
    files = glob.glob(os.path.join(path, "*/*.off"))

    return len(files) > 0


def _download(url, path):
    import requests

    filename = url.split('/')[-1]
    file_path = os.path.join(path, filename)

    if os.path.exists(file_path):
        return file_path

    print('Downloading ' + url)

    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return file_path


def _unzip(file_path, path):
    import zipfile

    # if os.path.exists(self.dir):
    #     return

    print('Unzip ' + file_path)

    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(path)
    zip_ref.close()
    os.unlink(file_path)


def download(path):

    if _check_exists(path):
        return

    # download files
    # os.makedirs(self.root)

    url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
    file_path = _download(url, path)
    _unzip(file_path)
    print('Done!')


if dl:
    download(data_path)

train_dataset = ModelNet40DatasetCache(data_path, 'train', nside=nside, augmentation=augmentation, nfile=None,
                                       experiment=experiment, fix=True)

val_dataset = ModelNet40DatasetCache(data_path, 'val', nside=nside, augmentation=augmentation, nfile=None,
                                     experiment=experiment, fix=True)

test_dataset = ModelNet40DatasetCache(data_path, 'test', nside=nside, augmentation=augmentation, nfile=None,
                                      experiment=experiment, fix=True)

compute_mean_std(train_dataset, 'train', data_path, nside)
compute_mean_std(val_dataset, 'train', data_path, nside)
compute_mean_std(test_dataset, 'train', data_path, nside)
