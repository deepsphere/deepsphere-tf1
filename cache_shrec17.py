#!/usr/bin/env python3
# coding: utf-8

import sys
from SHREC17.load_shrec import fix_dataset, Shrec17DatasetCache

data_path = '../data/shrec17/'

if len(sys.argv) > 4:
    data_path = sys.argv[1]
    experiment = sys.argv[2]
    nside = int(sys.argv[3])
    augmentation = int(sys.argv[4])
    verbose = False
else:
    nside = 32
    augmentation = 3
    data_path = '../data/shrec17/'
    experiment = 'deepsphere_norot'
    verbose = False

train = Shrec17DatasetCache(data_path, 'train', nside=nside, augmentation=augmentation, experiment = experiment, nfile=None, verbose=verbose)
for elem in train.iter(1):
    pass
#fix_dataset(data_path+'val_perturbed')
val = Shrec17DatasetCache(data_path, 'val', nside=nside, augmentation=augmentation, experiment = experiment, nfile=None, verbose=verbose)
for elem in val.iter(1):
    pass
#fix_dataset(data_path+'test_perturbed')
test = Shrec17DatasetCache(data_path, 'test', nside=nside, augmentation=augmentation, experiment = experiment, nfile=None, verbose=verbose)
for elem in test.iter(1):
    pass