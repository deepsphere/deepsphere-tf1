#!/usr/bin/env python3
# coding: utf-8

import sys
sys.path.append('../../Experiments/SHREC17/')
from load_shrec import fix_dataset, Shrec17Dataset


if len(sys.argv) > 4:
    data_path = sys.argv[1]
    experiment = sys.argv[2]
    nside = int(sys.argv[3])
    augmentation = int(sys.argv[4])
    download = bool(sys.argv[5])
    # verbose = False
else:
    nside = 32
    augmentation = 0
    data_path = './'
    experiment = 'deepsphere_rot'
    download = True

verbose = True

Shrec17Dataset(data_path, 'train', nside=nside, augmentation=augmentation, download=download,
               experiment=experiment, nfile=None, verbose=verbose, load=False)

Shrec17Dataset(data_path, 'val', nside=nside, augmentation=augmentation, download=download,
               experiment=experiment, nfile=None, verbose=verbose, load=False)

Shrec17Dataset(data_path, 'test', nside=nside, augmentation=augmentation, download=download,
               experiment=experiment, nfile=None, verbose=verbose, load=False)
