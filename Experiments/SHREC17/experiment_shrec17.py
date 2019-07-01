#!/usr/bin/env python3
# coding: utf-8

"""
Script to run the DeepSphere experiment with SHREC17 dataset.
"""

import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # change to chosen GPU to use, nothing if work on CPU

import numpy as np
import time
import matplotlib.pyplot as plt
import healpy as hp

from deepsphere import models, experiment_helper, plot, utils
from deepsphere.data import LabeledDatasetWithNoise, LabeledDataset
import hyperparameters

from SHREC17.load_shrec import Shrec17Dataset, Shrec17DatasetCache, Shrec17DatasetTF, fix_dataset

Nside = 128
experiment_type = 'CNN'
ename = '_'+experiment_type
datapath = '../data/shrec17/' # localisation of the .obj files

augmentation = 3        # number of element per file (1 = no augmentation of dataset)
nfeat = 6

EXP_NAME = 'shrec17_best_4K_cache_{}aug_{}sides{}'.format(augmentation, Nside, ename)

train_dataset = Shrec17DatasetTF(datapath, 'train', nside=Nside, augmentation=augmentation, nfile=None, nfeat=nfeat, verbose=False)
val_dataset = Shrec17DatasetCache(datapath, 'val', nside=Nside, augmentation=1, nfile=None, nfeat=nfeat, verbose=False)

nclass = train_dataset.nclass

#x_train, labels_train, ids_train = train_dataset.return_data(train=True, train_ratio=1., verbose=False)
#x_val, labels_val, ids_val = val_dataset.return_data(train=False, verbose=False)


#training = LabeledDataset(x_train, labels_train)
#validation = LabeledDataset(x_val, labels_val)

params = hyperparameters.get_params_shrec17_optim(train_dataset.N, EXP_NAME, Nside, nclass, nfeat_in=nfeat, architecture=experiment_type, verbose=False)
params["tf_dataset"] = train_dataset.get_tf_dataset(params["batch_size"])
model = models.deepsphere(**params)

shutil.rmtree('summaries/{}/'.format(EXP_NAME), ignore_errors=True)
shutil.rmtree('checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)

accuracy_validation, loss_validation, loss_training, t_step, t_batch = model.fit(train_dataset, val_dataset, 
                                                                        verbose=False, cache=True,use_tf_dataset=True)
filepath = 'shrec17_results_4K_{}aug_{}sides{}'.format(augmentation, Nside, ename)
results = [loss_validation, loss_training, t_step, t_batch]
np.savez(filepath, data=results)