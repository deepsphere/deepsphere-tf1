#!/usr/bin/env python3
# coding: utf-8

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

from SHREC17.load_shrec import Shrec17DeepSphere as shrecDataset
from SHREC17.load_shrec import fix_dataset

Nside = 32
experiment_type = 'CNN' # 'FCN'
ename = '_'+experiment_type
datapath = '../data/shrec17/' # localisation of the .obj files

sigma_noise = 0
augmentation = 1        # number of element per file (1 = no augmentation of dataset)

EXP_NAME = 'shrec17_Cohen_simple_reg_2_{}sides_{}noise{}'.format(Nside, sigma_noise, ename)

train_dataset = shrecDataset(datapath, 'train', nside=Nside, augmentation=augmentation, nfile=None)
val_dataset = shrecDataset(datapath, 'val', nside=Nside, augmentation=augmentation, nfile=None)

x_train, labels_train, ids_train = train_dataset.return_data(train=True, train_ratio=1., verbose=False)
x_val, labels_val, ids_val = val_dataset.return_data(train=False, verbose=False)

training = LabeledDataset(x_train, labels_train)
validation = LabeledDataset(x_val, labels_val)

params = hyperparameters.get_params_shrec17(training.N, EXP_NAME, Nside, train_dataset.nclass, architecture=experiment_type, verbose=False)
model = models.deepsphere(**params)

shutil.rmtree('summaries/{}/'.format(EXP_NAME), ignore_errors=True)
shutil.rmtree('checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)

accuracy_validation, loss_validation, loss_training, t_step = model.fit(training, validation, verbose=False)