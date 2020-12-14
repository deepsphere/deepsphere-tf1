#!/usr/bin/env python3
# coding: utf-8

"""
Script to run the DeepSphere experiment with ModelNet40 dataset.
"""

import os
import shutil
import sys
sys.path.append('../../')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # change to chosen GPU to use, nothing if work on CPU

from deepsphere import models
from . import hyperparameters

from load_MN40 import ModelNet40DatasetTF, ModelNet40DatasetCache

Nside = 32
exp='norot' # in ['rot', 'norot', 'pert', 'Z']
datapath = '../../data/ModelNet40/' # localisation of the .OFF files
proc_path = datapath[1:]

augmentation = 1        # number of element per file (1 = no augmentation of dataset)
nfeat = 6

experiment = 'deepsphere'+('_rot' if exp == 'rot' else '')+('_Z' if exp == 'Z' else '')+('_notr' if 'pert' not in exp and exp != 'Z' else '')
train_TFDataset = ModelNet40DatasetTF(datapath, 'train', nside=Nside,
                                      nfeat=nfeat, augmentation=augmentation, nfile=None, experiment=experiment)

test_dataset = ModelNet40DatasetCache(datapath, 'test', nside=Nside, nfeat=nfeat, augmentation=1, nfile=None,
                                      experiment=experiment)

nclass = train_TFDataset.nclass
num_elem = train_TFDataset.N
print('number of class:',nclass,'\nnumber of elements:',num_elem)

EXP_NAME = 'MN40_{}_{}feat_{}aug_{}sides'.format(exp, nfeat, augmentation, Nside)

params = hyperparameters.get_params_mn40(train_TFDataset.N, EXP_NAME, Nside, nclass,
                                         nfeat_in=nfeat, architecture='CNN')  # get_params_shrec17_optim
params["tf_dataset"] = train_TFDataset.get_tf_dataset(params["batch_size"])
model = models.deepsphere(**params)

shutil.rmtree('../../summaries/{}/'.format(EXP_NAME), ignore_errors=True)
shutil.rmtree('../../checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)

accuracy_validation, loss_validation, loss_training, t_step, t_batch = model.fit(train_TFDataset,
                                                                                 test_dataset,
                                                                                 use_tf_dataset=True, cache=True)

print(model.evaluate(test_dataset, None, cache=True))
# predictions, loss = model.predict(test_dataset, None, cache=True)
