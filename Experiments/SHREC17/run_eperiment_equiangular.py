#!/usr/bin/env python3
# coding: utf-8

"""
Script to run the DeepSphere experiment with SHREC17 dataset.
"""

import os
import shutil
import sys
sys.append('../..')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # change to chosen GPU to use, nothing if work on CPU

import numpy as np

from deepsphere import models
from . import hyperparameters

from load_shrec import Shrec17DatasetCache, Shrec17DatasetTF, fix_dataset

bandwidth = 64
experiment = 'equiangular'
experiment_type = 'CNN'
ename = '_'+experiment_type
datapath = '../../data/shrec17/' # localisation of the .obj files

augmentation = 3        # number of element per file (1 = no augmentation of dataset)
nfeat = 6

EXP_NAME = 'shrec17_equiangular_{}aug_{}bw{}'.format(augmentation, bandwidth, ename)

train_dataset = Shrec17DatasetTF(datapath, 'train', nside=bandwidth, augmentation=augmentation, nfile=None,
                                 nfeat=nfeat, experiment=experiment, verbose=False)
val_dataset = Shrec17DatasetCache(datapath, 'val', nside=bandwidth, augmentation=1, nfile=None, nfeat=nfeat,
                                  experiment=experiment, verbose=False)

nclass = train_dataset.nclass
num_elem = train_dataset.N
print('number of class:',nclass,'\nnumber of elements:',num_elem)


params = hyperparameters.get_params_shrec17_equiangular(num_elem, EXP_NAME, nclass, architecture=experiment_type)
params["tf_dataset"] = train_dataset.get_tf_dataset(params["batch_size"])
model = models.deepsphere(**params)

shutil.rmtree('summaries/{}/'.format(EXP_NAME), ignore_errors=True)
shutil.rmtree('checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)

accuracy_validation, loss_validation, loss_training, t_step, t_batch = model.fit(train_dataset, val_dataset, 
                                                                        verbose=False, cache=True,use_tf_dataset=True)

print(model.evaluate(val_dataset, None, cache=True))
probabilities,_ = model.probs(val_dataset, nclass, cache=True)
predictions = np.argmax(probabilities, axis=1)
ids_val = val_dataset.get_ids()
os.makedirs(os.path.join(datapath,'results_equiangular/val_perturbed'), exist_ok=True)
for i,_id in enumerate(ids_val):
    idfile = os.path.join(datapath,'results_equiangular/val_perturbed',_id)
    # predictions batchxclass
    # pred_class batch == predictions
    retrieved = [(probabilities[j, predictions[j]], ids_val[j]) for j in range(len(ids_val)) if predictions[j] == predictions[i]]
    retrieved = sorted(retrieved, reverse=True)
    retrieved = [i for _, i in retrieved]
    with open(idfile, "w") as f:
        f.write("\n".join(retrieved))

test_dataset = Shrec17DatasetCache(datapath, 'test', nside=bandwidth,
                                   augmentation=augmentation, nfile=None, experiment=experiment)
print(model.evaluate(test_dataset, None, cache=True))
ids_test = test_dataset.get_ids()
probabilities,_ = model.probs(test_dataset, nclass, cache=True)
if augmentation>1:
    probabilities = probabilities.reshape((-1,augmentation,nclass))
    probabilities = probabilities.mean(axis=1)
    ids_test = ids_test[::augmentation]
predictions = np.argmax(probabilities, axis=1)
os.makedirs(os.path.join(datapath,'results_equiangular/test_perturbed'), exist_ok=True)
for i, _id in enumerate(ids_test):
    idfile = os.path.join(datapath,'results_equiangular/test_perturbed',_id)
    # predictions batchxclass
    # pred_class batch == predictions
    retrieved = [(probabilities[j, predictions[j]], ids_test[j]) for j in range(len(ids_test)) if predictions[j] == predictions[i]]
    retrieved = sorted(retrieved, reverse=True)
    retrieved = [i for _, i in retrieved]
    with open(idfile, "w") as f:
        f.write("\n".join(retrieved))
