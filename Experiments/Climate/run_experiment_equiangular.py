#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
import sys
sys.path.append('../..')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # change to chosen GPU to use, nothing if work on CPU

import numpy as np
import time
import matplotlib.pyplot as plt
import h5py

from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs

from deepsphere import models
from deepsphere.data import LabeledDataset
from ClimateDataLoader import EquiangularDataset
from pygsp.graphs import EquiangularGraph

if __name__ == '__main__':
    sampling = 'equiangular'
    
    path = '../../data/Climate/'
    g = SphereEquiangular(bandwidth=(384, 576), sampling='SOFT')
    glong, glat = np.rad2deg(g.long), np.rad2deg(g.lat)
    del g
    
    fstats = h5py.File(path+'stats.h5')
    stats = fstats['climate']["stats"] # (16 X 4) (mean, max, min, std)
    mean = stats[:,0]
    std = stats[:,-1]
    fstats.close()
    
    data = {}
    print('load data')
    training = EquiangularDataset(path, False)
    
    for partition in ['val']:
        with open(path+partition+".txt", "r") as f:
            lines = f.readlines()
        flist = [os.path.join(path, 'data_5_all', l.replace('\n', '')) for l in lines]
        data[partition] = {'data': np.zeros((len(flist),10242,16)),
                           'labels': np.zeros((len(flist),10242))}
        for i, f in enumerate(flist):
            file = np.load(f)
            data[partition]['data'][i] = (file['data'].T - precomp_mean) / precomp_std
            data[partition]['labels'][i] = np.argmax(file['labels'].astype(np.int), axis=0)
    
    print('data loaded')
    
    validation = LabeledDataset(data['val']['data'], data['val']['labels'])
    del data
    
    EXP_NAME = 'Climate_pooling_weight_{}'.format(sampling)
    
    # Cleanup before running again.
    shutil.rmtree('../../summaries/{}/'.format(EXP_NAME), ignore_errors=True)
    shutil.rmtree('../../checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)
    
    import tensorflow as tf
    params = {'nsides': [5, 5, 4, 3, 2, 1, 0, 0],
              'F': [32, 64, 128, 256, 512, 512, 512],#np.max(labels_train).astype(int)+1],
              'K': [4]*7,
              'batch_norm': [True]*7}
    params['sampling'] = sampling
    params['dir_name'] = EXP_NAME
    params['num_feat_in'] = 16 # x_train.shape[-1]
    params['conv'] = 'chebyshev5'
    params['pool'] = 'average'
    params['activation'] = 'relu'
    params['statistics'] = None # 'mean'
    params['regularization'] = 0
    params['dropout'] = 1
    params['num_epochs'] = 25  # Number of passes through the training data.
    params['batch_size'] = 8
    params['scheduler'] = lambda step: tf.train.exponential_decay(1e-3, step, decay_steps=2000, decay_rate=1)
    #params['optimizer'] = lambda lr: tf.train.GradientDescentOptimizer(lr)
    params['optimizer'] = lambda lr: tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.)
    n_evaluations = 50
    params['eval_frequency'] = int(params['num_epochs'] * (training.N) / params['batch_size'] / n_evaluations)
    params['M'] = []
    params['Fseg'] = 3 # np.max(labels_train).astype(int)+1
    params['dense'] = True
    params['profile'] = True
    
    model = models.deepsphere(**params)
    
    model.fit(training, validation)
    
    # TODO add test results