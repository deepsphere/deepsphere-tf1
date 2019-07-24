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
import healpy as hp

from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs

from deepsphere import models
from deepsphere.data import LabeledDatasetWithNoise, LabeledDataset
from deepsphere.utils import icosahedron_graph

if __name__ == '__main__':
    path = '../../data/Climate/'
    g = icosahedron_graph(5)
    icolong, icolat = np.rad2deg(g.long), np.rad2deg(g.lat)
    del g
    
    precomp_mean = [26.160023, 0.98314494, 0.116573125, -0.45998842, 0.1930554, 0.010749293, 98356.03, 100982.02, 216.13145, 258.9456, 3.765611e-08, 288.82578, 288.03925, 342.4827, 12031.449, 63.435772]
    precomp_std =  [17.04294, 8.164175, 5.6868863, 6.4967732, 5.4465833, 0.006383436, 7778.5957, 3846.1863, 9.791707, 14.35133, 1.8771327e-07, 19.866386, 19.094095, 624.22406, 679.5602, 4.2283397]
    
    data = {}
    print('load data')
    for partition in ['train', 'val']:
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
    x_train = data['train']['data']
    labels_train = data['train']['labels']
    
    training = LabeledDataset(data['train']['data'], data['train']['labels'])
    validation = LabeledDataset(data['val']['data'], data['val']['labels'])
    del data
    
    EXP_NAME = 'TestClimate_pooling_weight_time2'
    
    # Cleanup before running again.
    shutil.rmtree('summaries/{}/'.format(EXP_NAME), ignore_errors=True)
    shutil.rmtree('checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)
    
    import tensorflow as tf
    if EXP_NAME == 'TestClimate_pooling_weight_time2':
        params = {'nsides': [5, 5, 4, 3, 2, 1, 0, 0],
                  'F': [32, 64, 128, 256, 512, 512, 512],#np.max(labels_train).astype(int)+1],
                  'K': [4]*7,
                  'batch_norm': [True]*7}
        params['sampling'] = 'icosahedron'
        params['dir_name'] = EXP_NAME
        params['num_feat_in'] = x_train.shape[-1] # 2*days_pred+3
        params['conv'] = 'chebyshev5'
        params['pool'] = 'average'
        params['activation'] = 'relu'
        params['statistics'] = None#'mean'
        params['regularization'] = 0
        params['dropout'] = 1
        params['num_epochs'] = 100  # Number of passes through the training data.
        params['batch_size'] = 8
        params['scheduler'] = lambda step: tf.train.exponential_decay(1e-3, step, decay_steps=2000, decay_rate=1)
        #params['optimizer'] = lambda lr: tf.train.GradientDescentOptimizer(lr)
        params['optimizer'] = lambda lr: tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.)
        n_evaluations = 400
        params['eval_frequency'] = int(params['num_epochs'] * (training.N) / params['batch_size'] / n_evaluations)
        params['M'] = []
        params['Fseg'] = np.max(labels_train).astype(int)+1
        params['dense'] = True
        params['profile'] = True
    elif EXP_NAME=='TestClimate_nopooling_weight_time':
        params = {'nsides': [5, 5, 5, 5],
                  'F': [64, 128, 256],#np.max(labels_train).astype(int)+1],
                  'K': [4]*3,
                  'batch_norm': [True]*3}
        params['sampling'] = 'icosahedron'
        params['dir_name'] = EXP_NAME
        params['num_feat_in'] = x_train.shape[-1] # 2*days_pred+3
        params['conv'] = 'chebyshev5'
        params['pool'] = 'average'
        params['activation'] = 'relu'
        params['statistics'] = None#'mean'
        params['regularization'] = 0
        params['dropout'] = 1
        params['num_epochs'] = 100  # Number of passes through the training data.
        params['batch_size'] = 8
        params['scheduler'] = lambda step: tf.train.exponential_decay(1e-2, step, decay_steps=2000, decay_rate=1)
        #params['optimizer'] = lambda lr: tf.train.GradientDescentOptimizer(lr)
        # params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        params['optimizer'] = lambda lr: tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.)
        n_evaluations = 200
        params['eval_frequency'] = int(params['num_epochs'] * (training.N) / params['batch_size'] / n_evaluations)
        params['M'] = []
        params['Fseg'] = np.max(labels_train).astype(int)+1
        params['dense'] = True
        params['profile'] = True
    
    model = models.deepsphere(**params)
    
    model.fit(training, validation)