#!/usr/bin/env python3
# coding: utf-8

import tensorflow as tf
import numpy as np

def hyperparameters_dense(graph, EXP, neighbour, nfeat, N):
    params = {'L': [graph.L.astype(np.float32)]*4,
          'p': [1,1,1,1],
          'F': [50, 100, 100, 1],
          'K': [5]*4,
          'batch_norm': [True]*4}
    EXP_NAME = params['dir_name'] = 'GHCN_{}_{}neighbours_K{}'.format(EXP, neighbour, params['K'][0])
    params['num_feat_in'] = nfeat
    params['conv'] = 'chebyshev5'
    params['pool'] = 'max'
    params['activation'] = 'relu'
    params['statistics'] = None
    params['regularization'] = 0
    params['dropout'] = 1
    params['num_epochs'] = 250  # Number of passes through the training data.
    params['batch_size'] = 64
    params['scheduler'] = lambda step: tf.train.exponential_decay(1e-3, step, decay_steps=2000, decay_rate=1)
    #params['optimizer'] = lambda lr: tf.train.GradientDescentOptimizer(lr)
    params['optimizer'] = lambda lr: tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.)
    n_evaluations =250
    params['eval_frequency'] = int(params['num_epochs'] * (N) / params['batch_size'] / n_evaluations)
    params['M'] = []
    params['regression']=True
    params['dense'] = True
    
    return params


def hyperparameters_global(graph, EXP, neighbour, nfeat, N):
    params = {'L': [graph.L.astype(np.float32)]*3,
          'p': [1,1,1],
          'F': [50, 100, 100],
          'K': [5]*3,
          'batch_norm': [True]*3}
    EXP_NAME = params['dir_name'] = 'GHCN_{}_{}neighbours_K{}'.format(EXP, neighbour, params['K'][0])
    params['num_feat_in'] = nfeat
    params['conv'] = 'chebyshev5'
    params['pool'] = 'max'
    params['activation'] = 'relu'
    params['statistics'] = 'mean'
    params['regularization'] = 0
    params['dropout'] = 1
    params['num_epochs'] = 250  # Number of passes through the training data.
    params['batch_size'] = 64
    params['scheduler'] = lambda step: tf.train.exponential_decay(1e-3, step, decay_steps=2000, decay_rate=1)
    #params['optimizer'] = lambda lr: tf.train.GradientDescentOptimizer(lr)
    params['optimizer'] = lambda lr: tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.)
    n_evaluations =250
    params['eval_frequency'] = int(params['num_epochs'] * (N) / params['batch_size'] / n_evaluations)
    params['M'] = [1]
    params['regression']=True
    params['dense'] = False
    
    return params