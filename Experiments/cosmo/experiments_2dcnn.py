#!/usr/bin/env python3
# coding: utf-8

"""
Script to run the DeepSphere experiment.
Both the fully convolutional (FCN) and the classic (CNN) architecture variants
are supported.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import sys
sys.path.append('../..')

import numpy as np
import time

from deepsphere.utils import build_index
from deepsphere import models, experiment_helper, utils
from deepsphere.data import LabeledDatasetWithNoise, LabeledDataset


def get_params(ntrain, EXP_NAME, order, Nside, architecture="FCN", verbose=True):
    """Parameters for the cgcnn and cnn2d defined in deepsphere/models.py"""

    n_classes = 2

    params = dict()
    params['dir_name'] = EXP_NAME

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [16, 32, 64, 64, 64, n_classes]  # Graph convolutional layers: number of feature maps.
    params['K'] = [5] * 6  # Polynomial orders.
    params['batch_norm'] = [True] * 6  # Batch normalization.
    params['M'] = []  # Fully connected layers: output dimensionalities.

    # Pooling.
    nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//32, Nside//32]
    params['nsides'] = nsides
    params['indexes'] = utils.nside2indexes(nsides, order)
#     params['batch_norm_full'] = []

    if architecture == "CNN":
        # Classical convolutional neural network.
        # Replace the last graph convolution and global average pooling by a fully connected layer.
        # That is, change the classifier while keeping the feature extractor.
        params['F'] = params['F'][:-1]
        params['K'] = params['K'][:-1]
        params['batch_norm'] = params['batch_norm'][:-1]
        params['nsides'] = params['nsides'][:-1]
        params['indexes'] = params['indexes'][:-1]
        params['statistics'] = None
        params['M'] = [n_classes]
    elif architecture == "FCN":
        pass
    elif architecture == 'CNN-2d':
        params['F'] = [8, 16, 32, 32, 16]
        params['K'] = [[5, 5]] * 5
        params['p'] = [2, 2, 2, 2, 2]
        params['input_shape'] = [1024//order, 1024//order]
        params['batch_norm'] = params['batch_norm'][:-1]
        params['statistics'] = None
        params['M'] = [n_classes]
        del params['indexes']
        del params['nsides']
        del params['conv']

    elif architecture == 'FCN-2d':
        params['F'] = [8, 16, 32, 32, 16, 2]
        params['K'] = [[5, 5]] * 6
        params['p'] = [2, 2, 2, 2, 2, 1]
        params['input_shape'] = [1024//order, 1024//order]
        del params['indexes']
        del params['nsides']
        del params['conv']
    else:
        raise ValueError('Unknown architecture {}.'.format(architecture))

    # Regularization (to prevent over-fitting).
    params['regularization'] = 0  # Amount of L2 regularization over the weights (will be divided by the number of weights).
    if '2d' in architecture:
        params['regularization'] = 3
    params['dropout'] = 1  # Percentage of neurons to keep.

    # Training.
    params['num_epochs'] = 80  # Number of passes through the training data.
    params['batch_size'] = max(8 * order, 1)    # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: tf.train.exponential_decay(2e-4, step, decay_steps=1, decay_rate=0.999)
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

    # Number of model evaluations during training (influence training time).
    n_evaluations = 80
    params['eval_frequency'] = int(params['num_epochs'] * ntrain / params['batch_size'] / n_evaluations)

    if verbose:
        print('#sides: {}'.format(nsides))
#         print('#pixels: {}'.format([(nside//order)**2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
#         print('=> #pixels per batch (input): {:,}'.format(params['batch_size']*(Nside//order)**2))
#         print('=> #pixels for training (input): {:,}'.format(params['num_epochs']*ntrain*(Nside//order)**2))

        n_steps = params['num_epochs'] * ntrain // params['batch_size']
        lr = [params['scheduler'](step).eval(session=tf.Session()) for step in [0, n_steps]]
        print('Learning rate will start at {:.1e} and finish at {:.1e}.'.format(*lr))

    return params


def single_experiment(sigma, order, sigma_noise, experiment_type):

    ename = '_'+experiment_type

    Nside = 1024

    data_path = '../../data/same_psd/'

    EXP_NAME = '40sim_{}sides_{}noise_{}order_{}sigma{}'.format(
        Nside, sigma_noise, order, sigma, ename)

    x_raw_train, labels_raw_train, x_raw_std = experiment_helper.get_training_data(sigma, order, data_path=data_path)
    x_raw_test, labels_test, _ = experiment_helper.get_testing_data(sigma, order, sigma_noise, x_raw_std, data_path=data_path[:-9])

    ret = experiment_helper.data_preprossing(x_raw_train, labels_raw_train, x_raw_test, sigma_noise, feature_type=None)
    features_train, labels_train, features_validation, labels_validation, features_test = ret

    nx = Nside//order
    nlevels = np.round(np.log2(nx)).astype(np.int)
    index = build_index(nlevels).astype(np.int)

    features_train = features_train[:, index]
    features_validation = features_validation[:, index]
    shuffle = np.random.permutation(len(features_test))
    features_test = features_test[:, index]
    features_test = features_test[shuffle]
    labels_test = labels_test[shuffle]

    training = LabeledDatasetWithNoise(features_train, labels_train, end_level=sigma_noise)
    validation = LabeledDataset(features_validation, labels_validation)

    # Better implementation, but it doesn't work for some reason.
    # params = hyperparameters.get_params_CNN2D(training.N, EXP_NAME, order, Nside, experiment_type)
    # model = Healpix2CNN(**params)

    params = get_params(training.N, EXP_NAME, order, Nside, experiment_type)
    model = models.cnn2d(**params)

    # Cleanup before running again.
    shutil.rmtree('summaries/{}/'.format(EXP_NAME), ignore_errors=True)
    shutil.rmtree('checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)

    model.fit(training, validation)

    error_validation = experiment_helper.model_error(model, features_validation, labels_validation)
    print('The validation error is {}%'.format(error_validation * 100), flush=True)

    error_test = experiment_helper.model_error(model, features_test, labels_test)
    print('The testing error is {}%'.format(error_test * 100), flush=True)

    return error_test


if __name__ == '__main__':

    if len(sys.argv) > 1:
        experiment_type = sys.argv[1]
    else:
        experiment_type = 'CNN-2d' # 'CNN'

    if len(sys.argv) > 2:
        sigma = int(sys.argv[2])
        order = int(sys.argv[3])
        sigma_noise = float(sys.argv[4])
        grid = [(sigma, order, sigma_noise)]
    else:
        grid = [(3, 1, 3.5)]

    ename = '_'+experiment_type

    path = 'results/2dcnn/'
    os.makedirs(path, exist_ok=True)

    for sigma, order, sigma_noise in grid:
        print('Launch experiment for sigma={}, order={}, noise={}'.format(sigma, order, sigma_noise))
        # avoid all jobs starting at the same time
        time.sleep(np.random.rand())
        res = single_experiment(sigma, order, sigma_noise, experiment_type)
        filepath = os.path.join(path, 'reg3_deepsphere_results_list_sigma{}{}'.format(sigma,ename))
        new_data = [order, sigma_noise, res]
        if os.path.isfile(filepath+'.npz'):
            results = np.load(filepath+'.npz')['data'].tolist()
        else:
            results = []
        results.append(new_data)
        np.savez(filepath, data=results)
