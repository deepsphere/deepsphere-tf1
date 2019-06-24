#!/usr/bin/env python3
# coding: utf-8

"""
Script to run the DeepSphere experiment with SHREC17 dataset.
Both the fully convolutional (FCN) and the classic (CNN) architecture variants
are supported.
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''       # to mask the GPU before tensorflow is called
import shutil
import sys

import numpy as np
import time

from DeepSphere.deepsphere import models, experiment_helper
from DeepSphere.deepsphere.data import LabeledDatasetWithNoise, LabeledDataset
import DeepSphere.hyperparameters as hyperparameters
import DeepSphere.SHREC17.load_shrec as lds


def single_experiment(sigma_noise, experiment_type):

    ename = '_'+experiment_type

    Nside = 32  # for a bandwidth inferior to 64
    download = True

    EXP_NAME = '40sim_{}sides_{}noise{}'.format(
        Nside, sigma_noise, ename)

    input("gathering data")

    train_dataset = lds.Shrec17DeepSphere('data', 'train', download=download, nside=Nside)
    val_dataset = lds.Shrec17DeepSphere('data', 'val', perturbed=False, download=download, nside=Nside)
    test_dataset = lds.Shrec17DeepSphere('data', 'test', perturbed=False, download=download, nside=Nside)

    input("preprocess data")

    ret = train_dataset.return_data(sigma=sigma_noise)
    features_train, labels_train, features_validation, labels_validation = ret
    ret = test_dataset.return_data(test=True, sigma=0.)
    features_val, labels_val = ret
    ret = test_dataset.return_data(test=True, sigma=0.)
    features_test, _ = ret

    input("add noise")

    training = LabeledDatasetWithNoise(features_train, labels_train, end_level=sigma_noise)
    validation = LabeledDataset(features_validation, labels_validation)

    input("create model")

    params = hyperparameters.get_params_shrec17(training.N, EXP_NAME, Nside, train_dataset.nclass, features_train.shape[-1], experiment_type)
    model = models.deepsphere(**params)

    input("rmtree")

    # Cleanup before running again.
    shutil.rmtree('summaries/{}/'.format(EXP_NAME), ignore_errors=True)
    shutil.rmtree('checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)

    input("fit")

    model.fit(training, validation)

    input("error")

    error_validation = experiment_helper.model_error(model, features_validation, labels_validation)
    print('The validation error is {}%'.format(error_validation * 100), flush=True)

    error_test = experiment_helper.model_error(model, features_test, labels_test)
    print('The testing error is {}%'.format(error_test * 100), flush=True)

    return error_test


if __name__ == '__main__':

    if len(sys.argv) > 1:
        experiment_type = sys.argv[1]
    else:
        experiment_type = 'FCN' # 'CNN'

    ename = '_'+experiment_type

    path = 'results/shrec17/'
    os.makedirs(path, exist_ok=True)

    for sigma_noise in [0, 0.5, 1, 1.5, 2]:
        print('Launch experiment for noise={}'.format(sigma_noise))
        # avoid all jobs starting at the same time
        time.sleep(np.random.rand()*100)
        res = single_experiment(sigma_noise, experiment_type)
        filepath = os.path.join(path, 'shrec17_results_list_noise{}{}'.format(sigma_noise,ename))
        new_data = [sigma_noise, res]
        if os.path.isfile(filepath+'.npz'):
            results = np.load(filepath+'.npz')['data'].tolist()
        else:
            results = []
        results.append(new_data)
        np.savez(filepath, data=results)
