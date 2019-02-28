#!/usr/bin/env python3
# coding: utf-8

"""
Script to run DeepSphere with SHREC17 dataset.
Both the fully convolutional (FCN) and the classic (CNN) architecture variants
are supported.
"""

import os
import shutil
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import time

from deepsphere import models, experiment_helper
from deepsphere.data import LabeledDatasetWithNoise, LabeledDataset
import hyperparameters

from SHREC17.load_shrec import Shrec17DeepSphere as shrecDataset


def single_experiment(datapath, sigma_noise, experiment_type):

    ename = '_'+experiment_type

    Nside = 32

    EXP_NAME = 'shrec17_40sim_{}sides_{}noise{}'.format(
        Nside, sigma_noise, ename)

    input("gathering data")

    # x_raw_train, labels_raw_train, x_raw_std = experiment_helper.get_training_data(sigma, order)
    # x_raw_test, labels_test, _ = experiment_helper.get_testing_data(sigma, order, sigma_noise, x_raw_std)
    train_dataset = shrecDataset(datapath, 'train', nside=Nside)
    #test_dataset = shrecDataset(datapath, 'test', perturbed=False, nside=Nside)

    input("preprocess data")

    # ret = experiment_helper.data_preprossing(x_raw_train, labels_raw_train, x_raw_test, sigma_noise, feature_type=None)
    # features_train, labels_train, features_validation, labels_validation, features_test = ret
    x_train, labels_train, x_val, labels_val = train_dataset.return_data(train=True)
    #x_test, _ = test_dataset.return_data()

    input("add noise")

    training = LabeledDatasetWithNoise(x_train, labels_train, end_level=sigma_noise)
    validation = LabeledDataset(x_val, labels_val)

    input("create model")

    params = hyperparameters.get_params_shrec17(training.N, EXP_NAME, Nside, train_dataset.nclass, architecture=experiment_type)
    model = models.deepsphere(**params)

    input("rmtree")

    # Cleanup before running again.
    shutil.rmtree('summaries/{}/'.format(EXP_NAME), ignore_errors=True)
    shutil.rmtree('checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)

    input("fit")

    model.fit(training, validation)

    input("error")

    error_validation = experiment_helper.model_error(model, x_val, labels_val)
    print('The validation error is {}%'.format(error_validation * 100), flush=True)

    #model.predict(features)
    #error_test = experiment_helper.model_error(model, features_test, labels_test)
    #print('The testing error is {}%'.format(error_test * 100), flush=True)

    #return error_test
    return error_validation


if __name__ == '__main__':

    if len(sys.argv) > 1:
        experiment_type = sys.argv[1]
    else:
        experiment_type = 'FCN' # 'CNN'

    if len(sys.argv) > 2:
        datapath = str(sys.argv[2])
        sigma_noise = float(sys.argv[3])
        grid = [(sigma_noise)]
    else:
        datapath = '../data/shrec17/'
        grid = [0.]

    ename = '_'+experiment_type

    path = 'results/shrec17/'
    os.makedirs(path, exist_ok=True)

    for sigma_noise in grid:
        print('Launch experiment for noise={}'.format(sigma_noise))
        # avoid all jobs starting at the same time
        time.sleep(np.random.rand()*50)
        res = single_experiment(datapath, sigma_noise, experiment_type)
        filepath = os.path.join(path, 'shrec17_results_list{}'.format(ename))
        new_data = [sigma_noise, res]
        if os.path.isfile(filepath+'.npz'):
            results = np.load(filepath+'.npz')['data'].tolist()
        else:
            results = []
        results.append(new_data)
        np.savez(filepath, data=results)
