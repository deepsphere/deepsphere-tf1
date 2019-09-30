#!/usr/bin/env python3
# coding: utf-8

"""
Script to run the DeepSphere experiment.
Both the fully convolutional (FCN) and the classic (CNN) architecture variants
are supported.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import shutil
import sys

import numpy as np
import time

from deepsphere import models, experiment_helper
from deepsphere.data import LabeledDatasetWithNoise, LabeledDataset
import hyperparameters


def single_experiment(sigma, order, sigma_noise, experiment_type, new, n_neighbors):

    ename = '_'+experiment_type

    Nside = 1024
    
    if Nside == 1024:
        data_path = '/mnt/scratch/lts2/mdeff/deepsphere/data/same_psd/'
    else:
        data_path = 'data/same_psd/'
    
    EXP_NAME = 'cosmo' if new else 'oldgraph'
    EXP_NAME += '_{}sides_{}noise_{}order_{}sigma_{}neighbor{}_fold3'.format(
        Nside, sigma_noise, order, sigma, n_neighbors, ename)

    x_raw_train, labels_raw_train, x_raw_std = experiment_helper.get_training_data(sigma, order, data_path=data_path)
    x_raw_test, labels_test, _ = experiment_helper.get_testing_data(sigma, order, sigma_noise, x_raw_std, data_path=data_path[:-9])

    ret = experiment_helper.data_preprossing(x_raw_train, labels_raw_train, x_raw_test, sigma_noise, feature_type=None)
    features_train, labels_train, features_validation, labels_validation, features_test = ret

    training = LabeledDatasetWithNoise(features_train, labels_train, end_level=sigma_noise)
    validation = LabeledDataset(features_validation, labels_validation)

    # Cleanup before running again.
    
    shutil.rmtree('summaries/{}/'.format(EXP_NAME), ignore_errors=True)
    shutil.rmtree('checkpoints/{}/'.format(EXP_NAME), ignore_errors=True)
    
    params = hyperparameters.get_params(training.N, EXP_NAME, order, Nside, experiment_type)
    model = models.deepsphere(**params, new=new, n_neighbors=n_neighbors)

    accuracy_validation, loss_validation, loss_training, t_step, t_batch = model.fit(training, validation)
    print("inference time: ", t_batch/params["batch_size"])

    error_validation = experiment_helper.model_error(model, features_validation[:,:,np.newaxis], labels_validation)
    print('The validation error is {}%'.format(error_validation * 100), flush=True)

    error_test = experiment_helper.model_error(model, features_test[:,:,np.newaxis], labels_test)
    print('The testing error is {}%'.format(error_test * 100), flush=True)

    return error_test, t_batch


if __name__ == '__main__':

    if len(sys.argv) > 1:
        experiment_type = sys.argv[1]
    else:
        experiment_type = 'CNN' # 'CNN'

    if len(sys.argv) > 2:
        sigma = int(sys.argv[2])
        order = int(sys.argv[3])
        sigma_noise = float(sys.argv[4])
        grid = [(sigma, order, sigma_noise)]
    else:
        grid = [#(3, 1, 4, True, 8), 
                (3, 1, 3.5, False, 8),
                #(3, 1, 4, True, 20),
                #(3, 1, 4, True, 40),
                #(3, 1, 3, True, 40),# (3, 1, 3, False, 8), (3, 1, 3, True, 20), 
                #(3, 1, 3, True, 20),# (3, 1, 3.5, False, 8), (3, 1, 3.5, True, 20), 
                #(3, 1, 3, True, 8),# (3, 1, 4, False, 8), (3, 1, 4, True, 20), 
                #(3, 1, 3, False, 8),
                ]# pgrid() (3, 1, 4, False), (3, 1, 6, False), (3, 1, 8, False), 

    ename = '_'+experiment_type

    
    
    path = 'results/deepsphere/'
    os.makedirs(path, exist_ok=True)

    for sigma, order, sigma_noise, new, n_neighbors in grid:
        print('Launch experiment for sigma={}, order={}, noise={}, new graph={}, n neighbors={}'.format(sigma, order, 
                                                                                                        sigma_noise, 
                                                                                                        new, n_neighbors))
        np.random.seed(42)
        # avoid all jobs starting at the same time
        time.sleep(np.random.rand())
        res = single_experiment(sigma, order, sigma_noise, experiment_type, new, n_neighbors)
        filepath = os.path.join(path, 'deepsphere_results_list_sigma{}{}_fold2'.format(sigma,ename))
        new_data = [order, sigma_noise, n_neighbors, new, res]
        if os.path.isfile(filepath+'.npz'):
            results = np.load(filepath+'.npz')['data'].tolist()
        else:
            results = []
        results.append(new_data)
        np.savez(filepath, data=results)