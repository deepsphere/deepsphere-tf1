#!/usr/bin/env python3
# coding: utf-8

import os, sys
import shutil
sys.path.append('../..')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # change to chosen GPU to use, nothing if work on CPU

import numpy as np

from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error

from deepsphere import models
from GHCN_preprocessing import get_data, get_stations, sphereGraph, clean_nodes
from GHCN_train import hyperparameters_dense, hyperparameters_global



def mre(labels, predictions):
    return np.mean(np.abs((labels - predictions) / np.clip(labels, 1, None))) * 100


def _download(url, path):
    import requests

    filename = url.split('/')[-1]
    file_path = os.path.join(path, filename)

    if os.path.exists(file_path):
        return file_path

    print('Downloading ' + url)

    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return file_path


def download(dir_path, years):
    url = "https://www1.ncdc.noaa.gov/pub/data/ghcn/daily/"
    # download stations
    _download(url+"ghcnd-stations.txt", dir_path)
    for year in years:
        _download(url+str(year)+".csv.gz", dir_path)


if len(sys.argv) > 2:
    yearmin = sys.argv[1]
    yearmax = sys.argv[2]
    datapath = sys.argv[3]
    dense = sys.argv[4]
    dl_dataset = sys.argv[5]
else:
    yearmin = 2010
    yearmax = 2015
    datapath = "../../data/ghcn/"
    dense = True
    dl_dataset = True

years = np.arange(yearmin, yearmax)

if dl_dataset:
    download(datapath+"raw/", years)

feature_names = ['PRCP', 'TMIN', 'TMAX', 'SNOW', 'SNWD', 'WT']
n_features = len(feature_names)
n_years  = len(years)

n_stations, ghcn_to_local, lat, lon, alt, _ = get_stations(datapath, years)
full_data, n_days, valid_days = get_data(datapath, years, feature_names, ghcn_to_local)

assert n_stations == full_data.shape[0]

print('n_stations: {}, n_days: {}'.format(n_stations, n_days))

neighbour = 5

leap_years = np.zeros_like(years).astype(np.bool)
for i, in_year in enumerate(np.split(valid_days,len(years))):
    leap_years[i] = in_year.sum()==366
w_months = np.tile(np.repeat(np.arange(12), 31), years[-1]-years[0]+1)[valid_days]
w_days = np.tile(np.arange(365),years[-1]-years[0]+1)
for i, leap in enumerate(leap_years):
    if leap:
        w_days = np.insert(w_days, ((i+1)*365), 365)
w_days_sin = np.sin(w_days/367*np.pi)
w_days_cos = -np.cos(w_days/367*np.pi*2)/2+0.5

if dense:
    EXP = 'future'
    datas_temp_reg, keep_reg, gReg = clean_nodes(full_data, [1,3], lon, lat, figs=True, rad=False)
    from GHCN_preprocessing import dataset_reg
    training, validation = dataset_reg(datas_temp_reg, lon[keep_reg], lat[keep_reg], alt[keep_reg],
                                       w_days_sin, add_feat=False)
    placereg = np.empty_like(datas_temp_reg[:, :, 0])
    placereg[:, :] = np.arange(placereg.shape[1])
    nfeat = training.get_all_data()[0].shape[-1]
    params = hyperparameters_dense(gReg, EXP, neighbour, nfeat, training.N)
else:
    EXP = 'global_reg'
    datas_prec, keep_prec, gPrec = clean_nodes(full_data, [0,3], lon, lat, figs=True, rad=False)
    from GHCN_preprocessing import dataset_global
    training, validation = dataset_global(datas_prec, lon[keep_prec], lat[keep_prec],
                                          alt[keep_prec], w_days_cos, add_feat=False)
    nfeat = training.get_all_data()[0].shape[-1]
    params = hyperparameters_global(gPrec, EXP, neighbour, nfeat, training.N)

model = models.cgcnn(**params)

shutil.rmtree('summaries/{}/'.format(EXP), ignore_errors=True)
shutil.rmtree('checkpoints/{}/'.format(EXP), ignore_errors=True)

accuracy_validation, loss_validation, loss_training, t_step, t_batch = model.fit(training, validation)


x_val, labels_val = validation.get_all_data()
res = model.predict(np.atleast_3d(x_val))

if 'global' in EXP:
    mse_ = (mean_squared_error(labels_val, res))
    mae_ = (mean_absolute_error(labels_val, res))
    mre_ = (mre(labels_val, res))
    r2_ = (r2_score(labels_val, res))
    expvar_ = (explained_variance_score(labels_val, res))
    print("MSE={:.2f}, MAE={:.2f}, MRE={:.2f}, R2={:.3f}, Expvar={:.4f}".format(mse_, mae_, mre_, r2_, expvar_))
else:
    mse_ = (mean_squared_error(labels_val[:-1,:], res[1:,:]))
    mae_ = (mean_absolute_error(labels_val[:-1,:], res[1:,:]))
    mre_ = (mre(labels_val[:-1,:], res[1:,:]))
    r2_ = (r2_score(labels_val[:-1,:], res[1:,:]))
    expvar_ = (explained_variance_score(labels_val[:-1,:], res[1:,:]))
    print("MSE={:.2f}, MAE={:.2f}, MRE={:.2f}, R2={:.3f}, Expvar={:.4f}".format(mse_, mae_, mre_, r2_, expvar_))

# if 'future' in EXP:
#     predictions = []
#     for i in range(days_pred):
#         predictions.append(dataset_temp_reg[i,:,0])
#     for i in tqdm(range(len(dataset_temp_reg)-2*days_pred)):
#         x_pred = np.asarray(predictions[-days_pred:]).T
#         x_pred = np.hstack([x_pred,
#     #                      np.broadcast_to(w_months[np.newaxis,i:i+days_pred], x_pred.shape),
#                            coords_v,
#                            alt_v[:,np.newaxis],
#                            np.repeat(w_days_sin[i], x_pred.shape[0], axis=0)[:,np.newaxis]])
#                            #np.broadcast_to(w_days[np.newaxis, i:i+days_pred], x_pred.shape)])
#         x_pred = np.repeat(x_pred[np.newaxis,:,:], 64, axis=0)
#         res = model.predict(x_pred)
#         predictions.append(res[0,:])
