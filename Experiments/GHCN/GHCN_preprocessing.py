#!/usr/bin/env python3
# coding: utf-8

import os
import shutil
import sys

import numpy as np
import time
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd

from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import cartopy.crs as ccrs

from deepsphere import models, experiment_helper, plot, utils
from deepsphere.data import LabeledDatasetWithNoise, LabeledDataset
import hyperparameters

datapath = "/mnt/nas/LTS2/datasets/ghcn-daily/processed/"
rawpath = "/mnt/nas/LTS2/datasets/ghcn-daily/raw/"
newdatapath = "../../data/ghcn-daily/processed/"

def get_stations(data_path, years, rawpath=None):
    filename = 'stations_{:4d}-{:4d}.npz'.format(years[0], years[-1])
    if not os.path.isfile(datapath+filename):
        print('Problem occured')
        raise ValueError()

        id_ghcn, lat, lon, elev, name = [], [], [], [], []
        with open(rawpath+'ghcnd-stations.txt', 'r') as f:
            for line in f:

                iid, ilat, ilon, ielev, iname = line[0:11], line[12:20], line[21:30], line[31:37], line[41:71]

                assert (not iid.isspace()) and (not ilat.isspace()) and (not ilon.isspace()) \
                    and (not ielev.isspace()) and (not iname.isspace())

                id_ghcn.append(iid.strip())
                lat.append(float(ilat.strip()))
                lon.append(float(ilon.strip()))
                elev.append(float(ielev.strip()))
                name.append(iname.strip())

        id_ghcn, lat, lon, elev, name = np.array(id_ghcn), np.array(lat), np.array(lon), np.array(elev), np.array(name)
        id_ghcn_relevant = set([])

        for yearIdx,year in enumerate(years):

            filename2 = rawpath+'{:4}.csv.gz'.format(year)
            print('- pre-parsing : {}'.format(filename2))

            df = pd.read_csv(filename2, names=['id_ghcn', 'date', 'type', 'value', '?0', '?1', '?2', '?3'], \
                             nrows=None, usecols=[0,1,2,3])


            id_ghcn_relevant |= set(df["id_ghcn"].values)

        # second, find identifiers both in id_ghcn and id_ghcn_relevant
        id_ghcn_relevant = set(id_ghcn) & id_ghcn_relevant

        # third, keep only relevant station data 
        keep = [id in id_ghcn_relevant for id in id_ghcn] 
        id_ghcn, lat, lon, elev, name = id_ghcn[keep], lat[keep], lon[keep], elev[keep], name[keep] 

        # free up some memory
        del id_ghcn_relevant, keep

        np.savez_compressed(datapath+filename, id_ghcn=id_ghcn, lat=lat, lon=lon, elev=elev, name=name, years=years)

    else:
        station_file = np.load(datapath+filename)
        id_ghcn, lat, lon, elev, name = station_file['id_ghcn'], station_file['lat'], station_file['lon'], station_file['elev'], station_file['name']
        del station_file

    n_stations = id_ghcn.shape[0]
    print('{} weather stations identified.'.format(n_stations))

    #  a dictionary mapping GHCN ids to local ids (rows in id array) 
    ghcn_to_local = dict(zip(id_ghcn, np.arange(n_stations)))
    return n_stations, ghcn_to_local, lat, lon, elev, name


def get_data(datapath, years, feature_names, ghcn_to_local, rawpath=None):
    filenames = []
    datas = []
    n_years  = len(years)
    for feature_name in feature_names:
        filenames.append('data_{:4d}-{:4d}_{}.npz'.format(years[0], years[-1], feature_name))
        print(f'- Checking if file {filenames[-1]} exists..')

        # only recompute if necessary
        if not os.path.isfile(newdatapath+filenames[-1]):

            print('- The file is not there. Parsing everything from raw. This will take a while.')
            os.makedirs(newdatapath, exist_ok=True)
            # Load the station measurements into a year-list of dataframes
            df_years = []

            for yearIdx,year in enumerate(years):

                filename_year = rawpath+'{:4}.csv.gz'.format(year)
                print(' - parsing *{}*'.format(filename_year))

                df = pd.read_csv(filename_year, names=['id_ghcn', 'date', 'type', 'value', 'MF', 'qualityF', 'source', '?0'], \
                                 nrows=None, usecols=[0,1,2,3,5])

                # create a new column with the id_local
                id_local = [ghcn_to_local.get(id_g) for id_g in df["id_ghcn"].values]
                id_local = [-1 if v is None else v for v in id_local]
                id_local = np.array(id_local).astype(np.int)

                df = df.assign(id_local=pd.Series(id_local, index=df.index).values)

                # remove measurement of stations with unknown id_local
                df = df[df.id_local != -1] 

                # replace measurements with bad quality flag
                #df.value[~df.qualityF.isna()] = np.nan
                df = df[df.qualityF.isna()]
                df = df.drop('qualityF', axis=1)

                df_years.append(df)

            del df, id_local
            print('done!')

            # Construct one array per feature and save it to disk

            # indicate for which days we have measurements (this also deals with months of different lengths)
            valid_days = np.zeros((n_years, 12, 31), dtype=np.bool)

            for _, name in enumerate(feature_names):

                print(f' - Looking at {name}')

                data = np.zeros((n_stations, n_years, 12, 31), dtype=np.float) * np.nan

                for yearIdx,year in enumerate(years):

                    df = df_years[yearIdx]
                    idf = df.loc[df.type.str.contains(name)]

                    print(f'  - year {year}')

                    # remove measurement of stations with unknown id_local
                    idf = idf[idf.id_local != -1] 

                    for monthIdx,month in enumerate(range(1,12+1)): 
                        for dayIdx,day in enumerate(range(1,31+1)):        

                            date = int('{:4d}{:02d}{:02d}'.format(year,month,day))
                            jdf = idf.loc[idf['date'] == date]

                            # sort data according to the id_local 
                            jdf.set_index('id_local', inplace=True)
                            jdf = jdf.sort_index()

                            index = jdf.index.values
                            if name is 'WT' or name is 'WV':
                                values = jdf.type.str.extract(r'(\d+)').values.astype(int)
                                values = values[:,0]
                            else:
                                values = jdf['value'].values.astype(np.float)

                            if len(index) != 0: 
                                data[index,yearIdx,monthIdx,dayIdx] = values
                                valid_days[yearIdx,monthIdx,dayIdx] = True

                print('  - saving to disk')
                np.savez_compressed(newdatapath+'data_{:4d}-{:4d}_{}.npz'.format(years[0], years[-1], name), data=data, valid_days=valid_days)

                del index, values, df, idf, jdf    

        else:    
            print('- Loading data from disk..')

            data_file = np.load(newdatapath+filenames[-1])
            data, valid_days = data_file['data'], data_file['valid_days']        
            n_stations = data.shape[0]
            print(f'- {n_stations} stations loaded.')
            data = data.reshape((n_stations, n_years*12*31))
            if feature_name == 'TMIN' or feature_name == 'TMAX' or feature_name == 'PRCP':
                data = data.astype(np.float)
                data /= 10
            datas.append(data)
            valid_days = np.squeeze(valid_days.reshape(n_years*12*31)).astype(np.bool)
            
    full_data = np.stack(datas, axis=2)
    full_data = full_data[:, valid_days, :]

    n_days = full_data.shape[1]
    return full_data, n_days, valid_days


def clean_nodes(data, feat, lon, lat, superset=False, neighbor=10, figs=False, **kwargs):
    """
    data: full data from GHCN
    feat: list or tuple containing indices of first and last feature to keep
    lon, lat: position of weath nodes
    superset: keep only minset if False, else nodes having at least 75% of measurements
    neighbor: number of neighbor in knn-graph
    figs: print the figs
    """
    sl = slice(*feat)
    dataset = data.transpose((1, 0, 2))
    keepToo = ~np.isnan(dataset[:,:,sl]).any(axis=0)
    keepSuper = ((~np.isnan(dataset[:,:,sl])).sum(axis=0)>0.75*dataset.shape[0])
    keepToo = keepToo.all(axis=1)
    keepSuper = keepSuper.all(axis=1)
    dataset = dataset[:, keepToo, sl]
    print("number of stations in min set: {}\nnumber of stations in super set: {}".format(keepToo.sum(), keepSuper.sum()))
    keep = keepSuper if superset else keepToo
    
    if keep.sum()==0:
        print("no nodes for the current configuration")
        return [None]*3
    
    graph = sphereGraph(lon[keep], lat[keep], neighbor, **kwargs)
    graph.compute_laplacian("combinatorial")
    
    if figs:
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines()
        plt.plot(lon[keep], lat[keep], 'or', marker='o', markerfacecolor='r', markersize=2)
        fig2 = plt.figure(figsize=(20,20))
        axes = fig2.add_subplot(111, projection='3d')
        graph.plot(vertex_size=10, edges=True, ax=axes)
    return dataset, keep, graph


sys.path.append('../../deepsphere')
from data import LabeledDataset
def dataset_temp(datas, lon=None, lat=None, alt=None, w_days=None, add_feat=True, ratio=0.7):
    n_days = datas.shape[0]
    limit = int(ratio*n_days)

    mean = datas.mean(axis=(0,1))[0]
    std = datas.std(axis=(0,1))[0]

    x_train = (datas[:limit,:,0] - mean) / std
    labels_train = datas[:limit,:,1]
    x_val = (datas[limit:,:,0] - mean) / std
    labels_val = datas[limit:,:,1]

    if add_feat:
        # location of stations
        coords_v = np.stack([lon, lat], axis=-1)
        coords_v = (coords_v-coords_v.mean(axis=0))/coords_v.std(axis=0)
        # altitude of stations
        alt_v = alt
        alt_v = (alt_v-alt_v.mean())/alt_v.std()

        x_train = np.dstack([x_train, np.repeat(coords_v[np.newaxis,:], x_train.shape[0], axis=0),
                             np.repeat(alt_v[np.newaxis,:], x_train.shape[0], axis=0),
                             np.repeat(w_days[:limit, np.newaxis], x_train.shape[1], axis=1)])
        x_val = np.dstack([x_val, np.repeat(coords_v[np.newaxis,:], x_val.shape[0], axis=0),
                          np.repeat(alt_v[np.newaxis,:], x_val.shape[0], axis=0),
                          np.repeat(w_days[limit:, np.newaxis], x_val.shape[1], axis=1)])

    training = LabeledDataset(x_train, labels_train)
    validation = LabeledDataset(x_val, labels_val)
    return training, validation

def dataset_prec(datas, lon=None, lat=None, alt=None, w_days=None, add_feat=True, ratio=0.7):
    n_days = datas.shape[0]
    limit = int(ratio*n_days)

    mean = datas.mean(axis=(0,1))[1:3]
    std = datas.std(axis=(0,1))[1:3]

    x_train = (datas[:limit,:,1:3] - mean) / std
    labels_train = datas[:limit,:,0]
    x_val = (datas[limit:,:,1:3] - mean) / std
    labels_val = datas[limit:,:,0]

    if add_feat:
        # location of stations
        coords_v = np.stack([lon, lat], axis=-1)
        coords_v = (coords_v-coords_v.mean(axis=0))/coords_v.std(axis=0)
        # altitude of stations
        alt_v = alt
        alt_v = (alt_v-alt_v.mean())/alt_v.std()

        x_train = np.dstack([x_train, np.repeat(coords_v[np.newaxis,:], x_train.shape[0], axis=0),
                             np.repeat(alt_v[np.newaxis,:], x_train.shape[0], axis=0),
                             np.repeat(w_days[:limit, np.newaxis], x_train.shape[1], axis=1)])
        x_val = np.dstack([x_val, np.repeat(coords_v[np.newaxis,:], x_val.shape[0], axis=0),
                          np.repeat(alt_v[np.newaxis,:], x_val.shape[0], axis=0),
                          np.repeat(w_days[limit:, np.newaxis], x_val.shape[1], axis=1)])

    training = LabeledDataset(x_train, labels_train)
    validation = LabeledDataset(x_val, labels_val)
    return training, validation


def dataset_reg(datas, lon=None, lat=None, alt=None, w_days=None, add_feat=False, days_pred=5, ratio=0.7):
    n_days, n_stations, n_feature= datas.shape
    limit = int(0.7*(n_days-days_pred))
    
    dataset_x = np.vstack([np.roll(datas, -i, axis=0) for i in range(days_pred)])
    dataset_x = dataset_x.reshape(days_pred, n_days, n_stations, n_feature).transpose((1,2,3,0))
    
    days_x = np.vstack([np.roll(w_days, -i, axis=0) for i in range(days_pred)])
    days_x = days_x.reshape(days_pred, n_days).transpose()

    x_train = dataset_x[:limit,:,:,:].transpose(0, 2, 1, 3).reshape(-1, n_stations, days_pred)
    labels_train = datas[days_pred:limit+days_pred,:,:].transpose(0,2,1).reshape(-1, n_stations)
    x_val = dataset_x[limit:n_days-days_pred,:,:,:].transpose(0, 2, 1, 3).reshape(-1, n_stations, days_pred)
    labels_val = datas[days_pred+limit:,:,:].transpose(0,2,1).reshape(-1, n_stations)

    if add_feat:
        # location of stations
        coords_v = np.stack([lon, lat], axis=-1)
        coords_v = (coords_v-coords_v.mean(axis=0))/coords_v.std(axis=0)
        # altitude of stations
        alt_v = alt
        alt_v = (alt_v-alt_v.mean())/alt_v.std()

        x_train = np.dstack([x_train, 
#                     np.broadcast_to(month_x[:n_days-days_pred,np.newaxis, :], x_train.shape),
                     np.repeat(coords_v[np.newaxis,:], x_train.shape[0],axis=0),
                     np.repeat(alt_v[np.newaxis,:], x_train.shape[0],axis=0),
                     np.tile(np.repeat(w_days[:limit, np.newaxis], x_train.shape[1],axis=1), (2,1))])
#                      np.broadcast_to(days_x[:n_days-days_pred,np.newaxis, :], x_train.shape)])

        x_val = np.dstack([x_val, 
#                   np.broadcast_to(month_x[:n_days-days_pred,np.newaxis, :], x_val.shape), 
                   np.repeat(coords_v[np.newaxis,:], x_val.shape[0],axis=0),
                   np.repeat(alt_v[np.newaxis,:], x_val.shape[0],axis=0),
                   np.tile(np.repeat(w_days[limit:n_days-days_pred, np.newaxis], x_val.shape[1],axis=1), (2,1))])
#                    np.broadcast_to(days_x[:n_days-days_pred,np.newaxis, :], x_val.shape)])

    training = LabeledDataset(x_train, labels_train)
    validation = LabeledDataset(x_val, labels_val)
    return training, validation

def dataset_snow(datas, lon=None, lat=None, alt=None, w_days=None, add_feat=True, ratio=0.7):
    n_days = datas.shape[0]
    limit = int(ratio*n_days)

    mean = datas.mean(axis=(0,1))[:3]
    std = datas.std(axis=(0,1))[:3]

    x_train = (datas[:limit,:,:3] - mean) / std
    labels_train = datas[:limit,:,3]
    x_val = (datas[limit:,:,:3] - mean) / std
    labels_val = datas[limit:,:,3]

    if add_feat:
        # location of stations
        coords_v = np.stack([lon, lat], axis=-1)
        coords_v = (coords_v-coords_v.mean(axis=0))/coords_v.std(axis=0)
        # altitude of stations
        alt_v = alt
        alt_v = (alt_v-alt_v.mean())/alt_v.std()

        x_train = np.dstack([x_train, np.repeat(coords_v[np.newaxis,:], x_train.shape[0], axis=0),
                             np.repeat(alt_v[np.newaxis,:], x_train.shape[0], axis=0),
                             np.repeat(w_days[:limit, np.newaxis], x_train.shape[1], axis=1)])
        x_val = np.dstack([x_val, np.repeat(coords_v[np.newaxis,:], x_val.shape[0], axis=0),
                          np.repeat(alt_v[np.newaxis,:], x_val.shape[0], axis=0),
                          np.repeat(w_days[limit:, np.newaxis], x_val.shape[1], axis=1)])

    training = LabeledDataset(x_train, labels_train)
    validation = LabeledDataset(x_val, labels_val)
    return training, validation

def dataset_global(datas, lon=None, lat=None, alt=None, w_days=None, add_feat=True, ratio=0.7):
    n_days = datas.shape[0]
    limit = int(ratio*n_days)

    mean = datas.mean(axis=(0,1))[0]
    std = datas.std(axis=(0,1))[0]

    x_train = np.atleast_3d((datas[:limit,:,0] - mean) / std)
    labels_train = w_days[:limit]
    x_val = np.atleast_3d((datas[limit:,:,0] - mean) / std)
    labels_val = w_days[limit:]

    if add_feat:
        # location of stations
        coords_v = np.stack([lon, lat], axis=-1)
        coords_v = (coords_v-coords_v.mean(axis=0))/coords_v.std(axis=0)
        # altitude of stations
        alt_v = alt
        alt_v = (alt_v-alt_v.mean())/alt_v.std()

        x_train = np.dstack([x_train, np.repeat(coords_v[np.newaxis,:], x_train.shape[0], axis=0),
                             np.repeat(alt_v[np.newaxis,:], x_train.shape[0], axis=0),
                             np.repeat(w_days[:limit, np.newaxis], x_train.shape[1], axis=1)])
        x_val = np.dstack([x_val, np.repeat(coords_v[np.newaxis,:], x_val.shape[0], axis=0),
                          np.repeat(alt_v[np.newaxis,:], x_val.shape[0], axis=0),
                          np.repeat(w_days[limit:, np.newaxis], x_val.shape[1], axis=1)])

    training = LabeledDataset(x_train, labels_train)
    validation = LabeledDataset(x_val, labels_val)
    return training, validation


from pygsp.graphs import NNGraph
class sphereGraph(NNGraph):
    def __init__(self, phi, theta, neighbors, rad=True, epsilon=False, **kwargs):
        if not rad:
            theta, phi = np.deg2rad(theta), np.deg2rad(phi)
        theta -= np.pi/2
        ct = np.cos(theta).flatten()
        st = np.sin(theta).flatten()
        cp = np.cos(phi).flatten()
        sp = np.sin(phi).flatten()
        x = st * cp
        y = st * sp
        z = ct
        self.coords = np.vstack([x, y, z]).T
        NNtype = 'radius' if epsilon else 'knn'
        plotting = {"limits": np.array([-1, 1, -1, 1, -1, 1])*0.5}
        self.n_vertices = len(self.coords)
        super(sphereGraph, self).__init__(self.coords, k=neighbors, NNtype=NNtype, center=False, rescale=False,
                                     plotting=plotting, **kwargs)