#!/usr/bin/env python3
# coding: utf-8

"""
 Load dataset year by year, interpolate each map, and add label for each pixel.
 No special preprocessing for the labels, only bouding box
"""

import os
import shutil
import sys

import numpy as np
import time
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd

from tqdm import tqdm
import h5py
import matplotlib.patches as patches
from scipy.interpolate import griddata, RegularGridInterpolator



def download(datapath, url, year):
    import requests

    url = url.format(year)
    
    filename = url.split('/')[-1]
    file_path = os.path.join(datapath, filename)

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

def interpolate(images, boxes):
    measures, channels, lat_x, lon_x = images.shape
    lon_ = np.arange(lon_x)/lon_x*360
    lat_ = np.arange(lat_x)/lat_x*180-90
    lon, lat = np.meshgrid(*(lon_, lat_))
    nfeat = 5
    Nside = [32, 64]
    for nside in Nside:
        print("preprocessing data at nside = {}".format(nside))
        npix = hp.nside2npix(nside)
        data = np.empty((measures, npix, channels))
        labels = np.zeros((measures, npix, nfeat))
        pix = np.arange(npix)
        coords_hp = hp.pix2ang(nside, pix, nest=True, lonlat=True)
        coords_hp = np.asarray(coords_hp).T
        for measure in tqdm(range(measures)):
            for channel in range(channels):
                f = RegularGridInterpolator((lon_, lat_), images[measure,channel].T)
                data[measure,:,channel] = f(coords_hp)
            for box in range(boxes.shape[1]):
                ymin, xmin, ymax, xmax, clas = boxes[measure,box]
                if ymin==-1:
                    continue
                ymin, ymax = lat_[ymin%lat_x], lat_[ymax%lat_x]
                xmin, xmax = lon_[xmin%lon_x], lon_[xmax%lon_x]
                if xmax>xmin and ymax>ymin:
                    indexes = np.where(np.logical_and(np.logical_and(coords_hp[:,0]>=xmin, coords_hp[:,0]<=xmax), 
                                            np.logical_and(coords_hp[:,1]>=ymin, coords_hp[:,1]<=ymax)))
                else:
                    indexes = np.where(np.logical_and(np.logical_or(coords_hp[:,0]>=xmin, coords_hp[:,0]<=xmax), 
                                            np.logical_and(coords_hp[:,1]>=ymin, coords_hp[:,1]<=ymax)))
                labels[measure, indexes,:] = clas + 1
        datapath = '../../data/ExtremeWeather/'
        file = datapath + 'EW_{}nside_{}'.format(nside, year)
        np.savez(file, data=data, labels=labels)
        print("save file at: "+file)
    pass

if __name__=='__main__':
    years = np.arange(1979, 2006)
    years = [1981, 1984]
    url = 'https://portal.nersc.gov/project/dasrepo/DO_NOT_REMOVE/extremeweather_dataset/h5data/climo_{}.h5'
    datapath = '../../data/ExtremeWeather/'
    file = 'EW_32nside_{}.npz'
    
    for year in years:
        if os.path.exists(os.path.join(datapath, file.format(year))):
            continue
        h5_path = download(datapath, url, year)
        h5f = h5py.File(h5_path)
        images = h5f["images"] # (1460,16,768,1152) numpy array
        boxes = h5f["boxes"] # (1460,15,5) numpy array
        interpolate(images, boxes)
        os.remove(h5_path)
        print("h5 file removed")
    
    
    
"""
List of parameters:
* PRECT: Total (convective and large-scale) precipitation rate (liq + ice)
* PS: Surface pressure
* PSL: sea level pressure
* QREFHT: Reference height humidity
* T200: temp at 200 mbar pressure surface
* T500: temp at 500 mbar pressure surface
* TMQ: Total (vertically integrated) precipitatable water
* TS: Surface temperature (radiative)
* U850: Zonal wind at 850 mbar pressure surface
* UBOT: Lowest model level zonal wind
* V850: Meridional wind at 850 mbar pressure surface
* VBOT: Lowest model level meridional wind
* Z100: Geopotential Z at 100 mbar pressure surface
* Z200: Geopotential Z at 200 mbar pressure surface
* ZBOT: Lowest model level height

4 measures per day, 365 days a year
resolution of 768 x 1152 equirectangular grid (25-km at equator)

boxes:
* ymin
* xmin
* ymax
* xmax
* class:
    * 0: Tropical Depression
    * 1: Tropical Cyclone
    * 2: Extratropical Cyclone
    * 3: Atmospheric River
"""
