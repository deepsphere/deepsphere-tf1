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
from itertools import product

from tqdm import tqdm
import h5py
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

def interpolate(images):
    channels, lat_x, lon_x = images.shape
    lon_ = np.arange(lon_x)/lon_x*360
    lat_ = np.arange(lat_x)/lat_x*180-90
    lon, lat = np.meshgrid(*(lon_, lat_))
    Nside = 32#[32, 64]
    for nside in Nside:
        print("preprocessing data at nside = {}".format(nside))
        npix = hp.nside2npix(nside)
        data = np.empty((npix, channels))
        pix = np.arange(npix)
        coords_hp = hp.pix2ang(nside, pix, nest=True, lonlat=True)
        coords_hp = np.asarray(coords_hp).T
        for channel in range(channels):
            f = RegularGridInterpolator((lon_, lat_), images[channel].T)
            data[:,channel] = f(coords_hp)
            
        
    pass

def valid_days(data):
    valid = ~np.isnan(data).all(axis=0)
    return valid

if __name__=='__main__':
    years = np.arange(2106, 2115)
    months = np.arange(1,13)
    days = np.arange(1,32)
    hours = np.arange(8)
    runs = np.arange(7)
    url = 'https://portal.nersc.gov/project/dasrepo/deepcam/segm_h5_v3_reformat/data-{}-{:0>2d}-{:0>2d}-{:0>2d}-{}.h5'
    datapath = '../../data/Climate/'
    file = 'EW_32nside_{}.npz'
    
    npix = hp.nside2npix(32)
    
    h5_path = download(datapath, 'https://portal.nersc.gov/project/dasrepo/deepcam/segm_h5_v3_reformat/stats.h5', year)
    
    for year in years:
        datas = np.ones(len(months), len(days), len(hours), 16, npix, len(runs)) * np.nan
        for month, day, hour, run in product(months, days, hours, runs):
            if os.path.exists(os.path.join(datapath, file.format(year, month, day, hour, run))):
                continue
            try:
                h5_path = download(datapath, url, year)
            except:
                continue
            h5f = h5py.File(h5_path)
            # Features
            # [TMQ, U850, V850, UBOT, VBOT, QREFHT, PS, PSL, T200, T500, PRECT, TS, TREFHT, Z1000, Z200, ZBOT]
            data = h5f['climate']["data"]     # 16x768x1152  Features X lat X lon
            labels = h5f['climate']["labels"] # 768x1152     lat X lon
            datas[year, month, day, hour, :, :, run] = interpolate(data)
            os.remove(h5_path)
            print("h5 file removed")
        datas = datas.reshape(-1, 16, npix, len(runs))
        datas = datas[valid_days(datas)]
        np.savez(file.format()year, datas=datas, labels=labels)
        print("save file at: "+file)
    
    
    
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
