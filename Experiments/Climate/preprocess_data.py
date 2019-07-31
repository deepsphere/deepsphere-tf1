#!/usr/bin/env python3
# coding: utf-8

"""
 Load dataset year by year, interpolate each map, and add label for each pixel.
 No special preprocessing for the labels, only bouding box
"""

import os

import numpy as np
# import healpy as hp
from itertools import product

# from tqdm import tqdm
# import h5py
# from scipy.interpolate import griddata, RegularGridInterpolator, NearestNDInterpolator



def download(datapath, url, info):
    import requests

    url = url.format(*info)
    
#     print(url)
    filename = url.split('/')[-1]
#     print(filename)
#     print(datapath)
    file_path = os.path.join(datapath, filename)

    if os.path.exists(file_path):
        return file_path


    r = requests.get(url, stream=True)
    r.raise_for_status()
#     if r.status_code == 404:
#         raise 
    
    print('Downloading ' + url)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return file_path

# def interpolate(images, labels):
#     channels, lat_x, lon_x = images.shape
#     lon_ = np.arange(lon_x)/lon_x*360
#     lat_ = np.arange(lat_x)/lat_x*180-90
#     lon, lat = np.meshgrid(*(lon_, lat_))
#     coords_map = np.stack([lon, lat], axis=-1).reshape((-1, 2))
#     Nside = [32]#[32, 64]
#     for nside in Nside:
#         print("preprocessing data at nside = {}".format(nside))
#         npix = hp.nside2npix(nside)
#         data = np.empty((npix, channels))
#         new_labels = np.empty((npix))
#         pix = np.arange(npix)
#         coords_hp = hp.pix2ang(nside, pix, nest=True, lonlat=True)
#         coords_hp = np.asarray(coords_hp).T
#         for channel in range(channels):
#             f = RegularGridInterpolator((lon_, lat_), images[channel].T)
#             data[:,channel] = f(coords_hp)
#         f = NearestNDInterpolator(coords_map, labels[:].flatten(), rescale=False)
#         new_labels = f(coords_hp)
            
        
#     return data, new_labels

def valid_days(data):
    valid = ~np.isnan(data).all(axis=(1,2,3))
    return valid

def clean_files():
    from glob import glob
    years = np.arange(2106, 2115)
    months = np.arange(1,13)
    days = np.arange(1,32)
    hours = np.arange(8)
    runs = np.arange(1,7)
    datapath = '../../data/Climate/'
    file = 'data-{}-{:0>2d}-{:0>2d}-{:0>2d}-*.h5'
    
    npix = hp.nside2npix(32)
    
    
    for year in years:
        for month, day, hour in product(months, days, hours):
            files = glob(datapath+file.format(year, month, day, hour))
            if len(files)>1:
                for elem in files[1:]:
                    print("remove file", elem)
                    os.remove(elem)
                

if __name__=='__main__':
#     clean_files()
    years = np.arange(2106, 2115)
    months = np.arange(1,13)
    days = np.arange(1,32)
    hours = np.arange(8)
    runs = np.arange(1,7)
    url = 'https://portal.nersc.gov/project/dasrepo/deepcam/segm_h5_v3_reformat/data-{}-{:0>2d}-{:0>2d}-{:0>2d}-{}.h5'
    datapath = '../../data/Climate/'
    file = 'data-{}-{:0>2d}-{:0>2d}-{:0>2d}-{}-32nside.npz'
    
    npix = hp.nside2npix(32)
    
    os.makedirs(datapath, exist_ok=True)
    h5_path = download(datapath, 'https://portal.nersc.gov/project/dasrepo/deepcam/segm_h5_v3_reformat/stats.h5', (None,))
    
    
    for year in years:
#         datas = np.ones((len(months), len(days), len(hours), 16, npix, len(runs))) * np.nan
#         print(datas.shape)
        for month, day, hour in product(months, days, hours):
            for run in runs:
                if os.path.exists(os.path.join(datapath, file.format(year, month, day, hour, run))):
                    continue
                try:
                    h5_path = download(datapath, url, (year, month, day, hour, run))
                except Exception as e:
    #                 print(e)
                    continue
#                 try:
#                     h5f = h5py.File(h5_path)
#                     # Features
#                     # [TMQ, U850, V850, UBOT, VBOT, QREFHT, PS, PSL, T200, T500, PRECT, TS, TREFHT, Z1000, Z200, ZBOT]
#                     data = h5f['climate']["data"]     # 16x768x1152  Features X lat X lon
#                 except:
#                     os.remove(h5_path)
#                     print("h5 file {} removed".format(h5_path))
#                     continue
#                 labels = h5f['climate']["labels"] # 768x1152     lat X lon
#                 data, labels = interpolate(data, labels)
    #             datas[month-1, day-1, hour, :, :, run-1] = interpolate(data)
#                 if year>2106:
#                     os.remove(h5_path)
#                     print("h5 file removed")
    #         datas = datas.reshape(-1, 16, npix, len(runs))
    #         print(datas.shape)
    #         print(valid_days(datas).shape)
    #         datas = datas[valid_days(datas)]
#                 np.savez(datapath+file.format(year, month, day, hour, run), datas=data, labels=labels)
#                 print("save file at: "+file.format(year, month, day, hour, run))
#                 break
    
    
    
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

resolution of 768 x 1152 equirectangular grid (25-km at equator)


* class:
    * 0: Nothing
    * 1: Tropical Cyclone
    * 2: Atmospheric River
"""
