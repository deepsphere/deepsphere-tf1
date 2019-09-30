#!/usr/bin/env python3
# coding: utf-8

# # Climate event detection
#
# https://portal.nersc.gov/project/dasrepo/deepcam/segm_h5_v3_reformat/gb_data_readme
#
# The 16 features are:
#
# * TMQ: Total (vertically integrated) precipitatable water
# * U850: Zonal wind at 850 mbar pressure surface
# * V850: Meridional wind at 850 mbar pressure surface
# * UBOT: Lowest model level zonal wind
# * VBOT: Lowest model level meridional wind
# * QREFHT: Reference height humidity
# * PS: Surface pressure
# * PSL: sea level pressure
# * T200: temp at 200 mbar pressure surface
# * T500: temp at 500 mbar pressure surface
# * PRECT: Total (convective and large-scale) precipitation rate (liq + ice)
# * TS: Surface temperature (radiative)
# * Z100: Geopotential Z at 100 mbar pressure surface
# * Z200: Geopotential Z at 200 mbar pressure surface
# * ZBOT: Lowest model level height
#
# resolution of 768 x 1152 equirectangular grid (25-km at equator)

# The labels are 0 for background class, 1 for tropical cyclone (TC), and 2 for atmoshperic river (AR)


import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import cartopy.crs as ccrs

import h5py


# plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{lmodern}')

# Load simulation data

year, month, day, hour, run = 2106, 1, 1, 0, 1
datapath = '../../data/Climate/data-{}-{:0>2d}-{:0>2d}-{:0>2d}-{}.h5'.format(year, month, day, hour, run)

h5f = h5py.File(datapath)
data = h5f['climate']["data"] # (16,768,1152) numpy array
labels = h5f['climate']["labels"] # (768,1152) numpy array

lon_ = np.arange(1152)/1152*360
lat_ = np.arange(768)/768*180-90
lon, lat = np.meshgrid(lon_, lat_)

# Figure

fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(90, 0))
ax.set_global()
ax.coastlines(linewidth=2)

scat1 = plt.scatter(lon, lat, s=1, rasterized=True,
            c=data[0,:,:], cmap=plt.get_cmap('RdYlBu_r'), alpha=1, transform=ccrs.PlateCarree())

AR = labels[:,:]==1
TC = labels[:,:]==2
scat2 = ax.scatter(lon[AR], lat[AR], s=0.5, color='c', label='AR', transform=ccrs.PlateCarree())
            #c=labels[show], cmap=plt.get_cmap('cool'), alpha=0.6, transform=ccrs.PlateCarree())
scat3 = ax.scatter(lon[TC], lat[TC], s=0.5, color='m', label='TC', transform=ccrs.PlateCarree())
            #c=labels[show], cmap=plt.get_cmap('cool'), alpha=0.6, transform=ccrs.PlateCarree())
ax.legend(markerscale=5, fontsize=10, loc=1, frameon=False, ncol=1, bbox_to_anchor=(0.1, 0.18))
ticks = range(np.min(data[0,:,:]).astype(int), np.max(data[0,:,:]).astype(int), 20)
cb = plt.colorbar(scat1, ax=ax, orientation="horizontal",anchor=(1.0,0.0), shrink=0.7, pad=0.05, ticks=ticks)
cb.ax.tick_params(labelsize=10)
cb.ax.set_xticklabels([f'${t}mm$' for t in ticks[1:]])

ax.text(0, 7e6, f'HAPPI20 Climate, TMQ, {year}-{month:02d}-{day:02d}-{hour:02d}-{run}', horizontalalignment='center')

#fig.tight_layout()
filename = os.path.splitext(os.path.basename(__file__))[0] + '.pdf'
fig.savefig(filename, transparent=True)
