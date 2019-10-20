#!/usr/bin/env python3
# coding: utf-8

import os, sys

import numpy as np


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
else:
    yearmin = 2010
    yearmax = 2015
    datapath = "./"

years = np.arange(yearmin, yearmax)

download(datapath+"raw/", years)
