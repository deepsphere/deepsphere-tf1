#!/usr/bin/env python3

"""
Script to download the main cosmological dataset.
The dataset is availlable at https://doi.org/10.5281/zenodo.1303272.
"""

import os
import sys
sys.path.append("../..")

from deepsphere import utils


if __name__ == '__main__':

    url_readme = 'https://zenodo.org/record/1303272/files/README.md?download=1'
    url_training = 'https://zenodo.org/record/1303272/files/training.zip?download=1'
    url_testing = 'https://zenodo.org/record/1303272/files/testing.zip?download=1'

    md5_readme = '6f52f6c2d8270907e7bc6bb852666b6f'
    md5_training = '6b0f5072481397fa8842ef99524b5482'
    md5_testing = '62757429ebb0a257c3d54775e08c9512'

    print('Download README')
    utils.download(url_readme, './')
    assert (utils.check_md5('README.md', md5_readme))

    print('Download training set')
    utils.download(url_training, './')
    assert (utils.check_md5('training.zip', md5_training))
    print('Extract training set')
    utils.unzip('training.zip', './')
    os.remove('training.zip')

    print('Download testing set')
    utils.download(url_testing, './')
    assert (utils.check_md5('testing.zip', md5_testing))
    print('Extract testing set')
    utils.unzip('testing.zip', './')
    os.remove('testing.zip')

    print('Dataset downloaded')
