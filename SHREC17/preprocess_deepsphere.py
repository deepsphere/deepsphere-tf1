#!/usr/bin/env python3
# coding: utf-8

from load_shrec import Shrec17DeepSphere as shrecSet

data_path = '../data/shrec17/'

train_dataset = shrecSet(data_path, 'train', nside=32, augmentation=5)