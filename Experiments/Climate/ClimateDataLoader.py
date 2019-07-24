#!/usr/bin/env python3
# coding: utf-8

"""
    Load dataset in h5 files using s3
"""

import csv
import glob
import os
import re
import h5py
import numpy as np
from tqdm import tqdm

import time
import tensorflow as tf

#import tensorflow as tf
from itertools import cycle
# To handle python 2
try:
    from itertools import zip_longest as zip_longest
except:
    from itertools import izip_longest as zip_longest
    

class IcosahedronDataset():

    def __init__(self, path, partition):
        if partition not in ['train', 'val', 'test']:
            raise ValueError('Invalid dataset: {}'.format(partition))
        self.path = path
        self.partition = partition
        with open(path+partition+".txt", "r") as f:
            lines = f.readlines()
        self.filenames = [os.path.join(path, l.replace('\n', '')) for l in lines]
        if len(self.filenames)==0:
            raise ValueError('Files not found')
        self.N = len(self.filenames)
        
        
        self.precomp_mean = [26.160023, 0.98314494, 0.116573125, -0.45998842, 
                             0.1930554, 0.010749293, 98356.03, 100982.02, 
                             216.13145, 258.9456, 3.765611e-08, 288.82578, 
                             288.03925, 342.4827, 12031.449, 63.435772]
        self.precomp_std =  [17.04294, 8.164175, 5.6868863, 6.4967732, 
                             5.4465833, 0.006383436, 7778.5957, 3846.1863, 
                             9.791707, 14.35133, 1.8771327e-07, 19.866386, 
                             19.094095, 624.22406, 679.5602, 4.2283397]
        
        
#         data[partition] = {'data': np.zeros((len(flist),10242,16)),
#                            'labels': np.zeros((len(flist),10242))}
#         for i, f in enumerate(flist):
#             file = np.load(f)
#             data[partition]['data'][i] = (file['data'].T - precomp_mean) / precomp_std
#             data[partition]['labels'][i] = np.argmax(file['labels'].astype(np.int), axis=0)

    def get_tf_dataset(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        
        def get_elem(filename):
            try:
                file = np.load(filename.decode())#.astype(np.float32)
                data = file['data'].T
                data = data - self.precomp_mean
                data = data / self.precomp_std
                label = np.argmax(file['labels'].astype(np.int), axis=0)
                data = data.astype(np.float32)
            except Exception as e:
                print(e)
                raise
            return data, label
        
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.N))
        #dataset = dataset.batch(batch_size).map(parse_fn, num_parallel_calls=4)  # change to py_function in future
        parse_fn = lambda file: tf.py_func(get_elem, [file], [tf.float32, tf.int64])
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=batch_size, 
                                                              drop_remainder = self.partition=='train'))
        self.dataset = dataset.prefetch(buffer_size=4)
        return self.dataset


class EquiangularDataset():
    def __init__(self, path=None, s3=True):
        self.s3 = s3
        if s3:
#             import subprocess
            self.s3bucket = '10380-903b2ba14e0d980c25436f9ca5bb29f5'
            self.s3dir = 's3://{}/Datasets/Climate/'

#             cmd = 's3cmd ls '+self.s3dir.format(self.s3bucket)
#             filenames = subprocess.check_output(cmd, shell=True).decode()
            # without subprocess
            filenames = tf.gfile.ListDirectory(self.s3dir.format(self.s3bucket))  # maybe use Glob
            self.filenames = [self.s3dir.format(self.s3bucket)+elem for elem in fileneames]
#             self.filenames = [elem.split(' ')[-1] for elem in filenames.split('\n')[:-1]]
        else:
            self.filenames = glob.glob(path+'data*.h5')
        self.N = len(self.filenames)
        stats = h5py.File(path+'stats.h5')
        stats = stats['climate']["stats"]
        self.stats = stats
        self.mean = stats[:,0]
        self.std = stats[:,-1]
        
    def get_dataset(self, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.N))
        
        def s3_dataset(h5_file):
            sess.run(tf.io.read_file(h5_file.decode()))
            pass
        
        def local_dataset(h5_file):
            try:
                file = h5py.File(h5_file.decode())
                data = np.asarray(file['climate']["data"], dtype=np.float32).transpose(1,2,0)
                labels = np.asarray(file['climate']["labels"], dtype=np.int64)
                data = (data - self.mean)/self.std
            except KeyError:
                print(h5_file.decode())
                return
            except Exception as e:
                print(e)
                raise
            return data, labels
        
        if self.s3:
            parse_fn = lambda file: tf.py_func(s3_dataset, [file], [tf.float32, tf.int64])
        else:
            parse_fn = lambda file: tf.py_func(local_dataset, [file], [tf.float32, tf.int64])
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=batch_size))
        self.dataset = dataset.prefetch(buffer_size=4)
        return self.dataset
    
#     def get_dataset_v2(self, batch_size):
#         class generator:
#             def __call__(self, file):
#                 with h5py.File(file, 'r') as hf:
#                     for im in hf["climate"]:
#                         yield im

#         ds = tf.data.Dataset.from_tensor_slices(filenames)
#         ds = ds.interleave(lambda filename: tf.data.Dataset.from_generator(
#                 generator(), 
#                 tf.uint8, 
#                 tf.TensorShape([427,561,3]),
#                 args=(filename,)),
#                cycle_length, block_length)
    