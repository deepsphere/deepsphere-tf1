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
        self.rotmat = lambda lat: np.array([[np.cos(np.pi*lat/180),np.sin(np.pi*lat/180)],
                                            [-np.sin(np.pi*lat/180),np.cos(np.pi*lat/180)]])
        
        from pygsp.graphs import SphereIcosahedron as icosahedron_graph
        g = icosahedron_graph(5)
        self.lat = np.rad2deg(g.lat)
        del g
        
        
#         data[partition] = {'data': np.zeros((len(flist),10242,16)),
#                            'labels': np.zeros((len(flist),10242))}
#         for i, f in enumerate(flist):
#             file = np.load(f)
#             data[partition]['data'][i] = (file['data'].T - precomp_mean) / precomp_std
#             data[partition]['labels'][i] = np.argmax(file['labels'].astype(np.int), axis=0)

    def get_tf_dataset(self, batch_size, transform=False):
        dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        
        def get_elem(filename, transform=transform):
            try:
                file = np.load(filename.decode())#.astype(np.float32)
                data = file['data'].T
                data = data - self.precomp_mean
                data = data / self.precomp_std
                label = np.argmax(file['labels'].astype(np.int), axis=0)
                if transform:
                    data[:,1:3] = np.squeeze((data[:,np.newaxis,1:3] @ self.rotmat(self.lat).transpose(2,0,1)))
                    data[:,3:5] = np.squeeze((data[:,np.newaxis,3:5] @ self.rotmat(self.lat).transpose(2,0,1)))
#                 data[:,1:3] = np.linalg.norm(data[:,1:3]) # @ self.rotmat
#                 data[:,3:5] = np.linalg.norm(data[:,3:5]) # @ self.rotmat
#                 data[:,1] = np.arctan2(data[:,1], data[:,2])
#                 data[:,2] = data[:,1]
#                 data[:,3] = np.arctan2(data[:,3], data[:,4])
#                 data[:,4] = data[:,3]
                data = data.astype(np.float32)
            except Exception as e:
                print(e)
                raise
            return data, label
        if self.partition == 'train':
            dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.N))
        else:
            dataset = dataset.repeat()
        #dataset = dataset.batch(batch_size).map(parse_fn, num_parallel_calls=4)  # change to py_function in future
        parse_fn = lambda file: tf.py_func(get_elem, [file], [tf.float32, tf.int64])
        dataset = dataset.map(parse_fn, num_parallel_calls=batch_size*1).batch(batch_size, drop_remainder = self.partition=='train')
        # dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=batch_size, drop_remainder = self.partition=='train'))
        self.dataset = dataset.prefetch(buffer_size=8)
        return self.dataset


class EquiangularDataset():
    def __init__(self, path=None, partition='train', s3=True):
        self.s3 = s3
        if partition not in ['train', 'val', 'test']:
            raise ValueError('invalid partition: {}'.format(partition))
        self.partition = partition
        with open(path+partition+".txt", "r") as f:
            lines = f.readlines()
        self.filenames = [os.path.join(path, l.replace('\n', '')) for l in lines]
        if len(self.filenames)==0:
            raise ValueError('Files not found')
        if s3:
#             import subprocess
            self.s3bucket = '10380-903b2ba14e0d980c25436f9ca5bb29f5'
            self.s3dir = 's3://{}/Datasets/Climate/'

#             cmd = 's3cmd ls '+self.s3dir.format(self.s3bucket)
#             filenames = subprocess.check_output(cmd, shell=True).decode()
            # without subprocess
            self.filenames = tf.gfile.Glob(self.s3dir.format(self.s3bucket)+'data*')
#             filenames = tf.gfile.ListDirectory(self.s3dir.format(self.s3bucket))  # maybe use Glob
#             self.filenames = [self.s3dir.format(self.s3bucket)+elem for elem in filenames]
#             self.filenames = [elem.split(' ')[-1] for elem in filenames.split('\n')[:-1]]
        else:
            filenames = glob.glob(path+'data*.h5')
            self.filenames = list(set(self.filenames) & set(filenames))
        if len(self.filenames)==0:
            raise ValueError('No files in partition {}'.format(self.partition))
        self.N = len(self.filenames)
        fstats = h5py.File(path+'stats.h5', 'r')
        stats = fstats['climate']["stats"]
        self.mean = stats[:,0]
        self.std = stats[:,-1]
        fstats.close()
        
    def get_tf_dataset(self, batch_size, dtype=tf.float32):
        dataset = tf.data.Dataset.from_tensor_slices(self.filenames)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.N))
        
        def s3_dataset(h5_file):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            try:
                fdata = sess.run(tf.io.read_file(h5_file.decode()))
                file = h5py.File(fdata)
#                 print(file.keys())
                data = np.asarray(file['climate']["data"], dtype=np.float32).transpose(1,2,0)
#                 data[:,1:3] = data[:,1:3] @ self.rotmat
#                 data[:,3:5] = data[:,3:5] @ self.rotmat
                data = data.reshape(-1, 16)
                labels = np.asarray(file['climate']["labels"], dtype=np.int64)
                labels = labels.reshape(-1)
                data = (data - self.mean)/self.std
            except Exception as e:
                print(e)
                raise
            return data, labels
        
        def local_dataset(h5_file):
            try:
#                 start = time.time()
                file = h5py.File(h5_file.decode())
                data = np.asarray(file['climate']["data"], dtype=np.float32).transpose(1,2,0)
                data = data.reshape(-1, 16)
                labels = np.asarray(file['climate']["labels"], dtype=np.int64)
                labels = labels.reshape(-1)
                data = (data - self.mean)/self.std
#                 print("time preprocess: ", time.time()-start)
            except KeyError:
                print(h5_file.decode())
                return
            except Exception as e:
                print(e)
                raise
            return data, labels
        
        if self.s3:
            parse_fn = lambda file: tf.py_func(s3_dataset, [file], [dtype, tf.int64])
        else:
            parse_fn = lambda file: tf.py_func(local_dataset, [file], [dtype, tf.int64])
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=batch_size))
        self.dataset = dataset.prefetch(buffer_size=8)
        return self.dataset
    
    def get_dataset_s3(self, batch_size):
        np.random.shuffle(self.filenames)
        dataset = tf.data.FixedLengthRecordDataset(self.filenames, 60171866)
        # shuffle filenames instead of data
        dataset = dataset.repeat()
        # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.N))
        
        def s3_dataset(h5_file):
            import io
            try:
                start = time.time()
                h5 = io.BytesIO(h5_file)
                file = h5py.File(h5)
#                 print("time open bytes: ", time.time()-start)
                data = np.asarray(file['climate']["data"], dtype=np.float32).transpose(1,2,0)
                labels = np.asarray(file['climate']["labels"], dtype=np.int64)
                data = (data - self.mean)/self.std
                print("time preprocess: ", time.time()-start)
            except Exception as e:
                print(e)
                raise
            return data, labels
        parse_fn = lambda file: tf.py_func(s3_dataset, [file], [tf.float32, tf.int64])
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=batch_size))
        self.dataset = dataset.prefetch(buffer_size=16)
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
    
