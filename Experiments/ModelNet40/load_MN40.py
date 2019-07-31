#!/usr/bin/env python3
# coding: utf-8

"""Load dataset from SHREC17 and project it to a HEALpix sphere
    Code from: https://github.com/jonas-koehler/s2cnn/blob/master/examples/shrec17/dataset.py
    and https://github.com/AMLab-Amsterdam/lie_learn/blob/master/lie_learn/spaces/S2.py
"""

import csv
import glob
import os
import re
import numpy as np
import trimesh
import healpy as hp
from tqdm import tqdm

import time
import pickle as pkl
import tensorflow as tf

from itertools import cycle
# To handle python 2
try:
    from itertools import zip_longest as zip_longest
except:
    from itertools import izip_longest as zip_longest
    
    
def rotmat(a, b, c, hom_coord=False):   # apply to mesh using mesh.apply_transform(rotmat(a,b,c, True))
    """
    Create a rotation matrix with an optional fourth homogeneous coordinate
    :param a, b, c: ZYZ-Euler angles
    """
    def z(a):
        return np.array([[np.cos(a), np.sin(a), 0, 0],
                         [-np.sin(a), np.cos(a), 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    def y(a):
        return np.array([[np.cos(a), 0, np.sin(a), 0],
                         [0, 1, 0, 0],
                         [-np.sin(a), 0, np.cos(a), 0],
                         [0, 0, 0, 1]])

    r = z(a).dot(y(b)).dot(z(c))  # pylint: disable=E1101
    if hom_coord:
        return r
    else:
        return r[:3, :3]


def make_sgrid(nside, alpha=0, beta=0, gamma=0):

    npix = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix), nest=True)
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords, dtype=np.float32)           # shape 3 x npix

#     R = rotmat(alpha, beta, gamma, hom_coord=False)
#     sgrid = np.einsum('ij,nj->ni', R, coords)    # inner(A,B).T
    sgrid = coords
    return sgrid

def render_model(mesh, sgrid, outside=False, multiple=False):

    # Cast rays
    # triangle_indices = mesh.ray.intersects_first(ray_origins=sgrid, ray_directions=-sgrid)
    if outside:
        index_tri, index_ray, loc = mesh.ray.intersects_id(
            ray_origins=(sgrid-sgrid), ray_directions=sgrid, multiple_hits=multiple, return_locations=True)
    else:
        index_tri, index_ray, loc = mesh.ray.intersects_id(
            ray_origins=sgrid, ray_directions=-sgrid, multiple_hits=multiple, return_locations=True)
    loc = loc.reshape((-1, 3))  # fix bug if loc is empty
    
    if multiple:
        grid_hits = sgrid[index_ray]
        if outside:
            dist = np.linalg.norm(loc, axis=-1)
        else:
            dist = np.linalg.norm(grid_hits - loc, axis=-1)
        dist_im = np.ones((sgrid.shape[0],3))*-1
        for index in range(np.max(index_ray)+1):
            for i, ind in enumerate(np.where(index_ray==index)[0]):
                if dist[ind] > 1:
                    continue
                try:
                    dist_im[index, i] = dist[ind]
                except:
                    pass
        return dist_im
                
#         max_index = np.argsort(index_ray)[1]
#         s=np.sort(index_ray)
#         print(s[:-1][s[1:] == s[:-1]])
#         index_tri_mult, index_mult, loc_mult = index_tri[max_index:], index_ray[max_index:], loc[max_index:]
#         index_tri, index_ray, loc = index_tri[:max_index], index_ray[:max_index], loc[:max_index]
    
    # Each ray is in 1-to-1 correspondence with a grid point. Find the position of these points
    grid_hits = sgrid[index_ray]
    grid_hits_normalized = grid_hits / np.linalg.norm(grid_hits, axis=1, keepdims=True)
    

    # Compute the distance from the grid points to the intersection pionts
    if outside:
        dist = np.linalg.norm(loc, axis=-1)
    else:
        dist = np.linalg.norm(grid_hits - loc, axis=-1)

    # For each intersection, look up the normal of the triangle that was hit
    normals = mesh.face_normals[index_tri]
    normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Construct spherical images
    dist_im = np.ones(sgrid.shape[0])
    dist_im[index_ray] = dist
    # dist_im = dist_im.reshape(theta.shape)

    n_dot_ray_im = np.zeros(sgrid.shape[0])
    n_dot_ray_im[index_ray] = np.einsum("ij,ij->i", normalized_normals, grid_hits_normalized)   # sum(A*B,axis=1)

    nx, ny, nz = normalized_normals[:, 0], normalized_normals[:, 1], normalized_normals[:, 2]
    gx, gy, gz = grid_hits_normalized[:, 0], grid_hits_normalized[:, 1], grid_hits_normalized[:, 2]
    wedge_norm = np.sqrt((nx * gy - ny * gx) ** 2 + (nx * gz - nz * gx) ** 2 + (ny * gz - nz * gy) ** 2)
    n_wedge_ray_im = np.zeros(sgrid.shape[0])
    n_wedge_ray_im[index_ray] = wedge_norm

    # Combine channels to construct final image
    im = np.stack((dist_im, n_dot_ray_im, n_wedge_ray_im), axis=0)

    return im


def rnd_rot(a=None, z=None, c=None):
    if a is None:
        a = np.random.rand() * 2 * np.pi
    if z is None:
        z = np.arccos(np.random.rand() * 2 - 1)
    if c is None:
        c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, z, c, True)
    return rot

def ToMesh(path, rot=False, tr=0.):
    '''
    * rot = random rotations, boolean
    * tr = random translation, amount of translation max vector
    '''
    mesh = trimesh.load_mesh(path)
    mesh.remove_degenerate_faces()
    mesh.fix_normals()
    mesh.fill_holes()
    mesh.remove_duplicate_faces()
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    
    mesh.apply_translation(-mesh.centroid)

    r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
    mesh.apply_scale(1 / r)

    if tr > 0:
        tr = np.random.rand() * tr
        rotR = rnd_rot()
        mesh.apply_transform(rotR)
        mesh.apply_translation([tr, 0, 0])

        if not rot:
            mesh.apply_transform(rotR.T)

    if rot:
        mesh.apply_transform(rnd_rot())  #z=np.pi, c=0

    r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
    mesh.apply_scale(0.99 / r)
#     mesh.remove_degenerate_faces()
    mesh.fix_normals()
#     mesh.fill_holes()
#     mesh.remove_duplicate_faces()
#     mesh.remove_infinite_values()
#     mesh.remove_unreferenced_vertices()
    
    return mesh

def ProjectOnSphere(nside, mesh, outside=False, multiple=False):
    ## outside = {'equator', 'pole', 'both'}
    if outside is 'equator':
        rot = rnd_rot(0,np.arccos(1-np.random.rand()*0.3)-np.pi/8,0)
        mesh.apply_translation([1.5, 0, 0])
        mesh.apply_transform(rot)
    if outside is 'pole':
        rot = rnd_rot(np.random.rand()*np.pi/4-np.pi/8,np.pi/2,0)
        mesh.apply_translation([1.5, 0, 0])
        mesh.apply_transform(rot.T)
    if outside is 'both':
        mesh.apply_translation([1.5, 0, 0])
        mesh.apply_transform(rnd_rot(0,-np.random.rand()*np.pi/2,0))
    sgrid = make_sgrid(nside, alpha=0, beta=0, gamma=0)
    im = render_model(mesh, sgrid, outside=outside, multiple=multiple)
    if multiple:
        return im.astype(np.float32)
    npix = sgrid.shape[0]
    im = im.reshape(3, npix)

    from scipy.spatial.qhull import QhullError  # pylint: disable=E0611
    try:
        convex_hull = mesh.convex_hull
    except QhullError:
        convex_hull = mesh

    hull_im = render_model(convex_hull, sgrid, outside=outside, multiple=multiple)
    hull_im = hull_im.reshape(3, npix)

    im = np.concatenate([im, hull_im], axis=0)
    assert len(im) == 6

    im = im.astype(np.float32).T  # pylint: disable=E1101

    return im       # must be npix x nfeature

def check_trans(nside, file_path, rot=False):
        print("transform {}...".format(file_path))
        try:
            mesh = ToMesh(file_path, rot=rot, tr=0.)
            data = ProjectOnSphere(nside, mesh)
            return data
        except Exception as e:
            print(e)
            print("Exception during transform of {}".format(file_path))
            raise
            
def compute_mean_std(dataset, name, root, nside, delete=False):
    dataset.mean = 0.
    dataset.std = 1.
    dataset.loaded = True
    data_iter = dataset.iter(1)
    N = dataset.N
    file = os.path.join(root, 'info.pkl')
    try:
        info = pkl.load(open(file,'rb'))
    except:
        print("file non-existent")
        info = {}
    if delete:
        if nside in info.keys():
            info[nside].pop(name, None)
        return
    mean = 0.
    std = 1.
    for i in tqdm(range(N)):
        data, _ = next(data_iter)
        mean += np.mean(data, axis=(0,1))
    mean /= N
    
    for i in tqdm(range(N)):
        data, _ = next(data_iter)
        std += ((data - mean)**2).mean(axis=(0,1))
    std /= N
    std = np.sqrt(std)
    
    if nside in info.keys():
        info[nside][name]={"mean":mean,"std":std}
    else:
        info[nside] = {name:{"mean":mean,"std":std}}
    pkl.dump(info, open(file, 'wb'))
    dataset.mean = mean
    dataset.std = std
#     print(mean)
#     print(std)
    return mean, std


def plot_healpix_projection(file, nside, outside=False, rotp=True, multiple=False, **kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    try:
        mesh = ToMesh(file, rot=rotp, tr=0.)
        data = ProjectOnSphere(nside, mesh, outside, multiple)
    except:
        print("Exception during transform of {}".format(file))
        raise
    im1 = data[:,0]
    id_im = os.path.splitext(os.path.basename(file))[0]
    cm = plt.cm.RdBu_r
    cm.set_under('w')
    cmin = np.min(im1)
    cmax = np.max(im1)
    im1[im1>cmax] = np.nan
    #norm = colors.LogNorm(vmin=cmin, vmax=cmax)
    #norm = colors.PowerNorm(gamma=4)
    hp.orthview(im1, title='', nest=True, cmap=cm, min=cmin, max=cmax, **kwargs)
    plt.plot()
    if multiple:
        hp.orthview(data[:,1], title=id_im, nest=True, cmap=cm, min=cmin, max=cmax, norm=norm)
        plt.plot()
        hp.orthview(data[:,2], title=id_im, nest=True, cmap=cm, min=cmin, max=cmax, norm=norm)
    return im1


class ModelNet40DatasetCache():
    
    def __init__(self, root, dataset, nside=32, nfeat = 6,
                 augmentation=1, nfile=2000, experiment = 'deepsphere', fix=False, verbose=True):
        self.experiment = experiment
        self.nside = nside
        self.nfeat = nfeat
        self.root = os.path.expanduser(root)
        self.repeat = augmentation
        self.dataset = dataset
        file = root+"/info.pkl"
        try:
            info = pkl.load(open(file,'rb'))
            self.mean = info[nside][dataset]['mean'][:nfeat]
            self.std = info[nside][dataset]['std'][:nfeat]
            self.loaded = True
        except:
            self.mean = 0.
            self.std = 1.
            self.loaded = False
            if verbose:
                print("no information currently available")

        classes = sorted(glob.glob(os.path.join(self.root, '*/')))
        self.classes = [os.path.split(clas[:-1])[-1] for clas in classes]
        self.nclass = len(self.classes)
        self.dir = os.path.join(self.root, "{}", dataset)
        
        if dataset not in ["train", "test"]:
            raise ValueError("Invalid dataset")

        if not self._check_exists():
            print(self.dir)
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.files = []
        self.labels = []    # might be utterly useless
        for i, _class in enumerate(self.classes):
            files = sorted(glob.glob(os.path.join(self.dir.format(_class), '*.off')))
            self.files += files
            self.labels += [i]*len(files)
            
        # self.nfile = len(self.files)
        self.proc_dir = os.path.join(self.root, dataset)[1:]
        os.makedirs(self.proc_dir + '/' + experiment, exist_ok=True)
        if nfile is not None:
            self.files = self.files[:nfile]

        if nfile is None or nfile < 0:
            nfile = len(self.files)
        self.nfile = nfile
        self.N = nfile * augmentation
        self.files = np.asarray(self.files).repeat(augmentation)
        self.labels = np.asarray(self.labels).repeat(augmentation)
        if self.experiment == 'all':
            self.experiment = 'deepsphere*'
            self.N *= 3 
            self.N += nfile
        
        if fix:
            self._fix()
        
        self.transform = None
        self.limit = None
        self.old_N = self.N
            
            
    def get_labels(self, shuffle=True):
        if shuffle:
            p = self._p
        else:
            p = np.arange(self.N)
        return self.labels[p]
    
    def set_transform(self, transform):
        "give a transform function for augmentation purpose"
        self.transform = transform
        
    def reduce_dataset(self, limit):
        self.limit = limit
        if limit:
            self.N = limit
        else:
            self.N = self.old_N

    def iter(self, batch_size, shuffle=True):
        return self.__iter__(batch_size, shuffle)
    
    def __iter__(self, batch_size, shuffle=True):
        np.random.seed(42)
        if self.dataset is 'train' and shuffle:
            self._p = np.random.permutation(self.old_N)
        else:
            self._p = np.arange(self.old_N)
        
        if self.limit:
            self._p = self._p[:self.limit]
        
        self.ids = self.files[self._p]
        
        if batch_size>1:
            _iter = grouper(cycle(self._p), batch_size)
        else:
            _iter = cycle(self._p)
        for p in _iter:
            data, label = self.get_item(p)
            data, label = np.array(data), np.array(label)
            if not self.loaded:
                self.std = np.std(data[::1,:,:], axis=(0, 1))
                self.mean = np.mean(data[::1,:,:], axis=(0, 1))
#                 import warnings
#                 with warnings.catch_warnings():
#                     warnings.filterwarnings('error')
#                     try:
#                         (data - self.mean)/self.std
#                     except Warning:
#                         print(self.mean)
#                         print(self.std)
#                         print(self.files[p])
#                         file = self.files[p]
#                         suffix = os.path.splitext(os.path.split(file)[-1])[0]
#                         pattern = "nside{}_{}_{}.npy".format(self.nside, suffix, p%self.repeat)
#                         npy_path = os.path.join(self.proc_dir, self.experiment, pattern)
#                         os.remove(npy_path)
            data = data - self.mean
            data = data / self.std
#             if np.std(data[0,:,0])>2:
#                 print(np.std(data[0,:,0]))
#                 print(self.files[p])
#                 data *= self.std
#                 data += self.mean
            if self.transform:
                data = self.transform(data)
            yield data, label
    
    def get_item(self, p):
        datas = []
        labels = []
        if type(p) is not tuple:
            p = (p,)
        for elem in p:
#             if elem is None:
#                 continue
            file = self.files[elem]
            data = self.cache_npy(file, pick_randomly=False, repeat=self.repeat, experiment=self.experiment)
            datas.append(data[elem%self.repeat][:, :self.nfeat])
            #datas.append(self.cache_npy(file, pick_randomly=True, repeat=self.augmentation, experiment=self.experiment))
            labels.append(self.labels[elem])
#             if 'car_0229' in file:
#                 print(np.std(data[elem%self.repeat][:,0]))
            if np.std(data[elem%self.repeat][:,0])>0.7 or data[elem%self.repeat][:,0].max()>2:
                suffix = os.path.splitext(os.path.split(file)[-1])[0]
                pattern = "nside{}_{}_{}.npy".format(self.nside, suffix, elem%self.repeat)
                npy_path = os.path.join(self.proc_dir, self.experiment, pattern)
                os.remove(npy_path)
                print(npy_path)
        return datas, labels
    
    def get_npy_file(self, files):
        datas = []
        for file in files:
            data = self.cache_npy(file, pick_randomly=False, repeat=self.repeat, experiment=self.experiment)
            data = (np.stack(data)[:, :, :self.nfeat]-self.mean)/self.std
            if self.transform:
                data = self.transform(data)
            datas.append(data)
        return datas
            
    def cache_npy(self, file_path, pick_randomly=False, repeat=1, experiment='deepsphere'):    
        
        suffix = os.path.splitext(os.path.split(file_path)[-1])[0]
        if 'equiangular' in experiment:
            pattern = "b"
        else:
            pattern = "nside"
        pattern += "{}_{}_".format(self.nside, suffix)
        npy_path = os.path.join(self.proc_dir, self.experiment, pattern)
        npy_path += "{}.npy"
        _class = '_'.join(suffix.split("_")[:-1])
#         label = self.classes.index(_class)

        exists = [os.path.exists(npy_path.format(i)) for i in range(repeat)]

        if pick_randomly and all(exists):
            i = np.random.randint(repeat)
            try: return np.load(npy_path.format(i))
            except OSError: exists[i] = False

        if pick_randomly:
            img = check_trans(self.nside, file_path, rot=('rot' in experiment))
            np.save(npy_path.format(exists.index(False)), img)

            return img

        output = []
        for i in range(repeat):
            try:
                img = np.load(npy_path.format(i))
                if experiment is 'equiangular':
                    img = img.reshape((6,-1)).T
            except (OSError, FileNotFoundError):
                img = check_trans(self.nside, file_path, rot=('rot' in experiment))
                np.save(npy_path.format(i), img)
            output.append(img)

        return output
    
    
    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir.format(self.classes[0]), "*.off"))

        return len(files) > 0
    
    def _fix(self):
        print("Fix off files")
        
        r = re.compile(r'OFF[\n]?(-?\d+) (-?\d+) (-?\d+)')

        c = 0
        for i, f in enumerate(self.files):
            with open(f, "rt") as x:
#                 for line in x:
#                     print(line)
#                     if line != r:
#                         print("something")
#                     break
                y = x.read()
                yy = r.sub(r"OFF\n\1 \2 \3", y)
                if y != yy:
                    c += 1
                    with open(f, "wt") as x:
                        x.write(yy)
        print("{}/{}  {} fixed    ".format(i + 1, len(self.files), c), end="\r")


class ModelNet40DatasetTF():
    # TODO write TFrecords and read them for performance reasons
    def __init__(self, root, dataset, nside=1024, nfeat = 6,
                 augmentation=1, nfile=2000, experiment = 'deepsphere', fix=False, verbose=True):
        self.experiment = experiment
        self.nside = nside
        self.nfeat = nfeat
        self.root = os.path.expanduser(root)
        self.repeat = augmentation
        self.dataset = dataset
        file = root+"/info.pkl"
        try:
            info = pkl.load(open(file,'rb'))
            self.mean = info[nside][dataset]['mean'][:nfeat]
            self.std = info[nside][dataset]['std'][:nfeat]
            self.loaded = True
        except:
            self.mean = 0.
            self.std = 1.
            self.loaded = False
            if verbose:
                print("no information currently available")

        classes = sorted(glob.glob(os.path.join(self.root, '*/')))
        self.classes = [os.path.split(clas[:-1])[-1] for clas in classes]
        self.nclass = len(self.classes)
        self.dir = os.path.join(self.root, "{}", dataset)
        
        if dataset not in ["train", "test"]:
            raise ValueError("Invalid dataset")

        if not self._check_exists():
            print(self.dir)
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.files = []
        self.labels = []    # might be utterly useless
        for i, _class in enumerate(self.classes):
            files = sorted(glob.glob(os.path.join(self.dir.format(_class), '*.off')))
            self.files += files
            self.labels += [i]*len(files)
            
        # self.nfile = len(self.files)
        self.proc_dir = os.path.join(self.root, dataset)[1:]
        os.makedirs(self.proc_dir + '/' + experiment, exist_ok=True)
        if nfile is not None:
            self.files = self.files[:nfile]

        if nfile is None or nfile < 0:
            nfile = len(self.files)
        self.nfile = nfile
        self.N = nfile * augmentation
        if self.experiment == 'all':
            self.experiment = 'deepsphere*'
            self.N *= 5
        
        if fix:
            self._fix()
#         self.files = np.asarray(self.files).repeat(augmentation)
            
#         for i, file in enumerate(self.files):
#             self.ids.append(file.split('/')[-1].split('\\')[-1].split('.')[0])

    def get_tf_dataset(self, batch_size, transform=None):
#         file_pattern = os.path.join(self.dir, self.experiment, "nside{0}*{1}.npy")
#         file_list = []
#         for i in range(self.repeat):
#             if transform:
#                 for j in range(5):
#                     file_list+=glob.glob(file_pattern.format(self.nside, i))
#             else:
#                 file_list+=glob.glob(file_pattern.format(self.nside, i))
        file_list = [os.path.splitext(os.path.split(file)[-1])[0] for file in self.files] * (5 if transform else 1) * (self.repeat) * (3 if '*' in self.experiment else 1)
        if len(file_list)==0:
            raise RunTimeError('Files not found')
        dataset = tf.data.Dataset.from_tensor_slices(file_list)
        
        self.noise = [None]*32
        if '*' in self.experiment:
            list_dir = glob.glob(os.path.join(self.proc_dir, self.experiment))
            self.list_dir = [os.path.split(_dir)[-1] for _dir in list_dir]
            self.list_dir.remove('deepsphere_notr')
        
        def add_noise(data, label):
            size = data.shape
            if any(elem is None for elem in self.noise):
                index = 10 - sum(elem is None for elem in self.noise)#self.noise.index(None)
                self.noise[index] = np.random.normal(size=size, scale=0.1).astype(np.float32)
                data=data + self.noise[index].astype(np.float32)
            else:
                data = data + self.noise[int(np.random.rand()*10)].astype(np.float32)
            return data, label

        if transform is True:
            self.N = len(file_list)
            transform = add_noise
        
        def get_elem(file, transform=transform):
            if '*' in self.experiment:
                i = np.random.randint(4)
                experiment = self.list_dir[i]
            else:
                experiment = self.experiment
            pattern = "nside{}_{}_".format(self.nside, file.decode())
            file_path = os.path.join(self.proc_dir, experiment, pattern)
            file_path += "{}.npy"
            _class = '_'.join(file.decode().split("_")[:-1])
            label = self.classes.index(_class)
            img = cache_npy(file_path, file, _class)
            data = img.astype(np.float32)
            data = data[:, :self.nfeat]
            data = data - self.mean
            data = data / self.std
            data = data.astype(np.float32)
            if transform:
                data, label = transform(data, label)
            return data, label
        
        def get_elem_batch(batch_file):
            batch_data = []
            batch_labels = []
            for file in batch_file:
                data, label = get_elem(file)
                batch_data.append(data)
                batch_labels.append(label)
            return batch_data, batch_labels
        
        def cache_npy(file_path, file, _class):
            exists = [os.path.exists(file_path.format(i)) for i in range(self.repeat)]
            if all(exists):
                i = np.random.randint(self.repeat)
                try: return np.load(file_path.format(i))
                except OSError: exists[i] = False
            
#             print(os.path.join(self.dir.format(_class), file.decode()))
#             print(file_path.format(exists.index(False)))
            img = check_trans(self.nside, os.path.join(self.dir.format(_class), file.decode())+'.off', rot=('rot' in self.experiment))
            np.save(file_path.format(exists.index(False)), img)
            return img
        
#         dataset = dataset.shuffle(buffer_size=self.N)
#         dataset = dataset.repeat()    # optional
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.N))
        parse_fn_batch = lambda file: tf.py_func(get_elem, [file], [tf.float32, tf.int64])   # doesn't seem to work anymore. where is the bug?
        parse_fn = lambda file: tf.py_func(get_elem, [file], [tf.float32, tf.int64]) # change to py_function in future
        dataset = dataset.map(parse_fn, num_parallel_calls=batch_size*1).batch(batch_size, drop_remainder=False)  
#         dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=batch_size, drop_remainder = False))

        self.dataset = dataset.prefetch(buffer_size=4)
        return self.dataset
    
        
#     def check_trans(self, file_path):
#         #print("transform {}...".format(file_path))
#         try:
#             mesh = ToMesh(file_path, rot=False, tr=0.1)
#             data = ProjectOnSphere(self.nside, mesh)
#             return data
#         except:
#             print("Exception during transform of {}".format(file_path))
#             raise

    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir.format(self.classes[0]), "*.off"))

        return len(files) > 0
    
    def _fix(self):
        print("Fix off files")
        
        r = re.compile(r'OFF[\n]?(-?\d+) (-?\d+) (-?\d+)')

        c = 0
        for i, f in enumerate(self.files):
            with open(f, "rt") as x:
#                 for line in x:
#                     print(line)
#                     if line != r:
#                         print("something")
#                     break
                y = x.read()
                yy = r.sub(r"OFF\n\1 \2 \3", y)
                if y != yy:
                    c += 1
                    with open(f, "wt") as x:
                        x.write(yy)
        print("{}/{}  {} fixed    ".format(i + 1, len(self.files), c), end="\r")
        
        
def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks.
    This function comes from itertools.
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)
