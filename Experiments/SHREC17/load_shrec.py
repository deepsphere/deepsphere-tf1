#!/usr/bin/env python3
# coding: utf-8

"""Load dataset from SHREC17 and project it to a HEALpix sphere
    Code from: https://github.com/jonas-koehler/s2cnn/blob/master/examples/shrec17/dataset.py
    and https://github.com/AMLab-Amsterdam/lie_learn/blob/master/lie_learn/spaces/S2.py
    
    Use of Cohen equiangular files, and not created by us.
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

#import tensorflow as tf
from itertools import cycle
# To handle python 2
try:
    from itertools import zip_longest as zip_longest
except:
    from itertools import izip_longest as zip_longest

from scipy.spatial.distance import pdist, squareform

def shrec_output(descriptors, ids, probabilities, datapath, savedir='results_deep/test_perturbed'):
    os.makedirs(os.path.join(datapath, savedir), exist_ok=True)
    dist_mat = squareform(pdist(descriptors, 'cosine'))
    predictions = np.argmax(probabilities, axis=1)
    for dist, name, score in zip(dist_mat, ids, probabilities):
        most_feat = np.argsort(score)[::-1][0]
        retrieved = [(dist[j], ids[j]) for j in range(len(ids)) if predictions[j] == most_feat]
        thresh = np.median([ret[0] for ret in retrieved])  # need to change dynamically?
        retrieved += [(d, _id) for d, _id in zip(dist, ids) if d < thresh]
        retrieved = sorted(retrieved, reverse=True)
        retrieved = [i for _, i in retrieved]
        retrieved = np.array(retrieved)[sorted(np.unique(retrieved, return_index=True)[1])]
        idfile = os.path.join(datapath,savedir,name)
        with open(idfile, "w") as f:
            f.write("\n".join(retrieved))

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


def make_sgrid(nside, alpha, beta, gamma):

    npix = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix), nest=True)
#     _beta = np.pi * (2 * np.arange(2 * nside) + 1) / (4. * nside)
#     _alpha = np.arange(2 * nside) * np.pi / nside
#     theta, phi = np.meshgrid(*(_beta, _alpha),indexing='ij')
#     ct = np.cos(theta).flatten()
#     st = np.sin(theta).flatten()
#     cp = np.cos(phi).flatten()
#     sp = np.sin(phi).flatten()
#     x = st * cp
#     y = st * sp
#     z = ct
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords, dtype=np.float32)           # shape 3 x npix
    R = rotmat(alpha, beta, gamma, hom_coord=False)
    sgrid = np.einsum('ij,nj->ni', R, coords)    # inner(A,B).T

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
    dist_im = np.zeros(sgrid.shape[0])
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
        mesh.apply_transform(rnd_rot())

    r = np.max(np.linalg.norm(mesh.vertices, axis=-1))
    mesh.apply_scale(0.99 / r)

    return mesh


def ProjectOnSphere(nside, mesh, outside=False, multiple=False):
    ## outside = {'equator', 'pole', 'both'}
    if outside is 'equator':
#         rot = rnd_rot(-np.random.rand()*np.pi/4+np.pi/8,1,0)
        rot = rnd_rot(0,np.arccos(1-np.random.rand()*0.3)-np.pi/8,0)
        #mesh.apply_transform(rot)
        mesh.apply_translation([1.5, 0, 0])
        mesh.apply_transform(rot)
    if outside is 'pole':
#         mesh.apply_translation([0, 0, 2.])
        rot = rnd_rot(np.random.rand()*np.pi/4-np.pi/8,np.pi/2,0)
        mesh.apply_translation([1.5, 0, 0])
        mesh.apply_transform(rot.T)
    if outside is 'both':
#         rnd = np.random.rand()*2.
#         mesh.apply_translation([rnd, 0, np.sqrt(4-rnd**2)])
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
    # hull_im = hull_im.reshape(3, 2 * self.bandwidth, 2 * self.bandwidth)
    hull_im = hull_im.reshape(3, npix)

    im = np.concatenate([im, hull_im], axis=0)
    assert len(im) == 6

    im = im.astype(np.float32).T  # pylint: disable=E1101

    return im       # must be npix x nfeature

def fix_dataset(dir):
    """
    Remove unnecessary information from obj files
    """
    print("Fix obj files")

    r = re.compile(r'f (\d+)[/\d]* (\d+)[/\d]* (\d+)[/\d]*')

    path = os.path.join(dir, "*.obj")
    files = sorted(glob.glob(path))

    c = 0
    for i, f in enumerate(files):
        with open(f, "rt") as x:
            y = x.read()
            yy = r.sub(r"f \1 \2 \3", y)
            if y != yy:
                c += 1
                with open(f, "wt") as x:
                    x.write(yy)
        print("{}/{}  {} fixed    ".format(i + 1, len(files), c), end="\r")

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
    #norm = colors.LogNorm(vmin=cmin, vmax=cmax)
    #norm = colors.PowerNorm(gamma=4)
    hp.orthview(im1, title=id_im, nest=True, cmap=cm, min=cmin, max=cmax, **kwargs)
    plt.plot()
    if multiple:
        hp.orthview(data[:,1], title=id_im, nest=True, cmap=cm, min=cmin, max=cmax, norm=norm)
        plt.plot()
        hp.orthview(data[:,2], title=id_im, nest=True, cmap=cm, min=cmin, max=cmax, norm=norm)
    return im1

def cache_healpix_projection(root, dataset, nside, repeat=1, outside=False, rot=False):
    experiment = 'outside' if outside else 'inside'
    _dir = os.path.join(root, dataset + "_perturbed")
    files = sorted(glob.glob(os.path.join(_dir, '*.obj')))
    
    head, _ = os.path.split(files[0])
    os.makedirs(head+'/'+experiment, exist_ok=True)
    from tqdm import tqdm
    for file in tqdm(files):
        prefix = "nside{}_".format(nside)
        head, tail = os.path.split(file)
        _id, _ = os.path.splitext(tail)
        if outside:
            npy_path = os.path.join(head, experiment, prefix + _id + '_' + outside + '_{0}.npy')
        else:
            npy_path = os.path.join(head, experiment, prefix + _id + '_{0}.npy')
        for i in range(repeat):
            try:
                np.load(npy_path.format(i))
            except:
                try:
                    mesh = ToMesh(file, rot=rot, tr=0.)
                    data = ProjectOnSphere(nside, mesh, outside)
                except:
                    print("Exception during transform of {}".format(file))
                    raise
                if outside:
                    img = data[:,0]
                else:
                    img = data
                np.save(npy_path.format(i), img)
                

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


class Shrec17Dataset(object):
    '''
    Download SHREC17 and output spherical HEALpix maps of obj files
    * root = folder where data is stored
    * dataset ['train','test','val']
    * perturbed = use the perturbation dataset version
    * download = is the data already downloaded
    '''

    url_data = 'http://3dvision.princeton.edu/ms/shrec17-data/{}.zip'
    url_label = 'http://3dvision.princeton.edu/ms/shrec17-data/{}.csv'

    def __init__(self, root, dataset, perturbed=True, download=False, nside=1024, augmentation=1, nfeat=6,
                 nfile=2000, experiment = 'deepsphere', verbose=True, load=True):
        # nside is bw in case of equiangular experiment
        if not verbose:
            def fun(x):
                return x
        else:
            fun = tqdm
        self.experiment = experiment
        self.nside = nside
        self.nfeat = nfeat
        self.root = os.path.expanduser(root)
        self.repeat = augmentation

        if dataset not in ["train", "test", "val"]:
            raise ValueError("Invalid dataset")

        self.dir = os.path.join(self.root, dataset + ("_perturbed" if perturbed else ""))

        if download:
            self.download(dataset, perturbed)

        if not self._check_exists():
            print(self.dir)
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.files = sorted(glob.glob(os.path.join(self.dir, '*.obj')))
        if dataset != "test_pert":
            with open(os.path.join(self.root, dataset + ".csv"), 'rt') as f:
                reader = csv.reader(f)
                self.labels_dict = {}
                for row in [x for x in reader][1:]:
                    self.labels_dict[row[0]] = (row[1], row[2])
            self.labels = []
            for file in self.files:
                file = os.path.splitext(os.path.basename(file))[0]
                self.labels.append(self._target_transform(self.labels_dict[file]))
            self.labels = np.asarray(self.labels, dtype=int)
        else:
            self.labels = None
        head, _ = os.path.split(self.files[0])
        os.makedirs(head+'/'+experiment, exist_ok=True)
        if nfile is not None:
            self.files = self.files[:nfile]
            if self.labels is not None:
                self.labels = self.labels[:nfile]
        self.labels = self.labels.repeat(augmentation)
        self.ids = []
        if nfile is None or nfile < 0:
            nfile = len(self.files)
        if 'deepsphere' in experiment:
            self.data = np.zeros((nfile*augmentation, 12*nside**2, nfeat))       # N x npix x nfeature
            pass
        elif experiment is 'equiangular':
            self.data = np.zeros((nfile*augmentation, 4*nside**2, nfeat))
            pass
        for i, file in fun(enumerate(self.files)):
            if load:
                for j in range(augmentation):
                    self.ids.append(file.split('/')[-1].split('\\')[-1].split('.')[0])
            data = np.asarray(self.cache_npy(file, repeat=augmentation, experiment = experiment))
            #time1 = time.time()
            # must be smthg like (nbr map x nbr pixels x nbr feature)
            if load:
                self.data[augmentation*i:augmentation*(i+1)] = data[:,:,:nfeat]
            #time2 = time.time()
            #print("time elapsed for change elem:",(time2-time1)*1000.)
            del data
        if load:
        # better to remove mean before?
            file = root+"/info.pkl"
            try:
                info = pkl.load(open(file,'rb'))
            except:
                if verbose:
                    print("file non-existent")
                info = {}
            try:       
                self.mean = info[self.nside][dataset]['mean'][:nfeat]
                self.std = info[self.nside][dataset]['std'][:nfeat]
            except:
                if verbose:
                    print("info non-existent")
                self.std = np.std(self.data[::1,:,:], axis=(0, 1))
                self.mean = np.mean(self.data[::1,:,:], axis=(0, 1))
            self.data = self.data - self.mean
            self.data = self.data / self.std
            self.N = len(self.data)
            if self.nside in info.keys():
                info[self.nside][dataset]={"mean":self.mean,"std":self.std}
            else:
                info[self.nside] = {dataset:{"mean":self.mean,"std":self.std}}
            pkl.dump(info, open(file, 'wb'))
        

    def check_trans(self, file_path):
        # print("transform {}...".format(file_path))
        try:
            if self.experiment=='equiangular':
                raise NotImplementError("equiangular projection creation file not implemented yet")
            mesh = ToMesh(file_path, rot=False, tr=0.1)
            data = ProjectOnSphere(self.nside, mesh)
            return data
        except:
            print("Exception during transform of {}".format(file_path))
            raise

    def cache_npy(self, file_path, pick_randomly=False, repeat=1, experiment='deepsphere'):
        prefix = "nside{}_".format(self.nside)

        head, tail = os.path.split(file_path)
        root, _ = os.path.splitext(tail)
        npy_path = os.path.join(head, experiment, prefix + root + '_{0}.npy')
        if experiment is 'equiangular':
            prefix = "b{}_".format(self.nside)
            npy_path = os.path.join(head, prefix + root + '_{0}.npy')

        exists = [os.path.exists(npy_path.format(i)) for i in range(repeat)]

        if pick_randomly and all(exists):
            i = np.random.randint(repeat)
            try: return np.load(npy_path.format(i))
            except OSError: exists[i] = False

        if pick_randomly:
            img = self.check_trans(file_path)
            np.save(npy_path.format(exists.index(False)), img)

            return img

        output = []
        for i in range(repeat):
            try:
                img = np.load(npy_path.format(i))
                if experiment is 'equiangular':
                    img = img.reshape((6,-1)).T
            except (OSError, FileNotFoundError):
                img = self.check_trans(file_path)
                np.save(npy_path.format(i), img)
            output.append(img)

        return output

    def return_data(self, train=False, sigma=0., train_ratio=0.8, verbose=True):
        if train:
            ret = self._data_preprocess(self.data, sigma, train_ratio, verbose=verbose)
        else:
            #data = self.data.reshape((-1, self.repeat, 12*self.nside**2, 6))
            ret = self.data, self.labels, self.ids
            if verbose:
                self._print_histogram(self.labels)
        # features_train, labels_train, features_validation, labels_validation = ret
        return ret

    def _data_preprocess(self, x_raw_train, sigma_noise=0., train_ratio=0.8, verbose=True):
        if train_ratio == 1.0:
            p = np.random.permutation(len(x_raw_train))
            labels_train = self.labels[p]
            ids_train = np.asarray(self.ids)[p]
            if verbose:
                print('Number of elements / class')
                self._print_histogram(labels_train)
#             print('  Training set: ')
#             for i in range(self.nclass):
#                 print('    Class {}: {} elements'.format(i, np.sum(labels_train == i)), flush=True)
            return x_raw_train[p,:,:], labels_train, ids_train
        from sklearn.model_selection import train_test_split
        rs = np.random.RandomState(1)
        x_noise = x_raw_train + sigma_noise * rs.randn(*x_raw_train.shape)
        ret = train_test_split(x_raw_train, x_noise, self.labels, self.ids, test_size=None, train_size=train_ratio, shuffle=True, random_state=0)
        x_raw_train, x_raw_validation, x_noise_train, x_noise_validation, labels_train, labels_validation, ids_train, ids_val = ret
        if verbose:
            print('Number of elements / class')
            self._print_histogram(labels_train, labels_val)
#         print('  Training set: ')
#         for i in range(self.nclass):
#             print('    Class {}: {} elements'.format(i, np.sum(labels_train == i)), flush=True)

#         print('  Validation set: ')
#         for i in range(self.nclass):
#             print('    Class {}: {} elements'.format(i, np.sum(labels_validation == i)), flush=True)

        return x_raw_train, labels_train, x_noise_validation, labels_validation, ids_train, ids_val
    
    def _print_histogram(self, labels_train, labels_val=None):
        if labels_train is None:
            return
        import matplotlib.pyplot as plt
        from collections import Counter
        hist_train=Counter(labels_train)
#         for i in range(self.nclass):
#             hist_train.append(np.sum(labels_train == i))
        labels, values = zip(*hist_train.items())
        indexes = np.asarray(labels)
        width = 1
        plt.bar(indexes, values, width)
        plt.title("labels distribution")
        #plt.xticks(indexes + width * 0.5, labels)
        if labels_val is not None:
            hist_val=Counter(labels_val)
            plt.figure()
            labels, values = zip(*hist_val.items())
            indexes = np.asarray(labels)
            width = 1
            plt.bar(indexes, values, width)
            plt.title("validation labels distribution")
        plt.show()

    def _target_transform(self, target, reverse=False):
        classes = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657',
                   '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03001627', '03046257',
                   '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134',
                   '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244',
                   '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04379243',
                   '04401088', '04460130', '04468005', '04530566', '04554684']
        self.nclass = len(classes)
        if reverse:
            return classes[target]
        return classes.index(target[0])

    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir, "*.obj"))

        return len(files) > 0

    def _download(self, url):
        import requests

        filename = url.split('/')[-1]
        file_path = os.path.join(self.root, filename)

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

    def _unzip(self, file_path):
        import zipfile

        if os.path.exists(self.dir):
            return

        print('Unzip ' + file_path)

        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()
        os.unlink(file_path)

    def _fix(self):
        print("Fix obj files")

        r = re.compile(r'f (\d+)[/\d]* (\d+)[/\d]* (\d+)[/\d]*')

        path = os.path.join(self.dir, "*.obj")
        files = sorted(glob.glob(path))

        c = 0
        for i, f in enumerate(files):
            with open(f, "rt") as x:
                y = x.read()
                yy = r.sub(r"f \1 \2 \3", y)
                if y != yy:
                    c += 1
                    with open(f, "wt") as x:
                        x.write(yy)
            print("{}/{}  {} fixed    ".format(i + 1, len(files), c), end="\r")

    def download(self, dataset, perturbed):

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == os.errno.EEXIST:
                pass
            else:
                raise

        url = self.url_data.format(dataset + ("_perturbed" if perturbed else ""))
        file_path = self._download(url)
        self._unzip(file_path)
        self._fix()

        if dataset != "test":
            url = self.url_label.format(dataset)
            self._download(url)

        print('Done!')

        
class Shrec17DatasetCache(object):
    '''
    Download SHREC17 and output spherical HEALpix maps of obj files
    * root = folder where data is stored
    * dataset ['train','test','val']
    * perturbed = use the perturbation dataset version
    * download = is the data already downloaded
    '''

    url_data = 'http://3dvision.princeton.edu/ms/shrec17-data/{}.zip'
    url_label = 'http://3dvision.princeton.edu/ms/shrec17-data/{}.csv'

    def __init__(self, root, dataset, perturbed=True, download=False, nside=1024, nfeat=6,
                 augmentation=1, nfile=2000, experiment = 'deepsphere', verbose=True):
        self.experiment = experiment
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
        self.nside = nside
        self.nfeat = nfeat
        self.root = os.path.expanduser(root)
        self.repeat = augmentation

        if dataset not in ["train", "test", "val"]:
            raise ValueError("Invalid dataset")

        self.dir = os.path.join(self.root, dataset + ("_perturbed" if perturbed else ""))

        if download:
            self.download(dataset, perturbed)

        if not self._check_exists():
            print(self.dir)
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.files = sorted(glob.glob(os.path.join(self.dir, '*.obj')))
        if dataset != "test_pert":
            with open(os.path.join(self.root, dataset + ".csv"), 'rt') as f:
                reader = csv.reader(f)
                self.labels_dict = {}
                for row in [x for x in reader][1:]:
                    self.labels_dict[row[0]] = (row[1], row[2])
            self.labels = []
            for file in self.files:
                file = os.path.splitext(os.path.basename(file))[0]
                self.labels.append(self._target_transform(self.labels_dict[file]))
            self.labels = np.asarray(self.labels, dtype=int)
        else:
            self.labels = None
        head, _ = os.path.split(self.files[0])
        os.makedirs(head+'/'+experiment, exist_ok=True)
        if nfile is not None:
            self.files = self.files[:nfile]
            if self.labels is not None:
                self.labels = self.labels[:nfile]
        self.labels = self.labels.repeat(augmentation)
        self.ids = []
        if nfile is None:
            nfile = len(self.files)
        if nfile < 0:
            nfile = len(self.files) + nfile
        self.nfile = nfile
        self.augmentation = augmentation
        self.N = nfile * augmentation
        self.files = np.asarray(self.files).repeat(augmentation)
        
        if self.experiment == 'equator' or self.experiment == 'pole':
            self.outside = '_' + experiment
            self.experiment = 'outside'
        elif self.experiment == 'outside':
            self.N *=2
            self.outside = '_equator'
        #super(Shrec17DatasetCache, self).__init__()
            
        for i, file in enumerate(self.files):
            self.ids.append(file.split('/')[-1].split('\\')[-1].split('.')[0])
#             data = np.asarray(self.cache_npy(file, repeat=augmentation))
#             #time1 = time.time()
#             #self.data = np.vstack([self.data, data])       # must be smthg like (nbr map x nbr pixels x nbr feature)
#             self.data[augmentation*i:augmentation*(i+1)] = data
#             #time2 = time.time()
#             #print("time elapsed for change elem:",(time2-time1)*1000.)
#             del data
#         p = np.random.permutation(len(x_raw_train))
#         labels_train = self.labels[p]
#         ids_train = np.asarray(self.ids)[p]

    def get_labels(self, shuffle=True):
        if shuffle:
            p = self._p
        else:
            p = np.arange(self.N)
        return self.labels[p]
    
    def get_ids(self):
        return self.ids

    def iter(self, batch_size):
        return self.__iter__(batch_size)
    
    def __iter__(self, batch_size):
        #np.random.seed(42)
        if self.dataset is 'train':
            self._p = np.random.permutation(self.N)
        else:
            self._p = np.arange(self.N)
        self.ids = np.array(self.ids)[self._p]
        
        if batch_size>1:
#             if len(self._p)%batch_size != 0:
#                 _p = np.append(self._p, [None]*(batch_size-len(self._p)%batch_size))
#             else:
#                 _p = self._p
            _iter = grouper(cycle(self._p), batch_size)
        else:
            _iter = cycle(self._p)
        for p in _iter:
            data, label = self.get_item(p)
            data, label = np.array(data), np.array(label)
            if not self.loaded or self.experiment == 'outside':
                self.std = np.nanstd(data, axis=(0, 1))
                self.mean = np.nanmean(data, axis=(0, 1))
            data = data - self.mean
            data = data / self.std
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
            data = self.cache_npy(file, pick_randomly=False, repeat=self.augmentation, experiment=self.experiment)
            if self.experiment == 'outside':
                temp = data[elem%self.repeat]
                temp[np.where(temp==0.)]=np.nan
                datas.append(temp)
            else:
                datas.append(data[elem%self.repeat][:, :self.nfeat])
            #datas.append(self.cache_npy(file, pick_randomly=True, repeat=self.augmentation, experiment=self.experiment))
            labels.append(self.labels[elem])
        return datas, labels

    def check_trans(self, file_path):
        #print("transform {}...".format(file_path))
        try:
            if self.experiment=='equiangular':
                raise NotImplementError("equiangular projection creation file not implemented yet")
            mesh = ToMesh(file_path, rot=False, tr=0.1)
            data = ProjectOnSphere(self.nside, mesh)
            return data
        except:
            print("Exception during transform of {}".format(file_path))
            raise

    def cache_npy(self, file_path, pick_randomly=False, repeat=1, experiment='deepsphere'):
        prefix = "nside{}_".format(self.nside)

        head, tail = os.path.split(file_path)
        root, _ = os.path.splitext(tail)
        if experiment == 'outside':
            npy_path = os.path.join(head, experiment, prefix + root + self.outside + '_{0}.npy')
        else:
            npy_path = os.path.join(head, experiment, prefix + root + '_{0}.npy')
        if experiment is 'equiangular':
            prefix = "b{}_".format(self.nside)
            npy_path = os.path.join(head, prefix + root + '_{0}.npy')

        exists = [os.path.exists(npy_path.format(i)) for i in range(repeat)]

        if pick_randomly and all(exists):
            i = np.random.randint(repeat)
            try: return np.load(npy_path.format(i))
            except OSError: exists[i] = False

        if pick_randomly:
            img = self.check_trans(file_path)
            np.save(npy_path.format(exists.index(False)), img)

            return img

        output = []
        for i in range(repeat):
            try:
                img = np.load(npy_path.format(i))
                if experiment is 'equiangular':
                    img = img.reshape((6,-1)).T
            except (OSError, FileNotFoundError):
                img = self.check_trans(file_path)
                np.save(npy_path.format(i), img)
            output.append(img)

        return output

    def _target_transform(self, target, reverse=False):
        classes = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657',
                   '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03001627', '03046257',
                   '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134',
                   '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244',
                   '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04379243',
                   '04401088', '04460130', '04468005', '04530566', '04554684']
        self.nclass = len(classes)
        if reverse:
            return classes[target]
        return classes.index(target[0])

    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir, "*.obj"))

        return len(files) > 0

    def _download(self, url):
        import requests

        filename = url.split('/')[-1]
        file_path = os.path.join(self.root, filename)

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

    def _unzip(self, file_path):
        import zipfile

        if os.path.exists(self.dir):
            return

        print('Unzip ' + file_path)

        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()
        os.unlink(file_path)

    def _fix(self):
        print("Fix obj files")

        r = re.compile(r'f (\d+)[/\d]* (\d+)[/\d]* (\d+)[/\d]*')

        path = os.path.join(self.dir, "*.obj")
        files = sorted(glob.glob(path))

        c = 0
        for i, f in enumerate(files):
            with open(f, "rt") as x:
                y = x.read()
                yy = r.sub(r"f \1 \2 \3", y)
                if y != yy:
                    c += 1
                    with open(f, "wt") as x:
                        x.write(yy)
            print("{}/{}  {} fixed    ".format(i + 1, len(files), c), end="\r")

    def download(self, dataset, perturbed):

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == os.errno.EEXIST:
                pass
            else:
                raise

        url = self.url_data.format(dataset + ("_perturbed" if perturbed else ""))
        file_path = self._download(url)
        self._unzip(file_path)
        self._fix()

        if dataset != "test":
            url = self.url_label.format(dataset)
            self._download(url)

        print('Done!')


def grouper(iterable, n, fillvalue=None):
    """
    Collect data into fixed-length chunks or blocks.
    This function comes from itertools.
    """
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


class Shrec17DatasetTF():
    # TODO write TFrecords and read them for performance reasons
    def __init__(self, root, dataset, perturbed=True, download=False, nside=1024, nfeat = 6,
                 augmentation=1, nfile=2000, experiment = 'deepsphere', verbose=True):
        self.experiment = experiment
        self.nside = nside
        self.nfeat = nfeat
        self.root = os.path.expanduser(root)
        self.repeat = augmentation
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

        if dataset not in ["train", "test", "val"]:
            raise ValueError("Invalid dataset")

        self.dir = os.path.join(self.root, dataset + ("_perturbed" if perturbed else ""))

        if download:
            self.download(dataset, perturbed)

        if not self._check_exists():
            print(self.dir)
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.files = sorted(glob.glob(os.path.join(self.dir, '*.obj')))
        
        with open(os.path.join(self.root, dataset + ".csv"), 'rt') as f:
            reader = csv.reader(f)
            self.labels_dict = {}
            for row in [x for x in reader][1:]:
                self.labels_dict[row[0]] = self._target_transform(row[1])
#         self.labels = []
#         for file in self.files:
#             file = os.path.splitext(os.path.basename(file))[0]
#             self.labels.append(self._target_transform(self.labels_dict[file]))
#         self.labels = np.asarray(self.labels, dtype=int)
            
        head, _ = os.path.split(self.files[0])
        # os.makedirs(head+'/'+experiment, exist_ok=True)
        if nfile is not None:
            self.files = self.files[:nfile]
#             if self.labels is not None:
#                 self.labels = self.labels[:nfile]
#         self.labels = self.labels.repeat(augmentation)
#         self.ids = []
        if nfile is None or nfile < 0:
            nfile = len(self.files)
        self.nfile = nfile
        self.N = nfile * augmentation
        if self.experiment == 'all':
            self.experiment = 'deepsphere*'
            self.N *= 2
        if self.experiment == 'equator' or self.experiment == 'pole':
            self.outside = experiment
            self.experiment = 'outside'
        elif self.experiment == 'outside':
            self.N *=2
            self.outside = ''
#         self.files = np.asarray(self.files).repeat(augmentation)
            
#         for i, file in enumerate(self.files):
#             self.ids.append(file.split('/')[-1].split('\\')[-1].split('.')[0])

    def get_tf_dataset(self, batch_size, transform=None):
        if self.experiment == 'outside':
            file_pattern = os.path.join(self.dir, self.experiment, "nside{0}*_"+self.outside+"_{1}.npy")
        elif self.experiment == 'equiangular':
            file_pattern = os.path.join(self.dir, "b{0}*{1}.npy")
        else:
            file_pattern = os.path.join(self.dir, self.experiment, "nside{0}*{1}.npy")
        file_list = []
        for i in range(self.repeat):
            if transform:
                for j in range(5):
                    file_list+=glob.glob(file_pattern.format(self.nside, i))
            else:
                file_list+=glob.glob(file_pattern.format(self.nside, i))
        if len(file_list)==0:
            raise ValueError('Files not found')
        dataset = tf.data.Dataset.from_tensor_slices(file_list)
        
        self.noise = [None]*32
        
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
            try:
                batch_data = []
                batch_labels = []
                #for file in files:
                data = np.load(file.decode()).astype(np.float32)
                if self.experiment is 'equiangular':
                    data = data.reshape((6,-1)).T
                if self.experiment != 'outside':
                    data = data[:, :self.nfeat]
                    data = data - self.mean
                    data = data / self.std
                file = os.path.splitext(os.path.basename(file.decode()))[0].split("_")[1]
                label = self.labels_dict[file]
                data = data.astype(np.float32)
                if transform:
                    data, label = transform(data, label)
            except Exception as e:
                print(e)
                raise
#             batch_data.append(data.astype(np.float32))
#             batch_labels.append(label)
            return data, label
        
#         dataset = dataset.shuffle(buffer_size=self.N)
#         dataset = dataset.repeat()    # optional
#         if transform is None:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.N))
        #dataset = dataset.batch(batch_size).map(parse_fn, num_parallel_calls=4)  # change to py_function in future
        parse_fn = lambda file: tf.py_func(get_elem, [file], [tf.float32, tf.int64])
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=batch_size, drop_remainder = True))
#         else:
#             # must shuffle after the data augmentation
# #             dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.N*5))
#             parse_fn = lambda file: tf.py_func(get_elem, [file], [tf.float32, tf.int64])
# #             dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=batch_size, drop_remainder = True))
#             dataset = dataset.map(parse_fn, num_parallel_calls=8)
#             dataset = dataset.shuffle(buffer_size=self.N*5)
#             dataset = dataset.repeat()
#             dataset = dataset.batch(batch_size)
        self.dataset = dataset.prefetch(buffer_size=2)
        return self.dataset
        
    def _target_transform(self, target):
        classes = ['02691156', '02747177', '02773838', '02801938', '02808440', '02818832', '02828884', '02843684', '02871439', '02876657',
                   '02880940', '02924116', '02933112', '02942699', '02946921', '02954340', '02958343', '02992529', '03001627', '03046257',
                   '03085013', '03207941', '03211117', '03261776', '03325088', '03337140', '03467517', '03513137', '03593526', '03624134',
                   '03636649', '03642806', '03691459', '03710193', '03759954', '03761084', '03790512', '03797390', '03928116', '03938244',
                   '03948459', '03991062', '04004475', '04074963', '04090263', '04099429', '04225987', '04256520', '04330267', '04379243',
                   '04401088', '04460130', '04468005', '04530566', '04554684']
        self.nclass = len(classes)
        return classes.index(target)

    def _check_exists(self):
        files = glob.glob(os.path.join(self.dir, "*.obj"))

        return len(files) > 0

    def _download(self, url):
        import requests

        filename = url.split('/')[-1]
        file_path = os.path.join(self.root, filename)

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

    def _unzip(self, file_path):
        import zipfile

        if os.path.exists(self.dir):
            return

        print('Unzip ' + file_path)

        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(self.root)
        zip_ref.close()
        os.unlink(file_path)

    def _fix(self):
        print("Fix obj files")

        r = re.compile(r'f (\d+)[/\d]* (\d+)[/\d]* (\d+)[/\d]*')

        path = os.path.join(self.dir, "*.obj")
        files = sorted(glob.glob(path))

        c = 0
        for i, f in enumerate(files):
            with open(f, "rt") as x:
                y = x.read()
                yy = r.sub(r"f \1 \2 \3", y)
                if y != yy:
                    c += 1
                    with open(f, "wt") as x:
                        x.write(yy)
            print("{}/{}  {} fixed    ".format(i + 1, len(files), c), end="\r")

    def download(self, dataset, perturbed):

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(self.root)
        except OSError as e:
            if e.errno == os.errno.EEXIST:
                pass
            else:
                raise

        url = self.url_data.format(dataset + ("_perturbed" if perturbed else ""))
        file_path = self._download(url)
        self._unzip(file_path)
        self._fix()

        if dataset != "test":
            url = self.url_label.format(dataset)
            self._download(url)

        print('Done!')