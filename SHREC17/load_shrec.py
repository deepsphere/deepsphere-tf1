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

#import tensorflow as tf
from itertools import cycle
# To handle python 2
try:
    from itertools import zip_longest as zip_longest
except:
    from itertools import izip_longest as zip_longest

from scipy.spatial.distance import pdist, squareform
    
def shrec_output(probabilities, ids, datapath, savedir='results_deep/test_perturbed'):
    # TODO: descriptors are not probabilities, but the step before the fully connected layer
    os.makedirs(os.path.join(datapath, savedir), exist_ok=True)
    dist_mat = squareform(pdist(probabilities, 'cosine'))
    predictions = np.argmax(probabilities, axis=1)
    for dist, name, score in zip(dist_mat, ids, probabilities):
        most_feat = np.argsort(score)[::-1][0]  # equal to prediction in actual configuration
        retrieved = [(dist[j], ids[j]) for j in range(len(ids)) if predictions[j] == most_feat]
        thresh = np.median([ret[0] for ret in retrieved])  # need to change dinamically?
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

    #theta, phi = S2.meshgrid(b=b, grid_type='Healpix')
    #sgrid = S2.change_coordinates(np.c_[theta[..., None], phi[..., None]], p_from='S', p_to='C')
    npix = hp.nside2npix(nside)
    x, y, z = hp.pix2vec(nside, np.arange(npix), nest=True)
    #sgrid = np.stack([x, y, z])
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords, dtype=np.float32)           # shape 3 x npix
    #sgrid = sgrid.T#reshape((-1, 3))

    R = rotmat(alpha, beta, gamma, hom_coord=False)
    sgrid = np.einsum('ij,nj->ni', R, coords)    # inner(A,B).T

    return sgrid


# def make_sgrid_daniildis(res=64, alpha=0, beta=0, gamma=0):     # nchannel = 1, 2 if d, sin(alpha) input_res=64, batch size 9843???
#     # res x res equiangular grid
#     beta = np.arange(2 * res) * np.pi / (2. * res)  # Driscoll-Heally
#     alpha = np.arange(2 * res) * np.pi / res
#     # beta = np.arange(1,2 * res, 2) * np.pi / (2 * res - 1)   # equiangular?
#     # alpha = 2 * np.pi * np.arange(2*res - 1) / (2*res-1)
#     theta, phi = np.meshgrid(*(beta, alpha),indexing='ij')
#     out = np.empty(theta.shape + (3,))
#     ct = np.cos(theta)
#     st = np.sin(theta)
#     cp = np.cos(phi)
#     sp = np.sin(phi)
#     x = st * cp
#     y = st * sp
#     z = ct
#     out[..., 0] = x
#     out[..., 1] = y
#     out[..., 2] = z
#     return out


def render_model(mesh, sgrid):

    # Cast rays
    # triangle_indices = mesh.ray.intersects_first(ray_origins=sgrid, ray_directions=-sgrid)
    index_tri, index_ray, loc = mesh.ray.intersects_id(
        ray_origins=sgrid, ray_directions=-sgrid, multiple_hits=False, return_locations=True)
    loc = loc.reshape((-1, 3))  # fix bug if loc is empty

    # Each ray is in 1-to-1 correspondence with a grid point. Find the position of these points
    grid_hits = sgrid[index_ray]
    grid_hits_normalized = grid_hits / np.linalg.norm(grid_hits, axis=1, keepdims=True)

    # Compute the distance from the grid points to the intersection pionts
    dist = np.linalg.norm(grid_hits - loc, axis=-1)

    # For each intersection, look up the normal of the triangle that was hit
    normals = mesh.face_normals[index_tri]
    normalized_normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Construct spherical images
    dist_im = np.zeros(sgrid.shape[0])
    dist_im[index_ray] = dist
    # dist_im = dist_im.reshape(theta.shape)

    # shaded_im = np.zeros(sgrid.shape[0])
    # shaded_im[index_ray] = normals.dot(light_dir)
    # shaded_im = shaded_im.reshape(theta.shape) + 0.4

    n_dot_ray_im = np.zeros(sgrid.shape[0])
    # n_dot_ray_im[index_ray] = np.abs(np.einsum("ij,ij->i", normals, grid_hits_normalized))
    n_dot_ray_im[index_ray] = np.einsum("ij,ij->i", normalized_normals, grid_hits_normalized)   # sum(A*B,axis=1)

    nx, ny, nz = normalized_normals[:, 0], normalized_normals[:, 1], normalized_normals[:, 2]
    gx, gy, gz = grid_hits_normalized[:, 0], grid_hits_normalized[:, 1], grid_hits_normalized[:, 2]
    wedge_norm = np.sqrt((nx * gy - ny * gx) ** 2 + (nx * gz - nz * gx) ** 2 + (ny * gz - nz * gy) ** 2)
    n_wedge_ray_im = np.zeros(sgrid.shape[0])
    n_wedge_ray_im[index_ray] = wedge_norm

    # Combine channels to construct final image
    # im = dist_im.reshape((1,) + dist_im.shape)
    im = np.stack((dist_im, n_dot_ray_im, n_wedge_ray_im), axis=0)

    return im


def rnd_rot():
    a = np.random.rand() * 2 * np.pi
    z = np.random.rand() * 2 - 1
    c = np.random.rand() * 2 * np.pi
    rot = rotmat(a, np.arccos(z), c, True)
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


def ProjectOnSphere(nside, mesh, outside=False):
    ## outside = {'equator', 'pole', 'both'}
    if outside is 'equator':
        mesh.apply_translation([1., 0, 0])
    if outside is 'pole':
        mesh.apply_translation([0, 0, 1.])
    if outside is 'both':
        rnd = np.random.rand()*1.
        mesh.apply_translation([rnd, 0, np.sqrt(1-rnd**2)])
    sgrid = make_sgrid(nside, alpha=0, beta=0, gamma=0)
    im = render_model(mesh, sgrid)
    npix = sgrid.shape[0]
    im = im.reshape(3, npix)

    from scipy.spatial.qhull import QhullError  # pylint: disable=E0611
    try:
        convex_hull = mesh.convex_hull
    except QhullError:
        convex_hull = mesh

    hull_im = render_model(convex_hull, sgrid)
    # hull_im = hull_im.reshape(3, 2 * self.bandwidth, 2 * self.bandwidth)
    hull_im = hull_im.reshape(3, npix)

    im = np.concatenate([im, hull_im], axis=0)
    assert len(im) == 6

    # im[0] -= 0.75
    # im[0] /= 0.26
    # im[1] -= 0.59
    # im[1] /= 0.50
    # im[2] -= 0.54
    # im[2] /= 0.29
    # im[3] -= 0.52
    # im[3] /= 0.19
    # im[4] -= 0.80
    # im[4] /= 0.18
    # im[5] -= 0.51
    # im[5] /= 0.25

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

def plot_healpix_projection(file, nside, outside=False, rot=True):
    import matplotlib.pyplot as plt
    try:
        mesh = ToMesh(file, rot=rot, tr=0.)
        data = ProjectOnSphere(nside, mesh, outside)
    except:
        print("Exception during transform of {}".format(file))
        raise
    im1 = data[:,0]
    id_im = os.path.splitext(os.path.basename(file))[0]
    cm = plt.cm.RdBu_r
    cm.set_under('w')
    cmin = np.min(im1)
    cmax = np.max(im1)
    hp.orthview(im1, title=id_im, nest=True, cmap=cm, min=cmin, max=cmax)
    plt.plot()
    return im1


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

    def __init__(self, root, dataset, perturbed=True, download=False, nside=1024, augmentation=1, 
                 nfile=2000, experiment = 'deepsphere', verbose=True, load=True):
        # nside is bw in case of equiangular experiment
        if not verbose:
            def fun(x):
                return x
        else:
            fun = tqdm
        self.nside = nside
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
            self.data = np.zeros((nfile*augmentation, 12*nside**2, 6))       # N x npix x nfeature
            pass
        elif experiment is 'equiangular':
            self.data = np.zeros((nfile*augmentation, 4*nside**2, 6))
            pass
        for i, file in fun(enumerate(self.files)):
            if load:
                for j in range(augmentation):
                    self.ids.append(file.split('/')[-1].split('\\')[-1].split('.')[0])
            data = np.asarray(self.cache_npy(file, repeat=augmentation, experiment = experiment))
            #time1 = time.time()
            # must be smthg like (nbr map x nbr pixels x nbr feature)
            if load:
                self.data[augmentation*i:augmentation*(i+1)] = data
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
                self.mean = info[self.nside][dataset]['mean']
                self.std = info[self.nside][dataset]['std']
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

    def __init__(self, root, dataset, perturbed=True, download=False, nside=1024, 
                 augmentation=1, nfile=2000, experiment = 'deepsphere', verbose=True):
        self.experiment = experiment
        self.dataset = dataset
#         if experiment is 'deepsphere':
#             if nside==32:
#                 if dataset is 'train':
#                     self.mean = np.array([ 0.76423665, -0.08207781,  0.58800055,  0.57971995,  0.74730743, 0.58085993])
#                     self.std = np.array([0.27203578, 0.7116338 , 0.28400871, 0.22612295, 0.2069686, 0.2432338 ])
#                 elif dataset is 'val':
#                     self.mean = np.array([ 0.77143387, -0.06222059, 0.58987524, 0.58569188, 0.74413875, 0.5852325])
#                     self.std = np.array([0.26941943, 0.71316672, 0.28338772, 0.22418983, 0.20745156, 0.24289361])
#         else:
#             self.mean = 0.
#             self.std = 1.
        file = root+"/info.pkl"
        try:
            info = pkl.load(open(file,'rb'))
            self.mean = info[nside][dataset]['mean']
            self.std = info[nside][dataset]['std']
            self.loaded = True
        except:
            self.mean = 0.
            self.std = 1.
            self.loaded = False
            if verbose:
                print("no information currently available")
        self.nside = nside
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
            if not self.loaded:
                self.std = np.std(data[::1,:,:], axis=(0, 1))
                self.mean = np.mean(data[::1,:,:], axis=(0, 1))
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
            datas.append(data[elem%self.repeat])
            #datas.append(self.cache_npy(file, pick_randomly=True, repeat=self.augmentation, experiment=self.experiment))
            labels.append(self.labels[elem])
        return datas, labels

    def check_trans(self, file_path):
        #print("transform {}...".format(file_path))
        try:
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
    def __init__(self, root, dataset, perturbed=True, download=False, nside=1024, 
                 augmentation=1, nfile=2000, experiment = 'deepsphere', verbose=True):
        self.experiment = experiment
        self.nside = nside
        self.root = os.path.expanduser(root)
        self.repeat = augmentation
        file = root+"/info.pkl"
        try:
            info = pkl.load(open(file,'rb'))
            self.mean = info[nside][dataset]['mean']
            self.std = info[nside][dataset]['std']
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
        os.makedirs(head+'/'+experiment, exist_ok=True)
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
#         self.files = np.asarray(self.files).repeat(augmentation)
            
#         for i, file in enumerate(self.files):
#             self.ids.append(file.split('/')[-1].split('\\')[-1].split('.')[0])

    def get_tf_dataset(self, batch_size):
        file_pattern = os.path.join(self.dir, self.experiment, "nside{0}*{1}.npy")
        file_list = []
        for i in range(self.repeat):
            file_list+=glob.glob(file_pattern.format(self.nside, self.repeat))
        dataset = tf.data.Dataset.from_tensor_slices(file_list)

        def get_elem(file):
            batch_data = []
            batch_labels = []
            #for file in files:
            data = np.load(file.decode()).astype(np.float32)
            data = data - self.mean
            data = data / self.std
            file = os.path.splitext(os.path.basename(file.decode()))[0].split("_")[1]
            label = self.labels_dict[file]
#             batch_data.append(data.astype(np.float32))
#             batch_labels.append(label)
            return data.astype(np.float32), label
        
#         dataset = dataset.shuffle(buffer_size=self.N)
#         dataset = dataset.repeat()    # optional
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(self.N))
        parse_fn = lambda file: tf.py_func(get_elem, [file], [tf.float32, tf.int64])
        #dataset = dataset.batch(batch_size).map(parse_fn, num_parallel_calls=4)  # change to py_function in future
        dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_fn, batch_size=batch_size, drop_remainder = True))
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