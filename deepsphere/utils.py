"""Utilities module."""

from __future__ import division

from builtins import range
import os
import sys
import hashlib
import zipfile

import numpy as np
from scipy import sparse, spatial
import matplotlib.pyplot as plt
import healpy as hp

from pygsp.graphs import NNGraph, SphereIcosahedron, SphereEquiangular, SphereHealpix

if sys.version_info[0] > 2:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve    
    
    
    
def healpix_weightmatrix(nside=16, nest=True, indexes=None, dtype=np.float32, std=None, full=False):
    """Return an unnormalized weight matrix for a graph using the HEALPIX sampling.

    Parameters
    ----------
    nside : int
        The healpix nside parameter, must be a power of 2, less than 2**30.
    nest : bool, optional
        if True, assume NESTED pixel ordering, otherwise, RING pixel ordering
    indexes : list of int, optional
        List of indexes to use. This allows to build the graph from a part of
        the sphere only. If None, the default, the whole sphere is used.
    dtype : data-type, optional
        The desired data type of the weight matrix.
    """
    if not nest:
        raise NotImplementedError()

    if indexes is None:
        indexes = range(nside**2 * 12)
    npix = len(indexes)  # Number of pixels.
    if npix >= (max(indexes) + 1):
        # If the user input is not consecutive nodes, we need to use a slower
        # method.
        usefast = True
        indexes = range(npix)
    else:
        usefast = False
        indexes = list(indexes)

    # Get the coordinates.
    x, y, z = hp.pix2vec(nside, indexes, nest=nest)
    coords = np.vstack([x, y, z]).transpose()
    coords = np.asarray(coords, dtype=dtype)
    # Get the 7-8 neighbors.
    if full:
        distances = spatial.distance.cdist(coords, coords)**2
    else:
        neighbors = hp.pixelfunc.get_all_neighbours(nside, indexes, nest=nest)
        # if use_4:
        #     print('Use 4')
        #     col_index = []
        #     row_index = []
        #     for el,neighbor in zip(indexes,neighbors.T):
        #         x, y, z = hp.pix2vec(nside, [el], nest=nest)
        #         coords_1 = np.vstack([x, y, z]).transpose()
        #         coords_1 = np.array(coords_1)

        #         x, y, z = hp.pix2vec(nside, neighbor, nest=nest)
        #         coords_2 = np.vstack([x, y, z]).transpose()
        #         coords_2 = np.asarray(coords_2)
        #         ind = np.argsort(np.sum((coords_2-coords_1)**2,axis=1),)[:4]
        #         col_index = col_index + neighbor[ind].tolist()
        #         row_index = row_index +[el]*4
        #     col_index = np.array(col_index)
        #     row_index = np.array(row_index)
        # else:
        # Indices of non-zero values in the adjacency matrix.
        col_index = neighbors.T.reshape((npix * 8))
        row_index = np.repeat(indexes, 8)

        # Remove pixels that are out of our indexes of interest (part of sphere).
        if usefast:
            keep = (col_index < npix)
            # Remove fake neighbors (some pixels have less than 8).
            keep &= (col_index >= 0)
            col_index = col_index[keep]
            row_index = row_index[keep]
        else:
            col_index_set = set(indexes)
            keep = [c in col_index_set for c in col_index]
            inverse_map = [np.nan] * (nside**2 * 12)
            for i, index in enumerate(indexes):
                inverse_map[index] = i
            col_index = [inverse_map[el] for el in col_index[keep]]
            row_index = [inverse_map[el] for el in row_index[keep]]

        # Compute Euclidean distances between neighbors.
        distances = np.sum((coords[row_index] - coords[col_index])**2, axis=1)
    # slower: np.linalg.norm(coords[row_index] - coords[col_index], axis=1)**2

    # Compute similarities / edge weights.
    if std is None:
        kernel_width = np.mean(distances)
    else:
        kernel_width = std
    weights = np.exp(-distances / (2 * kernel_width))
    
#     weights[weights>0]=1

    # Similarity proposed by Renata & Pascal, ICCV 2017.
    # weights = 1 / distances

    # Build the sparse matrix.
    if full:
        W = weights
        for i in range(np.alen(W)):
            W[i, i] = 0.
        k = 0.01#np.exp(-5)
        W[W < k] = 0
        W = sparse.csr_matrix(W, dtype=dtype)
    else:
        W = sparse.csr_matrix(
            (weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype)
    
    # if use_4:
    #     W = (W+W.T)/2

    return W


def equiangular_weightmatrix(bw=64, indexes=None, dtype=np.float32):
    if indexes is None:
        indexes = range((2*bw)**2)
    npix = len(indexes)  # Number of pixels.
    
    # Find a mean to take only indexes from grid
    #beta = np.arange(2 * bw) * np.pi / (2. * bw)  # Driscoll-Heally
    #alpha = np.arange(2 * bw) * np.pi / bw
    beta = np.pi * (2 * np.arange(2 * bw) + 1) / (4. * bw) #SOFT
    alpha = np.arange(2 * bw) * np.pi / bw
    theta, phi = np.meshgrid(*(beta, alpha),indexing='ij')
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    x = st * cp
    y = st * sp
    z = ct
    coords = np.vstack([x.flatten(), y.flatten(), z.flatten()]).transpose() 
    coords = np.asarray(coords, dtype=dtype)
    npix = len(coords)
    
#     distances = spatial.distance.cdist(coords, coords)**2
    
    def south(x, bw):
        if x >= npix - 2*bw:
            return (x + bw)%(2*bw) + npix - 2*bw
        else:
            return x + 2*bw
        
    def north(x, bw):
        if x < 2*bw:
            return (x + bw)%(2*bw)
        else:
            return x - 2*bw
        
    def west(x, bw):
        if x%(2*bw)==0:
            x += 2*bw
        return x -1
    
    def east(x, bw):
        if x%(2*bw)==2*bw-1:
            x -= 2*bw
        return x + 1
        
    neighbors = []
    col_index=[]
    for ind in indexes:
        # first line is the same point, so is connected to all points of second line
#         if ind < 2* bw:
#             neighbor = np.arange(2*bw)+2*bw
#         elif ind < 4*bw:
#             neighbor = [south(west(ind,bw),bw), west(ind,bw), east(ind,bw), south(east(ind,bw),bw), south(ind,bw)]
#             neighbor += list(range(2*bw))
#             #print(neighbor)
#         else:
        neighbor = [south(west(ind,bw),bw), west(ind,bw), north(west(ind,bw), bw), north(ind,bw), 
                        north(east(ind,bw),bw), east(ind,bw), south(east(ind,bw),bw), south(ind,bw)]
        neighbors.append(neighbor)
        col_index += list(neighbor)
    # neighbors = np.asarray(neighbors)
    col_index = np.asarray(col_index)
    
    #col_index = neighbors.reshape((-1))
#     row_index = np.hstack([np.repeat(indexes[:2*bw], 2*bw), np.repeat(indexes[2*bw:4*bw], 2*bw+5), 
#                           np.repeat(indexes[4*bw:], 8)])
    row_index = np.hstack([np.repeat(indexes, 8)])
    
    distances = np.sum((coords[row_index] - coords[col_index])**2, axis=1)
    # slower: np.linalg.norm(coords[row_index] - coords[col_index], axis=1)**2

    # Compute similarities / edge weights.
#     kernel_width = np.mean(distances)
#     weights = np.exp(-distances / (2 * kernel_width))

    # Similarity proposed by Renata & Pascal, ICCV 2017.
    weights = 1 / distances

    # Build the sparse matrix.
    W = sparse.csr_matrix(
        (weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype)
#     W=weights
#     for i in range(np.alen(W)):
#         W[i, i] = 0.
# #     k = np.exp(0)
# #     W[W < k] = 0
#     W = sparse.csr_matrix(W, dtype=dtype)
#     # adjustments
#     mat = W[:2*bw,2*bw:4*bw]*5/(2*bw)
#     W[:2*bw,2*bw:4*bw] = mat
#     W[2*bw:4*bw,:2*bw] = mat.T
    
    return W

def build_laplacian(W, lap_type='normalized', dtype=np.float32):
    """Build a Laplacian (tensorflow)."""
    d = np.ravel(W.sum(1))
    if lap_type == 'combinatorial':
        D = sparse.diags(d, 0, dtype=dtype)
        return (D - W).tocsc()
    elif lap_type == 'normalized':
        d12 = np.power(d, -0.5)
        D12 = sparse.diags(np.ravel(d12), 0, dtype=dtype).tocsc()
        return sparse.identity(d.shape[0], dtype=dtype) - D12 * W * D12
    else:
        raise ValueError('Unknown Laplacian type {}'.format(lap_type))


def healpix_graph(nside=16,
                  nest=True,
                  lap_type='normalized',
                  indexes=None,
                  use_4=False,
                  dtype=np.float32, 
                  new=True):
    """Build a healpix graph using the pygsp from NSIDE."""
    
    if new:
        G = SphereHealpix(nside=nside, indexes=indexes, nest=nest, lap_type=lap_type)
    else:
        from pygsp import graphs

        if indexes is None:
            indexes = range(4*bw**2)

        # 1) get the coordinates
        npix = hp.nside2npix(nside)  # number of pixels: 12 * nside**2
        pix = range(npix)
        x, y, z = hp.pix2vec(nside, pix, nest=nest)
        coords = np.vstack([x, y, z]).transpose()[indexes]
        # 2) computing the weight matrix
        if use_4:
            raise NotImplementedError()
            W = build_matrix_4_neighboors(nside, indexes, nest=nest, dtype=dtype)
        else:
            W = healpix_weightmatrix(
                nside=nside, nest=nest, indexes=indexes, dtype=dtype)
        # 3) building the graph
        G = graphs.Graph(
            W,
            lap_type=lap_type,
            coords=coords)
    return G

def equiangular_graph(bw=64,
                  lap_type='normalized',
                  indexes=None,
                  use_4=False,
                  dtype=np.float32):
    """Build a equiangular graph using the pygsp from given bandwidth."""
    G = SphereEquiangular(bandwidth=bw, sampling='SOFT')

#     if indexes is None:
#         indexes = range(4*bw**2)

#     # 1) get the coordinates    
#     beta = np.pi * (2 * np.arange(2 * bw) + 1) / (4. * bw) #SOFT
#     alpha = np.arange(2 * bw) * np.pi / bw
#     theta, phi = np.meshgrid(*(beta, alpha),indexing='ij')
#     ct = np.cos(theta)
#     st = np.sin(theta)
#     cp = np.cos(phi)
#     sp = np.sin(phi)
#     x = st * cp
#     y = st * sp
#     z = ct
#     coords = np.vstack([x.flatten(), y.flatten(), z.flatten()]).transpose() 
#     coords = np.asarray(coords, dtype=dtype)[indexes]
#     # 2) computing the weight matrix
#     if use_4:
#         raise NotImplementedError()
#         W = build_matrix_4_neighboors(nside, indexes, nest=nest, dtype=dtype)
#     else:
#         W = equiangular_weightmatrix(
#             bw=bw, indexes=indexes, dtype=dtype)
#     # 3) building the graph
#     G = graphs.Graph(
#         W,
#         lap_type=lap_type,
#         coords=coords)
    return G


def healpix_laplacian(nside=16,
                      nest=True,
                      lap_type='normalized',
                      indexes=None,
                      dtype=np.float32,
                      use_4=False, 
                      std=None,
                      full=False,
                      new=True,
                      n_neighbors=8):
    """Build a Healpix Laplacian."""
    if new:
        G = SphereHealpix(nside=nside, indexes=indexes, nest=nest, n_neighbors=n_neighbors)
        G.compute_laplacian(lap_type)
        L = sparse.csr_matrix(G.L, dtype=dtype)
    else:
        if use_4:
            W = build_matrix_4_neighboors(nside, indexes, nest=nest, dtype=dtype)
        else:
            W = healpix_weightmatrix(
                nside=nside, nest=nest, indexes=indexes, dtype=dtype, std=std, full=full)
        L = build_laplacian(W, lap_type=lap_type)
    return L

def equiangular_laplacian(bw=16,
                          lap_type='normalized',
                          indexes=None,
                          dtype=np.float32,
                          use_4=False):
    """Build a Equiangular Laplacian."""
    G = SphereEquiangular(bandwidth=bw, sampling='SOFT')
    G.compute_laplacian(lap_type)
    L = sparse.csr_matrix(G.L, dtype=dtype)
#     if use_4:
#         raise NotImplementedError()
#     else:
#         W = equiangular_weightmatrix(
#             bw=bw, indexes=indexes, dtype=dtype)
#     L = build_laplacian(W, lap_type=lap_type)# see if change
    return L


def icosahedron_graph(order=64,
                  lap_type='normalized',
                  indexes=None,
                  use_4=False,
                  dtype=np.float32):
    graph = SphereIcosahedron(level=order)
    return graph
    
def icosahedron_laplacian(order=0,
                          lap_type='combinatorial',
                          indexes=None,
                          dtype=np.float32):
    graph = SphereIcosahedron(level=order)
    graph.compute_laplacian(lap_type)
    return sparse.csr_matrix(graph.L, dtype=dtype)


def rescale_L(L, lmax=2, scale=1):
    """Rescale the Laplacian eigenvalues in [-scale,scale]."""
    M, M = L.shape
    I = sparse.identity(M, format='csr', dtype=L.dtype)
    L /= (lmax / 2)
    L -= I
    return L*scale


def build_laplacians(nsides, indexes=None, use_4=False, sampling='healpix', std=None, full=False, new=True, n_neighbors=8):
    """Build a list of Laplacians (and down-sampling factors) from a list of nsides."""
    L = []
    p = []
    if indexes is None:
        indexes = [None] * len(nsides)
    if not isinstance(std, list):
        std = [std] * len(nsides)
    if not isinstance(full, list):
        full = [full] * len(nsides)
    import time
    lstart = time.time()
    nside_last = -1
    for i, (nside, index, sigma, mat) in enumerate(zip(nsides, indexes, std, full)):
        bw = nside
        if isinstance(nside, tuple):
            nside = nside[0]
        if i > 0 and sampling != 'icosahedron':  # First is input dimension.
            p.append((nside_last // nside)**2)
        if nside == nside_last and i < len(nsides) - 1:
            L.append(L[-1].copy().tocoo())
            continue
        nside_last = nside
        if i < len(nsides) - 1:  # Last does not need a Laplacian.
            if sampling == 'healpix':
                laplacian = healpix_laplacian(nside=nside, indexes=index, use_4=use_4, 
                                              std=sigma, full=mat, new=new, n_neighbors=n_neighbors)
            elif sampling == 'equiangular':
                laplacian = equiangular_laplacian(bw=bw, indexes=index, use_4=use_4)
            elif sampling == 'icosahedron':
                laplacian = icosahedron_laplacian(order=nside, indexes=index)
            else:
                raise ValueError('Unknown sampling: '+sampling)
            print("build laplacian, time: ", time.time()-lstart)
#             L = sparse.csr_matrix(L)
            lmax = 1.02*sparse.linalg.eigsh(laplacian, k=1, which='LM', return_eigenvectors=False)[0]
            laplacian = rescale_L(laplacian, lmax=lmax)
            laplacian = laplacian.tocoo()
            print("rescale laplacian, time: ", time.time()-lstart)
            L.append(laplacian)
    if sampling == 'icosahedron':
        for order in nsides[1:]:
            p.append(10 * 4 ** order + 2)
    return L, p


def nside2indexes(nsides, order):
    """Return list of indexes from nside given a specific order.

    This function return the necessary indexes for a deepsphere when
    only a part of the sphere is considered.

    Parameters
    ----------
    nsides : list of nside for the desired scale
    order  : parameter specifying the size of the sphere part
    """
    nsample = 12 * order**2
    if order==0:
        nsample=1
    indexes = [np.arange(hp.nside2npix(nside) // nsample) for nside in nsides]
    return indexes


def ds_index(index, nsides, nest=True):
    """Return list of indexes sampled at specific nsides.
    
    The given index must be sampled at the first nside given
    Parameters
    ----------
    index : list of pixel position for part of sphere
    nsides : list of nside for the desired scale
    """
    assert isinstance(nsides, list)
    assert len(nsides) > 1
    assert nest  # not implemented yet
    
    indexes = [index]
    for nside in nsides[1:]:
        p = (nsides[0]/nside)**2
        if p < 1:
            raise NotImplementedError("upsampling not implemented yet")
        temp_index = index//p
        indexes.append(np.unique(temp_index).astype(int))            
    
    return indexes


def show_all_variables():
    """Show all variable of the curent tensorflow graph."""
    import tensorflow as tf
    import tensorflow.contrib.slim as slim
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def build_matrix_4_neighboors(nside, indexes, nest=True, dtype=np.float32):
    assert(nest)

    order = nside//hp.npix2nside(12*(max(indexes)+1))

    npix = hp.nside2npix(nside) // hp.nside2npix(order)
    new_indexes = list(range(npix))
    assert(set(indexes)==set(new_indexes))

    x, y, z = hp.pix2vec(nside, indexes, nest=True)
    coords = np.vstack([x, y, z]).transpose()
    coords = np.array(coords)

    def all_or(d3, v):
        v = np.array(v)
        for d in d3:
            if not (v == d).any():
                return False
        return True

    row_index = []
    col_index = []
    for index in indexes:
        # A) Start with the initial square
        d = index % 4
        base = index - d
        # 1) Add the next pixel
        row_index.append(index)
        if d == 0:
            col_index.append(base + 1)
        elif d == 1:
            col_index.append(base + 3)
        elif d == 2:
            col_index.append(base)
        elif d == 3:
            col_index.append(base + 2)
        else:
            raise ValueError('Error in the code')
        # 2) Add the previous pixel
        row_index.append(index)
        if d == 0:
            col_index.append(base + 2)
        elif d == 1:
            col_index.append(base)
        elif d == 2:
            col_index.append(base + 3)
        elif d == 3:
            col_index.append(base + 1)
        else:
            raise ValueError('Error in the code')

        # B) Connect the squares together...
        for it in range(int(np.log2(nside) - np.log2(order) - 1)):

            d2 = (index // (4**(it + 1))) % 4
            d3 = [d]
            for it2 in range(it):
                d3.append((index // (4**(it2 + 1)) % 4))
            d3 = np.array(d3)
            shift_o = []
            for it2 in range(it + 1):
                shift_o.append(4**it2)
            shift = 4**(it + 1) - sum(shift_o)
            if d2 == 0:
                if all_or(d3, [1, 3]):
                    row_index.append(index)
                    col_index.append(index + shift)
                if all_or(d3, [2, 3]):
                    row_index.append(index)
                    col_index.append(index + 2 * shift)
            elif d2 == 1:
                if all_or(d3, [0, 2]):
                    row_index.append(index)
                    col_index.append(index - shift)
                if all_or(d3, [2, 3]):
                    row_index.append(index)
                    col_index.append(index + 2 * shift)
            elif d2 == 2:
                if all_or(d3, [0, 1]):
                    row_index.append(index)
                    col_index.append(index - 2 * shift)
                if all_or(d3, [1, 3]):
                    row_index.append(index)
                    col_index.append(index + shift)
            elif d2 == 3:
                if all_or(d3, [0, 1]):
                    row_index.append(index)
                    col_index.append(index - 2 * shift)
                if all_or(d3, [0, 2]):
                    row_index.append(index)
                    col_index.append(index - shift)
            else:
                raise ValueError('Error in the code')

    # Compute Euclidean distances between neighbors.
    distances = np.sum((coords[row_index] - coords[col_index])**2, axis=1)
    # slower: np.linalg.norm(coords[row_index] - coords[col_index], axis=1)**2

    # Compute similarities / edge weights.
    kernel_width = np.mean(distances)
    weights = np.exp(-distances / (3 * kernel_width))

    # Build the sparse matrix.
    W = sparse.csr_matrix(
        (weights, (row_index, col_index)), shape=(npix, npix), dtype=dtype)

    return W


def require_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return None


def download(url, target_dir, filename=None):
    require_dir(target_dir)
    if filename is None:
        filename = url_filename(url)
    filepath = os.path.join(target_dir, filename)
    urlretrieve(url, filepath)
    return filepath


def url_filename(url):
    return url.split('/')[-1].split('#')[0].split('?')[0]

def check_md5(file_name, orginal_md5):
    # Open,close, read file and calculate MD5 on its contents
    with open(file_name, 'rb') as f:
        hasher = hashlib.md5()  # Make empty hasher to update piecemeal
        while True:
            block = f.read(64 * (1 << 20)) # Read 64 MB at a time; big, but not memory busting
            if not block:  # Reached EOF
                break
            hasher.update(block)  # Update with new block
    md5_returned = hasher.hexdigest()
    # Finally compare original MD5 with freshly calculated
    if orginal_md5 == md5_returned:
        print('MD5 verified.')
        return True
    else:
        print('MD5 verification failed!')
        return False

def unzip(file, targetdir):
    with zipfile.ZipFile(file,"r") as zip_ref:
        zip_ref.extractall(targetdir)

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def compute_spherical_harmonics(nside, lmax):
    """Compute the spherical harmonics up to lmax.

    Returns
    -------
    harmonics: array of shape n_pixels x n_harmonics
        Harmonics are in nested order.
    """

    n_harmonics = np.sum(np.arange(1, 2*lmax+2, 2))
    harmonics = np.empty((hp.nside2npix(nside), n_harmonics))
    midx = 0

    for l in range(lmax+1):
        for m in range(-l, l+1):
            size = hp.sphtfunc.Alm.getsize(l, mmax=l)
            alm = np.zeros(size, dtype=np.complex128)
            idx = hp.sphtfunc.Alm.getidx(l, l, abs(m))
            alm[idx] = 1 if m == 0 else (1 - 1j)/np.sqrt(2) if m < 0 else (1 + 1j)/np.sqrt(2)
            harmonic = hp.sphtfunc.alm2map(alm, nside, l, verbose=False)
            harmonic /= np.sqrt(np.sum(harmonic**2))
            harmonics[:, midx] = hp.reorder(harmonic, r2n=True)
            midx += 1

    return harmonics


def test_learning_rates(params, ntrain, lr_min=1e-6, lr_max=1e-1, num_epochs=20, exponential=True):
    """Test learning rates from lr_min to lr_max.

    The test is performed by linearly or exponentially increasing the
    learning rate from lr_min to lr_max during num_epochs epochs.
    The optimal learning rate can be determined by looking at the
    validation loss and choosing the largest value for which the loss
    still decreases.
    """
    import tensorflow as tf

    params['dir_name'] = 'lr_finder'
    params['num_epochs'] = num_epochs

    n_steps = num_epochs * ntrain // params['batch_size']

    if exponential:
        decay = np.power(lr_max/lr_min, 1/n_steps, dtype=np.float32)
        def scheduler(step):
            return lr_min * decay ** step
        params['scheduler'] = lambda step: scheduler(tf.to_float(step))
    else:
        def scheduler(step):
            return lr_min + step/n_steps * (lr_max-lr_min)
        params['scheduler'] = lambda step: scheduler(step)

    steps = np.arange(params['eval_frequency'], n_steps, params['eval_frequency'])
    steps = np.append(steps, n_steps) - 1
    learning_rate = scheduler(steps)

    return params, learning_rate
