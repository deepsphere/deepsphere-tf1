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


if sys.version_info[0] > 2:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


def unique_rows(data, digits=None):
    """
    Returns indices of unique rows. It will return the
    first occurrence of a row that is duplicated:
    [[1,2], [3,4], [1,2]] will return [0,1]
    Parameters
    ---------
    data: (n,m) set of floating point data
    digits: how many digits to consider for the purposes of uniqueness
    Returns
    --------
    unique:  (j) array, index in data which is a unique row
    inverse: (n) length array to reconstruct original
                 example: unique[inverse] == data
    """
    hashes = hashable_rows(data, digits=digits)
    garbage, unique, inverse = np.unique(hashes,
                                         return_index=True,
                                         return_inverse=True)
    return unique, inverse


def hashable_rows(data, digits=None):
    """
    We turn our array into integers based on the precision
    given by digits and then put them in a hashable format.
    Parameters
    ---------
    data:    (n,m) input array
    digits:  how many digits to add to hash, if data is floating point
             If none, TOL_MERGE will be turned into a digit count and used.
    Returns
    ---------
    hashable:  (n) length array of custom data which can be sorted
                or used as hash keys
    """
    # if there is no data return immediatly
    if len(data) == 0:
        return np.array([])

    # get array as integer to precision we care about
    as_int = float_to_int(data, digits=digits)

    # if it is flat integers already, return
    if len(as_int.shape) == 1:
        return as_int

    # if array is 2D and smallish, we can try bitbanging
    # this is signifigantly faster than the custom dtype
    if len(as_int.shape) == 2 and as_int.shape[1] <= 4:
        # time for some righteous bitbanging
        # can we pack the whole row into a single 64 bit integer
        precision = int(np.floor(64 / as_int.shape[1]))
        # if the max value is less than precision we can do this
        if np.abs(as_int).max() < 2**(precision - 1):
            # the resulting package
            hashable = np.zeros(len(as_int), dtype=np.int64)
            # loop through each column and bitwise xor to combine
            # make sure as_int is int64 otherwise bit offset won't work
            for offset, column in enumerate(as_int.astype(np.int64).T):
                # will modify hashable in place
                np.bitwise_xor(hashable,
                               column << (offset * precision),
                               out=hashable)
            return hashable

    # reshape array into magical data type that is weird but hashable
    dtype = np.dtype((np.void, as_int.dtype.itemsize * as_int.shape[1]))
    # make sure result is contiguous and flat
    hashable = np.ascontiguousarray(as_int).view(dtype).reshape(-1)
    return hashable


def float_to_int(data, digits=None, dtype=np.int32):
    """
    Given a numpy array of float/bool/int, return as integers.
    Parameters
    -------------
    data:   (n, d) float, int, or bool data
    digits: float/int precision for float conversion
    dtype:  numpy dtype for result
    Returns
    -------------
    as_int: data, as integers
    """
    # convert to any numpy array
    data = np.asanyarray(data)

    # if data is already an integer or boolean we're done
    # if the data is empty we are also done
    if data.dtype.kind in 'ib' or data.size == 0:
        return data.astype(dtype)

    # populate digits from kwargs
    if digits is None:
        digits = decimal_to_digits(1e-8)
    elif isinstance(digits, float) or isinstance(digits, np.float):
        digits = decimal_to_digits(digits)
    elif not (isinstance(digits, int) or isinstance(digits, np.integer)):
        log.warn('Digits were passed as %s!', digits.__class__.__name__)
        raise ValueError('Digits must be None, int, or float!')

    # data is float so convert to large integers
    data_max = np.abs(data).max() * 10**digits
    # ignore passed dtype if we have something large
    dtype = [np.int32, np.int64][int(data_max > 2**31)]
    # multiply by requested power of ten
    # then subtract small epsilon to avoid "go either way" rounding
    # then do the rounding and convert to integer
    as_int = np.round((data * 10 ** digits) - 1e-6).astype(dtype)

    return as_int


def decimal_to_digits(decimal, min_digits=None):
    """
    Return the number of digits to the first nonzero decimal.
    Parameters
    -----------
    decimal:    float
    min_digits: int, minumum number of digits to return
    Returns
    -----------
    digits: int, number of digits to the first nonzero decimal
    """
    digits = abs(int(np.log10(decimal)))
    if min_digits is not None:
        digits = np.clip(digits, min_digits, 20)
    return digits
    
    
    
    
    
    
    
    
    
    
    
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
                  dtype=np.float32):
    """Build a healpix graph using the pygsp from NSIDE."""
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
    from pygsp import graphs

    if indexes is None:
        indexes = range(4*bw**2)

    # 1) get the coordinates    
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
    coords = np.asarray(coords, dtype=dtype)[indexes]
    # 2) computing the weight matrix
    if use_4:
        raise NotImplementedError()
        W = build_matrix_4_neighboors(nside, indexes, nest=nest, dtype=dtype)
    else:
        W = equiangular_weightmatrix(
            bw=bw, indexes=indexes, dtype=dtype)
    # 3) building the graph
    G = graphs.Graph(
        W,
        lap_type=lap_type,
        coords=coords)
    return G


def healpix_laplacian(nside=16,
                      nest=True,
                      lap_type='normalized',
                      indexes=None,
                      dtype=np.float32,
                      use_4=False, 
                      std=None,
                      full=False):
    """Build a Healpix Laplacian."""
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
    if use_4:
        raise NotImplementedError()
    else:
        W = equiangular_weightmatrix(
            bw=bw, indexes=indexes, dtype=dtype)
    L = build_laplacian(W, lap_type=lap_type)# see if change
    return L


from pygsp.graphs import NNGraph
class SphereIcosahedron(NNGraph):
    def __init__(self, level, sampling='vertex', **kwargs):
        from collections import deque
        ## sampling in ['vertex', 'face']
        self.intp = None
        PHI = (1 + np.sqrt(5))/2
        radius = np.sqrt(PHI**2+1)
        coords = [-1, PHI, 0, 1, PHI, 0, -1, -PHI, 0, 1, -PHI, 0, 
                  0, -1, PHI, 0, 1, PHI, 0, -1, -PHI, 0, 1, -PHI,
                  PHI, 0, -1, PHI, 0, 1, -PHI, 0, -1, -PHI, 0, 1]
        coords =  np.reshape(coords, (-1, 3))/radius
        faces = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11,
                 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
                 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9,
                 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1]
        self.faces = np.reshape(faces, (20,3))
        self.level = level
        self.coords = coords
        
        self.coords = self._upward(coords, self.faces)
        ## rotate icosahedron?
        for i in range(level):
            self.divide()
            self.normalize()
            # self.coords = self.coords.reshape((4,-1)).T.flatten()
        
        if sampling=='face':
            self.coords = self.coords[self.faces].mean(axis=1)
            
        self.lat, self.long = self.xyz2latlong()
#         theta = [0] + 5*[np.pi/2-np.arctan(0.5)] + 5*[np.pi/2+np.arctan(0.5)] + [np.pi]
#         phi = [0] + np.linspace(0, 2*np.pi, 5, endpoint=False).tolist() +  (np.linspace(0, 2*np.pi, 5, endpoint=False)+(np.pi/5)).tolist() + [0]
        
        self.npix = len(self.coords)
        self.nf = 20 * 4**self.level
        self.ne = 30 * 4**self.level
        self.nv = self.ne - self.nf + 2
        self.nv_prev = int((self.ne / 4) - (self.nf / 4) + 2)
        self.nv_next = int((self.ne * 4) - (self.nf * 4) + 2)
        #W = np.ones((self.npix, self.npix))
        
        neighbours = 3 if 'face' in sampling else (5 if level == 0 else 6)
        super(SphereIcosahedron, self).__init__(self.coords, k=neighbours, **kwargs)
        
    def divide(self):
        """
        Subdivide a mesh into smaller triangles.
        """
        faces = self.faces
        vertices = self.coords
        face_index = np.arange(len(faces))

        # the (c,3) int set of vertex indices
        faces = faces[face_index]
        # the (c, 3, 3) float set of points in the triangles
        triangles = vertices[faces]
        # the 3 midpoints of each triangle edge vstacked to a (3*c, 3) float
        src_idx = np.vstack([faces[:, g] for g in [[0, 1], [1, 2], [2, 0]]])
        mid = np.vstack([triangles[:, g, :].mean(axis=1) for g in [[0, 1],
                                                                   [1, 2],
                                                                   [2, 0]]])
        mid_idx = (np.arange(len(face_index) * 3)).reshape((3, -1)).T
        # for adjacent faces we are going to be generating the same midpoint
        # twice, so we handle it here by finding the unique vertices
        unique, inverse = unique_rows(mid)

        mid = mid[unique]
        src_idx = src_idx[unique]
        mid_idx = inverse[mid_idx] + len(vertices)
        # the new faces, with correct winding
        f = np.column_stack([faces[:, 0], mid_idx[:, 0], mid_idx[:, 2],
                             mid_idx[:, 0], faces[:, 1], mid_idx[:, 1],
                             mid_idx[:, 2], mid_idx[:, 1], faces[:, 2],
                             mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2], ]).reshape((-1, 3))
        # add the 3 new faces per old face
        new_faces = np.vstack((faces, f[len(face_index):]))
        # replace the old face with a smaller face
        new_faces[face_index] = f[:len(face_index)]

        new_vertices = np.vstack((vertices, mid))
        # source ids
        nv = vertices.shape[0]
        identity_map = np.stack((np.arange(nv), np.arange(nv)), axis=1)
        src_id = np.concatenate((identity_map, src_idx), axis=0)

        self.coords = new_vertices
        self.faces = new_faces
        self.intp = src_id
        
    def normalize(self, radius=1):
        '''
        Reproject to spherical surface
        '''
        vectors = self.coords
        scalar = (vectors ** 2).sum(axis=1)**.5
        unit = vectors / scalar.reshape((-1, 1))
        offset = radius - scalar
        self.coords += unit * offset.reshape((-1, 1))
        
    def xyz2latlong(self):
        x, y, z = self.coords[:, 0], self.coords[:, 1], self.coords[:, 2]
        long = np.arctan2(y, x) + np.pi
        xy2 = x**2 + y**2
        lat = np.arctan2(z, np.sqrt(xy2))
        return lat, long
    
    def _upward(self, V_ico, F_ico, ind=11):
        V0 = V_ico[ind]
        Z0 = np.array([0, 0, 1])
        k = np.cross(V0, Z0)
        ct = np.dot(V0, Z0)
        st = -np.linalg.norm(k)
        R = self._rot_matrix(k, ct, st)
        V_ico = V_ico.dot(R)
        # rotate a neighbor to align with (+y)
        ni = self._find_neighbor(F_ico, ind)[0]
        vec = V_ico[ni].copy()
        vec[2] = 0
        vec = vec/np.linalg.norm(vec)
        y_ = np.eye(3)[1]

        k = np.eye(3)[2]
        crs = np.cross(vec, y_)
        ct = -np.dot(vec, y_)
        st = -np.sign(crs[-1])*np.linalg.norm(crs)
        R2 = self._rot_matrix(k, ct, st)
        V_ico = V_ico.dot(R2)
        return V_ico
    
    def _find_neighbor(self, F, ind):
        """find a icosahedron neighbor of vertex i"""
        FF = [F[i] for i in range(F.shape[0]) if ind in F[i]]
        FF = np.concatenate(FF)
        FF = np.unique(FF)
        neigh = [f for f in FF if f != ind]
        return neigh

    def _rot_matrix(self, rot_axis, cos_t, sin_t):
        k = rot_axis / np.linalg.norm(rot_axis)
        I = np.eye(3)

        R = []
        for i in range(3):
            v = I[i]
            vr = v*cos_t+np.cross(k, v)*sin_t+k*(k.dot(v))*(1-cos_t)
            R.append(vr)
        R = np.stack(R, axis=-1)
        return R

    def _ico_rot_matrix(self, ind):
        """
        return rotation matrix to perform permutation corresponding to 
        moving a certain icosahedron node to the top
        """
        v0_ = self.v0.copy()
        f0_ = self.f0.copy()
        V0 = v0_[ind]
        Z0 = np.array([0, 0, 1])

        # rotate the point to the top (+z)
        k = np.cross(V0, Z0)
        ct = np.dot(V0, Z0)
        st = -np.linalg.norm(k)
        R = self._rot_matrix(k, ct, st)
        v0_ = v0_.dot(R)

        # rotate a neighbor to align with (+y)
        ni = self._find_neighbor(f0_, ind)[0]
        vec = v0_[ni].copy()
        vec[2] = 0
        vec = vec/np.linalg.norm(vec)
        y_ = np.eye(3)[1]

        k = np.eye(3)[2]
        crs = np.cross(vec, y_)
        ct = np.dot(vec, y_)
        st = -np.sign(crs[-1])*np.linalg.norm(crs)

        R2 = self._rot_matrix(k, ct, st)
        return R.dot(R2)

    def _rotseq(self, V, acc=9):
        """sequence to move an original node on icosahedron to top"""
        seq = []
        for i in range(11):
            Vr = V.dot(self._ico_rot_matrix(i))
            # lexsort
            s1 = np.lexsort(np.round(V.T, acc))
            s2 = np.lexsort(np.round(Vr.T, acc))
            s = s1[np.argsort(s2)]
            seq.append(s)
        return tuple(seq)


def icosahedron_graph(order=64,
                  lap_type='normalized',
                  indexes=None,
                  use_4=False,
                  dtype=np.float32):
    graph = SphereIcosahedron(order, sampling='vertex')
    return graph
    
def icosahedron_laplacian(order=0,
                          lap_type='normalized',
                          indexes=None,
                          dtype=np.float32):
    graph = SphereIcosahedron(order, sampling='vertex')
    graph.compute_laplacian("combinatorial")
    return sparse.csr_matrix(graph.L, dtype=dtype)


def rescale_L(L, lmax=2, scale=1):
    """Rescale the Laplacian eigenvalues in [-scale,scale]."""
    M, M = L.shape
    I = sparse.identity(M, format='csr', dtype=L.dtype)
    L /= (lmax / 2)
    L -= I
    return L*scale


def build_laplacians(nsides, indexes=None, use_4=False, sampling='healpix', std=None, full=False):
    """Build a list of Laplacians (and down-sampling factors) from a list of nsides."""
    L = []
    p = []
    if indexes is None:
        indexes = [None] * len(nsides)
    if not isinstance(std, list):
        std = [std] * len(nsides)
    if not isinstance(full, list):
        full = [full] * len(nsides)
    for i, (nside, index, sigma, mat) in enumerate(zip(nsides, indexes, std, full)):
        if i > 0 and sampling != 'icosahedron':  # First is input dimension.
            p.append((nside_last // nside)**2)
        nside_last = nside
        if i < len(nsides) - 1:  # Last does not need a Laplacian.
            if sampling == 'healpix':
                laplacian = healpix_laplacian(nside=nside, indexes=index, use_4=use_4, std=sigma, full=mat)
            elif sampling == 'equiangular':
                laplacian = equiangular_laplacian(bw=nside, indexes=index, use_4=use_4)
            elif sampling == 'icosahedron':
                laplacian = icosahedron_laplacian(order=nside, indexes=index)
            else:
                raise ValueError('Unknown sampling: '+sampling)
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
