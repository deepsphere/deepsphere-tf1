"""Parameters used for the experiments of the paper."""

import tensorflow as tf
import numpy as np

from deepsphere import utils


def get_params(ntrain, EXP_NAME, order, Nside, architecture="FCN", verbose=True):
    """Parameters for the cgcnn and cnn2d defined in deepsphere/models.py"""

    n_classes = 2

    params = dict()
    params['dir_name'] = EXP_NAME

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [16, 32, 64, 64, 64, n_classes]  # Graph convolutional layers: number of feature maps.
    params['K'] = [5] * 6  # Polynomial orders.
    params['batch_norm'] = [True] * 6  # Batch normalization.
    params['M'] = []  # Fully connected layers: output dimensionalities.

    # Pooling.
    nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//32, Nside//32]
    params['nsides'] = nsides
    params['indexes'] = utils.nside2indexes(nsides, order)
#     params['batch_norm_full'] = []

    if architecture == "CNN":
        # Classical convolutional neural network.
        # Replace the last graph convolution and global average pooling by a fully connected layer.
        # That is, change the classifier while keeping the feature extractor.
        params['F'] = params['F'][:-1]
        params['K'] = params['K'][:-1]
        params['batch_norm'] = params['batch_norm'][:-1]
        params['nsides'] = params['nsides'][:-1]
        params['indexes'] = params['indexes'][:-1]
        params['statistics'] = None
        params['M'] = [n_classes]
    elif architecture == "FCN":
        pass
        
    else:
        raise ValueError('Unknown architecture {}.'.format(architecture))

    # Regularization (to prevent over-fitting).
    params['regularization'] = 0  # Amount of L2 regularization over the weights (will be divided by the number of weights).
    params['dropout'] = 1  # Percentage of neurons to keep.

    # Training.
    params['num_epochs'] = 80  # Number of passes through the training data.
    params['batch_size'] = max(8 * order, 1)    # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: tf.train.exponential_decay(2e-4, step, decay_steps=1, decay_rate=0.999)
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)

    # Number of model evaluations during training (influence training time).
    n_evaluations = 80
    params['eval_frequency'] = int(params['num_epochs'] * ntrain / params['batch_size'] / n_evaluations)

    if verbose:
        print('#sides: {}'.format(nsides))
#         print('#pixels: {}'.format([(nside//order)**2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
#         print('=> #pixels per batch (input): {:,}'.format(params['batch_size']*(Nside//order)**2))
#         print('=> #pixels for training (input): {:,}'.format(params['num_epochs']*ntrain*(Nside//order)**2))

        n_steps = params['num_epochs'] * ntrain // params['batch_size']
        lr = [params['scheduler'](step).eval(session=tf.Session()) for step in [0, n_steps]]
        print('Learning rate will start at {:.1e} and finish at {:.1e}.'.format(*lr))

    return params



def get_params_shrec17(ntrain, EXP_NAME, Nside, n_classes, nfeat_in=6, architecture="FCN", verbose=True):

    """
    :param ntrain: int, number of elements in the training set
    :param EXP_NAME: string, name of experiment
    :param n_classes: int, number of classes present in SHREC17 dataset
    :param Nside: int, parameter of HEALpix
    :param architecture: string, type of NN
    :param verbose: bool, print info
    :return: parameters needed to create a deepsphere model
    """

    params = dict()
    params['dir_name'] = EXP_NAME
    params['num_feat_in'] = nfeat_in

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [100, 100, n_classes]
    params['K'] = [5] * 3  # Polynomial orders.
#     params['K'] = [np.ceil(np.sqrt(3)*Nside).astype(int), 
#                    np.ceil(np.sqrt(3)*Nside//4).astype(int), 
#                    np.ceil(np.sqrt(3)*Nside//8).astype(int)]
    params['batch_norm'] = [True] * 3  # Batch normalization.
    params['M'] = []  # Fully connected layers: output dimensionalities.

    # Pooling.
    nsides = [Nside, Nside//4, Nside//8, Nside//8]
    params['nsides'] = nsides
    params['indexes'] = None

    if architecture == "CNN":
        # Replace the last graph convolution and global average pooling by a fully connected layer.
        # That is, change the classifier while keeping the feature extractor.
        params['F'] = params['F'][:-1]
        params['K'] = params['K'][:-1]
        params['nsides'] = params['nsides'][:-1]
        params['batch_norm'] = params['batch_norm'][:-1]
        params['statistics'] = 'mean'
        params['M'] = [n_classes]
    elif architecture != "FCN":
        raise ValueError('Unknown architecture {}.'.format(architecture))

    # Regularization (to prevent over-fitting).
    params['regularization'] = 0  # Amount of L2 regularization over the weights (will be divided by the number of weights).
    params['dropout'] = 1  # Percentage of neurons to keep.

    # Training.
    params['num_epochs'] = 100  # Number of passes through the training data.
    params['batch_size'] = 32  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: tf.train.exponential_decay(5e-1, step, decay_steps=5, decay_rate=1)#0.999)
    #params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params['optimizer'] = lambda lr: tf.train.GradientDescentOptimizer(lr)

    # Number of model evaluations during training (influence training time).
    n_evaluations = 200
    params['eval_frequency'] = int(params['num_epochs'] * ntrain / params['batch_size'] / n_evaluations)

    if verbose:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([12 * (nside)**2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size']*12*(Nside)**2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs']*ntrain*12*(Nside)**2))

        n_steps = params['num_epochs'] * ntrain // params['batch_size']
        lr = [params['scheduler'](step).eval(session=tf.Session(config=config)) for step in [0, n_steps]]
        print('Learning rate will start at {:.1e} and finish at {:.1e}.'.format(*lr))

    return params

def get_params_shrec17_optim(ntrain, EXP_NAME, Nside, n_classes, nfeat_in=6, architecture="FCN", verbose=True):

    """
    :param ntrain: int, number of elements in the training set
    :param EXP_NAME: string, name of experiment
    :param n_classes: int, number of classes present in SHREC17 dataset
    :param Nside: int, parameter of HEALpix
    :param architecture: string, type of NN
    :param verbose: bool, print info
    :return: parameters needed to create a deepsphere model
    """
    
    params = dict()
    params['dir_name'] = EXP_NAME
    params['num_feat_in'] = nfeat_in

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [16, 32, 64, 128, 256, n_classes]  # Graph convolutional layers: number of feature maps.
    params['K'] = [4] * 6  # Polynomial orders.
#     params['K'] = [np.ceil(np.sqrt(3)*Nside).astype(int), 
#                    np.ceil(np.sqrt(3)*Nside//4).astype(int), 
#                    np.ceil(np.sqrt(3)*Nside//8).astype(int)]
    params['batch_norm'] = [True] * 6  # Batch normalization.
    params['M'] = []  # Fully connected layers: output dimensionalities.

    # Pooling.
    nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//32, Nside//32]
    params['nsides'] = nsides
    params['indexes'] = None

    if architecture == "CNN":
        # Replace the last graph convolution and global average pooling by a fully connected layer.
        # That is, change the classifier while keeping the feature extractor.
        params['F'] = params['F'][:-1]
        params['K'] = params['K'][:-1]
        params['nsides'] = params['nsides'][:-1]
        params['batch_norm'] = params['batch_norm'][:-1]
        params['statistics'] = 'mean'
        params['M'] = [n_classes]
    elif architecture != "FCN":
        raise ValueError('Unknown architecture {}.'.format(architecture))

    # Regularization (to prevent over-fitting).
    params['regularization'] = 0  # Amount of L2 regularization over the weights (will be divided by the number of weights).
    params['dropout'] = 1  # Percentage of neurons to keep.
    params['dropFilt'] = 1 # percentage of filter to keep in each layer

    # Training.
    params['num_epochs'] = 30 #30  # Number of passes through the training data.
    params['batch_size'] = 32  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: tf.train.exponential_decay(5e-2, step, decay_steps=5, decay_rate=1)#decay_steps=7000, decay_rate=0.1, staircase=True)#0.999)
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=0.1)
    #params['optimizer'] = lambda lr: tf.train.GradientDescentOptimizer(lr)
    #params['optimizer'] = lambda lr: tf.train.RMSPropOptimizer(lr, decay=0.9, momentum=0.)

    # Number of model evaluations during training (influence training time).
    n_evaluations = 60
    params['eval_frequency'] = int(params['num_epochs'] * ntrain / params['batch_size'] / n_evaluations)

    if verbose:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([12 * (nside)**2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size']*12*(Nside)**2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs']*ntrain*12*(Nside)**2))

        n_steps = params['num_epochs'] * ntrain // params['batch_size']
        lr = [params['scheduler'](step).eval(session=tf.Session(config=config)) for step in [0, n_steps]]
        print('Learning rate will start at {:.1e} and finish at {:.1e}.'.format(*lr))

    return params

def get_params_shrec17_equiangular(ntrain, EXP_NAME, n_classes, nfeat_in=6, architecture="FCN", verbose=True):

    """
    :param ntrain: int, number of elements in the training set
    :param EXP_NAME: string, name of experiment
    :param n_classes: int, number of classes present in SHREC17 dataset
    :param Nside: int, parameter of HEALpix
    :param architecture: string, type of NN
    :param verbose: bool, print info
    :return: parameters needed to create a deepsphere model
    """

    params = dict()
    params['dir_name'] = EXP_NAME
    params['num_feat_in'] = nfeat_in
    params['sampling'] = 'equiangular'

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [16, 32, 64, 128, 256, n_classes]  # Graph convolutional layers: number of feature maps.
    params['K'] = [4] * 6  # Polynomial orders.
    params['batch_norm'] = [True] * 6  # Batch normalization.
    params['M'] = []  # Fully connected layers: output dimensionalities.

    # Pooling.
    bandwidth = [64, 32, 16, 8, 4, 2, 2]
    params['nsides'] = bandwidth
    params['indexes'] = None

    if architecture == "CNN":
        # Replace the last graph convolution and global average pooling by a fully connected layer.
        # That is, change the classifier while keeping the feature extractor.
        params['F'] = params['F'][:-1]
        params['K'] = params['K'][:-1]
        params['nsides'] = params['nsides'][:-1]
        params['batch_norm'] = params['batch_norm'][:-1]
        params['statistics'] = 'mean'
        params['M'] = [n_classes]
    elif architecture != "FCN":
        raise ValueError('Unknown architecture {}.'.format(architecture))

    # Regularization (to prevent over-fitting).
    params['regularization'] = 0#1e-4  # Amount of L2 regularization over the weights (will be divided by the number of weights).
    params['dropout'] = 1  # Percentage of neurons to keep.
    params['dropFilt'] = 1 # percentage of filter to keep in each layer

    # Training.
    params['num_epochs'] = 30  # Number of passes through the training data.
    params['batch_size'] = 32  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: tf.train.exponential_decay(5e-1, step, decay_steps=5, decay_rate=1)#0.999)
    #params['scheduler'] = lambda step: tf.train.exponential_decay(5e-5, step, decay_steps=5, decay_rate=0.999)
    #params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params['optimizer'] = lambda lr: tf.train.GradientDescentOptimizer(lr)

    # Number of model evaluations during training (influence training time).
    n_evaluations = 40
    params['eval_frequency'] = int(params['num_epochs'] * ntrain / params['batch_size'] / n_evaluations)

    if verbose:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        print('bandwidth: {}'.format(bandwidth[0]))
        print('#pixels: {}'.format([4 * (bw)**2 for bw in bandwidth]))

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size']*4*(bandwidth[0])**2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs']*ntrain*4*(bandwidth[0])**2))

        n_steps = params['num_epochs'] * ntrain // params['batch_size']
        lr = [params['scheduler'](step).eval(session=tf.Session(config=config)) for step in [0, n_steps]]
        print('Learning rate will start at {:.1e} and finish at {:.1e}.'.format(*lr))

    return params


def get_params_mn40_optim(ntrain, EXP_NAME, Nside, n_classes, nfeat_in=6, architecture="FCN", verbose=True):

    """
    :param ntrain: int, number of elements in the training set
    :param EXP_NAME: string, name of experiment
    :param n_classes: int, number of classes present in SHREC17 dataset
    :param Nside: int, parameter of HEALpix
    :param architecture: string, type of NN
    :param verbose: bool, print info
    :return: parameters needed to create a deepsphere model
    """
    
    params = dict()
    params['dir_name'] = EXP_NAME
    params['num_feat_in'] = nfeat_in

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [16, 32, 64, 128, 256, 512, 1024, 2048, 2048, n_classes]  # Graph convolutional layers: number of feature maps.
    params['K'] = [4] * 10  # Polynomial orders.
#     params['K'] = [np.ceil(np.sqrt(3)*Nside).astype(int), 
#                    np.ceil(np.sqrt(3)*Nside//4).astype(int), 
#                    np.ceil(np.sqrt(3)*Nside//8).astype(int)]
    params['batch_norm'] = [True] * 10  # Batch normalization.
    params['M'] = []  # Fully connected layers: output dimensionalities.

    # Pooling.
    nsides = [Nside, Nside//2, Nside//4, Nside//4, Nside//8, Nside//8, 
              Nside//16, Nside//16, Nside//32, Nside//32, Nside//32]
    params['nsides'] = nsides
    params['indexes'] = None

    if architecture == "CNN":
        # Replace the last graph convolution and global average pooling by a fully connected layer.
        # That is, change the classifier while keeping the feature extractor.
        params['F'] = params['F'][:-1]
        params['K'] = params['K'][:-1]
        params['nsides'] = params['nsides'][:-1]
        params['batch_norm'] = params['batch_norm'][:-1]
        params['statistics'] = 'mean'
        params['M'] = [n_classes]
    elif architecture != "FCN":
        raise ValueError('Unknown architecture {}.'.format(architecture))

    # Regularization (to prevent over-fitting).
    params['regularization'] = 0  # Amount of L2 regularization over the weights (will be divided by the number of weights).
    params['dropout'] = 1  # Percentage of neurons to keep.
    params['dropFilt'] = 1 # percentage of filter to keep in each layer

    # Training.
    params['num_epochs'] = 50 #30  # Number of passes through the training data.
    params['batch_size'] = 32  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: tf.train.exponential_decay(2e-2, step, decay_steps=5, decay_rate=1)#decay_steps=7000, decay_rate=0.1, staircase=True)#0.999)
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=0.1)
    #params['optimizer'] = lambda lr: tf.train.GradientDescentOptimizer(lr)

    # Number of model evaluations during training (influence training time).
    n_evaluations = 200
    params['eval_frequency'] = int(params['num_epochs'] * ntrain / params['batch_size'] / n_evaluations)

    if verbose:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([12 * (nside)**2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size']*12*(Nside)**2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs']*ntrain*12*(Nside)**2))

        n_steps = params['num_epochs'] * ntrain // params['batch_size']
        lr = [params['scheduler'](step).eval(session=tf.Session(config=config)) for step in [0, n_steps]]
        print('Learning rate will start at {:.1e} and finish at {:.1e}.'.format(*lr))

    return params

def get_params_mn40(ntrain, EXP_NAME, Nside, n_classes, nfeat_in=6, architecture="FCN", verbose=True):

    """
    :param ntrain: int, number of elements in the training set
    :param EXP_NAME: string, name of experiment
    :param n_classes: int, number of classes present in SHREC17 dataset
    :param Nside: int, parameter of HEALpix
    :param architecture: string, type of NN
    :param verbose: bool, print info
    :return: parameters needed to create a deepsphere model
    """

    params = dict()
    params['dir_name'] = EXP_NAME
    params['num_feat_in'] = nfeat_in

    # Types of layers.
    params['conv'] = 'chebyshev5'  # Graph convolution: chebyshev5 or monomials.
    params['pool'] = 'max'  # Pooling: max or average.
    params['activation'] = 'relu'  # Non-linearity: relu, elu, leaky_relu, softmax, tanh, etc.
    params['statistics'] = 'mean'  # Statistics (for invariance): None, mean, var, meanvar, hist.

    # Architecture.
    params['F'] = [16, 16, 32, 32, 64, 64, 128, 128, 256, 256, n_classes]
    params['K'] = [4] * 11  # Polynomial orders.
#     params['K'] = [np.ceil(np.sqrt(3)*Nside).astype(int), 
#                    np.ceil(np.sqrt(3)*Nside//4).astype(int), 
#                    np.ceil(np.sqrt(3)*Nside//8).astype(int)]
    params['batch_norm'] = [True] * 11  # Batch normalization.
    params['M'] = []  # Fully connected layers: output dimensionalities.

    # Pooling.
    nsides = [Nside, Nside, Nside//2, Nside//2, Nside//4, Nside//4, Nside//8, Nside//8, Nside//16, Nside//16, Nside//32, Nside//32]
    params['nsides'] = nsides
    params['indexes'] = None

    if architecture == "CNN":
        # Replace the last graph convolution and global average pooling by a fully connected layer.
        # That is, change the classifier while keeping the feature extractor.
        params['F'] = params['F'][:-1]
        params['K'] = params['K'][:-1]
        params['nsides'] = params['nsides'][:-1]
        params['batch_norm'] = params['batch_norm'][:-1]
        params['statistics'] = 'mean'
        params['M'] = [n_classes]
    elif architecture != "FCN":
        raise ValueError('Unknown architecture {}.'.format(architecture))

    # Regularization (to prevent over-fitting).
    params['regularization'] = 0  # Amount of L2 regularization over the weights (will be divided by the number of weights).
    params['dropout'] = 1  # Percentage of neurons to keep.
    params['dropFilt'] = 1

    # Training.
    params['num_epochs'] = 30  # Number of passes through the training data.
    params['batch_size'] = 32  # Constant quantity of information (#pixels) per step (invariant to sample size).

    # Optimization: learning rate schedule and optimizer.
    params['scheduler'] = lambda step: tf.train.exponential_decay(2e-2, step, decay_steps=5, decay_rate=1)#0.999)
    #params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params['optimizer'] = lambda lr: tf.train.AdamOptimizer(lr, beta1=0.9, beta2=0.999, epsilon=0.1)

    # Number of model evaluations during training (influence training time).
    n_evaluations = 60
    params['eval_frequency'] = int(params['num_epochs'] * ntrain / params['batch_size'] / n_evaluations)

    if verbose:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        print('#sides: {}'.format(nsides))
        print('#pixels: {}'.format([12 * (nside)**2 for nside in nsides]))
        # Number of pixels on the full sphere: 12 * nsides**2.

        print('#samples per batch: {}'.format(params['batch_size']))
        print('=> #pixels per batch (input): {:,}'.format(params['batch_size']*12*(Nside)**2))
        print('=> #pixels for training (input): {:,}'.format(params['num_epochs']*ntrain*12*(Nside)**2))

        n_steps = params['num_epochs'] * ntrain // params['batch_size']
        lr = [params['scheduler'](step).eval(session=tf.Session(config=config)) for step in [0, n_steps]]
        print('Learning rate will start at {:.1e} and finish at {:.1e}.'.format(*lr))

    return params
