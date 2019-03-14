# Deepsphere model
* Nside = 32 to be similar to Cohen and Esteves models
## experiment 1
* git commit: 9ec3a5f4de5c73333cf0cdb8ccf79eb9d639f328
* similar parameters as Cohen simple:
** nsides = [Nside, Nside//4, Nside//8, Nside//8] ==> npixel [12'228, 768, 192]
** params['F'] = [100, 100, n_classes]
** params['batch_norm'] = [True] * 3
** params['num_epochs'] = 30
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: tf.train.exponential_decay(2e-2, step, decay_steps=5, decay_rate=0.999)   # peut être changer
** params['K'] = [Nside]  Pas sur de ça, mais pas encore trouvé moyen de faire un truc similaire
** params['M'] = [55] Fully connected
* nparams = weights (500+50000+27500+580800) + bias (100 + 100 + 55) = ~600k
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (65.82, 64.05, 1.86+e03)
* test on val_perturbed dataset: P@N 0.520, R@N 0.566, F1@N 0.527, mAP 0.483, NDCG 0.530
* test on test_perturbed dataset: xxx
* time per batch: 0.33 s
TODO: revoir les résultats

## experiment 2
* git commit: 5874eb9a337610361a32f30c9d8060afd682ba97
* similar parameters as Cohen simple:
** nsides = [Nside, Nside//4, Nside//8] ==> npixel [12'228, 768, 192]
** params['F'] = [100, 100]
** params['M'] = [nclasses]
** params['batch_norm'] = [True] * 2
** params['num_epochs'] = 80
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 1
** params['dropout'] = 1
** params['scheduler'] = lambda step: tf.train.exponential_decay(2e-2, step, decay_steps=5, decay_rate=0.999)   # peut être changer
** params['K'] = sqrt(3) * nsides  Pas sur de ça, mais pas encore trouvé moyen de faire un truc similaire
** average pooling before fully-connected
** params['M'] = [55] Fully connected
* nparams = weights (33600+130000+5500) + bias (100 + 100) = ~170k
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (71.19, 68.20, 1.2)
* test on val_perturbed dataset: xxx
* test on test_perturbed dataset: P@N 0.550, R@N 0.607, F1@N 0.560, mAP 0.525, NDCG 0.608
* time per batch: 0.58 s

## experiment 0
* git commit: a120887dd9098a1e29b91066b63d7fdce5661428
* similar parameters as Cohen simple:
** nsides = [Nside, Nside//4, Nside//8] ==> npixel [12'228, 768, 192]
** params['F'] = [100, 100]
** params['batch_norm'] = [True] * 2
** params['num_epochs'] = 80
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 1
** params['dropout'] = 1
** params['scheduler'] = lambda step: tf.train.exponential_decay(2e-2, step, decay_steps=5, decay_rate=0.999)   # peut être changer
** params['K'] = [5] * 2  
** average pooling before fully-connected
** params['M'] = [55] Fully connected
* nparams = weights (500+50000+5500) + bias (100 + 100) = ~59k
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (72.53, 69.61, 1.23)
* accuracy, F1, loss of test part: (67.41, 64.92, 1.4)
* test on val_perturbed dataset: P@N 0.576, R@N 0.658, F1@N 0.596, mAP 0.579, NDCG 0.606
* test on test_perturbed dataset: P@N 0.564, R@N 0.600, F1@N 0.567, mAP 0.532, NDCG 0.619
* time per batch: 0.12 s

## essai
* git commit: ?? 
* random parameters:
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//16] 
** params['F'] = [16*6, 32*6, 64*6, 64*6, 55]
** params['batch_norm'] = [True] * 5
** params['num_epochs'] = 40
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: tf.train.exponential_decay(2e-4, step, decay_steps=1, decay_rate=0.999)   # peut être changer
** params['K'] = [5] * 5  
** average pooling but no fully connected
* nparams = weights  + bias  = 1.307M
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (74.56, 72.15, 7.57e-1)
* accuracy, F1, loss of test part: (69.99, 68.11, 1.1)
* test on val_perturbed dataset: P@N 0., R@N 0., F1@N 0., mAP 0., NDCG 0.
* test on test_perturbed dataset: P@N 0.601, R@N 0.624, F1@N 0.600, mAP 0.562, NDCG 0.651
* time per batch: 0.34 s

## experiment 3
* git commit: fd5fb8156d6c482c886ec6863f13b1f19ef41fdd
* similar parameters as Cohen simple:
** nsides = [Nside, Nside//4, Nside//8] ==> npixel [12'228, 768, 192]
** params['F'] = [100, 100]
** params['batch_norm'] = [True] * 2
** params['num_epochs'] = 145
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0.1
** params['dropout'] = 1
** params['scheduler'] = lambda step: tf.train.exponential_decay(4e-2, step, decay_steps=1, decay_rate=0.999)   # peut être changer
** params['K'] = [5] * 2  
** average pooling before fully-connected
** params['M'] = [55] Fully connected
* nparams = weights (500+50000+5500) + bias (100 + 100) = ~59k
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (75.36, 73.17, 1.12)
* accuracy, F1, loss of test part: (70.12, 68.36, 1.28)
* test on val_perturbed dataset: NaN
* test on test_perturbed dataset: P@N 0.594, R@N 0.621, F1@N 0.596, mAP 0.562, NDCG 0.645
* time per batch: 0.12 s

## experiment 4
* git commit: 62a667aea3b555b9c92cebeba02fcaabf4138d0b
* similar parameters as Cohen simple:
** nsides = [Nside, Nside//4, Nside//8] ==> npixel [12'228, 768, 192]
** params['F'] = [100, 100]
** params['batch_norm'] = [True] * 2
** params['num_epochs'] = 108
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0.1
** params['dropout'] = 1
** params['scheduler'] = lambda step: tf.train.exponential_decay(4e-2, step, decay_steps=1, decay_rate=0.999)   # peut être changer
** params['K'] = sqrt(3) * nsides 
** average pooling before fully-connected
** params['M'] = [55] Fully connected
* nparams = weights + bias = ~170k
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (76.12, 74.18, 1.03)
* accuracy, F1, loss of test part: (71.73, 70.29, 1.17)
* test on val_perturbed dataset: NaN
* test on test_perturbed dataset: P@N 0.617, R@N 0.641, F1@N 0.616, mAP 0.582, NDCG 0.672
* time per batch: 0.56 s

## experiment 4.2
* git commit: 
* similar parameters as Cohen simple:
** nsides = [Nside, Nside//4, Nside//8] ==> npixel [12'228, 768, 192]
** params['F'] = [100, 100]
** params['batch_norm'] = [True] * 2
** params['num_epochs'] = 108
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0.1
** params['dropout'] = 1
** params['scheduler'] = lambda step: tf.train.exponential_decay(4e-2, step, decay_steps=1, decay_rate=0.999)   # peut être changer
** params['K'] = [5] * 2 
** optimizer: SGD
** average pooling before fully-connected
** params['M'] = [55] Fully connected
* nparams = weights + bias = ~59k
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (76.17, 74.35, 0.97)
* accuracy, F1, loss of test part: (72.09, 70.68, 1.11)
* test on val_perturbed dataset: P@N 0.630, R@N 0.682, F1@N 0.641, mAP 0.619, NDCG 0.649
* test on test_perturbed dataset: P@N 0.623, R@N 0.639, F1@N 0.619, mAP 0.584, NDCG 0.672
* time per batch: 0.11 s

# Cohen model
## paper experiment
* parameters:
** bandwidth = [128, 32, 22, 7] ==> npixel [65'536, 262'144(8'192), 85'184(1'936), 2'744(196)]
** features = [50, 70, 350, n_classes]
** batch norm 3D = yes
** num epoch = 300
** batch_size = 32
** activation = relu
** no regularization
** no fully connected: pixel max (class with strongest representation)
** learning rate = 0.5
** kernel in spatial space, grid is a ring around equator of size (2 * bandwidth, 1) ==> whole space?
* nparams = ~1.4M
* augmented with fixed translation
* P@N 0.701, R@N 0.711, F1@N 0.699, mAP 0.676, NDCG 0.756

## experiment simple 0
* parameters
** bandwidth = [64, 16, 10] ==> npixel [16384, 32768(1024), 8000(400)]
** features = [100, 100, n_classes]
** batch norm = yes, affine
** num epoch = 125
** batch_size = 32
** activation = relu
** no regularization
** no fully connected: so3 integration
** learning rate = 0.5
** kernel in spatial space, grid is a ring around equator of size (2 * bandwidth, 1) ==> non-local filter
** integration on SO3 before fully-connected
* nparams = ~400k
* no augmentation, random translation and rotation
* test on val_perturbed dataset: P@N 0.701, R@N 0.708, F1@N 0.698, mAP 0.665, NDCG 0.695
* test on test_perturbed dataset: P@N 0.673, R@N 0.666, F1@N 0.662, mAP 0.627, NDCG 0.711
* time per batch: 0.39 s

## experiment simple
* parameters
** bandwidth = [64, 16, 10] ==> npixel [16384, 32768(1024), 8000(400)]
** features = [100, 100, n_classes]
** batch norm 3D = yes
** num epoch = 300
** batch_size = 32
** activation = relu
** no regularization
** no fully connected: so3 integration
** learning rate = 0.5
** kernel in spatial space, grid is a ring around equator of size (2 * bandwidth, 1) ==> non-local filter
** integration on SO3 before fully-connected
* nparams = ~400k
* no augmentation, random translation and rotation
* validation part: accuracy 80.81, f1 80.44
* test on val_perturbed dataset: P@N 0.701, R@N 0.710, F1@N 0.699, mAP 0.667, NDCG 0.699
* test on test_perturbed dataset: P@N 0.669, R@N 0.662, F1@N 0.659, mAP 0.621, NDCG 0.707
* time per batch: 0.43 s

## experiment augmentation
* parameters
** bandwidth = [64, 16, 10] ==> npixel [16384, 32768(1024), 8000(400)]
** features = [100, 100, n_classes]
** batch norm 3D = yes
** num epoch = 300
** batch_size = 32
** activation = relu
** no regularization
** no fully connected: so3 integration
** learning rate = 0.5
** kernel in spatial space, grid is a ring around equator of size (2 * bandwidth, 1) ==> non-local filter
** integration on SO3 before fully-connected
* nparams = ~400k
* augmentation, 3 random translations and rotations
* training part: accuracy 0.96, loss 0.13
* validation part: accuracy , f1 
* test on val_perturbed dataset: P@N 0.701, R@N 0.710, F1@N 0.699, mAP 0.667, NDCG 0.699
* test on test_perturbed dataset: P@N 0.669, R@N 0.662, F1@N 0.659, mAP 0.621, NDCG 0.707
* time per batch: 0.46 s