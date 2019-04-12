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
* git commit: f2ff2dd02136c97ab61b74d912222eba5cb60778
* similar parameters as Cohen simple:
** nsides = [Nside, Nside//4, Nside//8] ==> npixel [12'228, 768, 192]
** params['F'] = [100, 100]
** params['batch_norm'] = [True] * 2
** params['num_epochs'] = 108
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0.08
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

## experiment 5.1
* git commit: 5ffd65b74f6665ec7dcf01fcae4eecf5b1446d26
* similar parameters as Cohen simple:
** nsides = [Nside, Nside//4, Nside//8] ==> npixel [12'228, 768, 192]
** params['F'] = [100, 100]
** params['batch_norm'] = [True] * 2
** params['num_epochs'] = 100  //  300
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: 5e-1
** params['K'] = [5] * 2 
** optimizer: SGD
** average pooling before fully-connected
** params['M'] = [55] Fully connected
* nparams = weights + bias = ~59k
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of training part: (86.60, 86.44, 0.44)  //  (90.52, 90.21, 0.30)
* accuracy, F1, loss of validation part: (77.54, 76.64, 1.02)  //  (78.90, 78.05, 1.12)
* accuracy, F1, loss of test part: (73.07, 72.84, 1.22)  //  (74.71, 74.11, 1.30)
* test on val_perturbed dataset: P@N 0.656, R@N 0.684, F1@N 0.655, mAP 0.634, NDCG 0.660  //  0.676, 0.698, 0.677, 0.655, 0.680
* test on test_perturbed dataset: P@N 0.619, R@N 0.649, F1@N 0.617, mAP 0.588, NDCG 0.669  //  0.656, 0.659, 0.649, 0.620, 0.702
* time per batch: 0.12 s, 2415 MiB, 3h45 // 11h training

## experiment 5.2
* git commit: ecc809d1dc1aa27b9b5245056f9efa2469a4a9f2
* similar parameters as Cohen simple:
** nsides = [Nside, Nside//4, Nside//8] ==> npixel [12'228, 768, 192]
** params['F'] = [100, 100]
** params['batch_norm'] = [True] * 2
** params['num_epochs'] = 100  //  300
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: 5e-1
** params['K'] = sqrt(3) * nsides  
** optimizer: SGD
** average pooling before fully-connected
** params['M'] = [55] Fully connected
* nparams = weights + bias = ~180k
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of training part: (91.34, 91.17, 0.266)  //  (96.53, 96.54, 0.11)
* accuracy, F1, loss of validation part: (78.84, 78.02, 1.06)  //  (79.97, 79.57, 1.21)
* accuracy, F1, loss of test part: (73.95, 73.62, 1.26)  //  (74.91, 75.08, 1.45)
* test on val_perturbed dataset: P@N 0.675, R@N 0.667, F1@N 0.674, mAP 0.654, NDCG 0.677  //  0.694, 0.703, 0.691, 0.663, 0.694
* test on test_perturbed dataset: P@N 0.664, R@N 0.656, F1@N 0.650, mAP 0.618, NDCG 0.702  //  0.666, 0.659, 0.655, 0.624, 0.710
* time per batch: 0.59 s
* Remarks: num_epochs too big, validation loss is increasing, but f1 score keeps increasing, 2415 MiB, 18 // 61 hours 

## experiment 5.3
* git commit: b8f4ff3507cd4dc691eac7aaaabe13dfeb46d47e
* similar parameters as Cohen simple:
** nsides = [Nside, Nside//4, Nside//8] ==> npixel [12'228, 768, 192]
** params['F'] = [100, 100]
** params['batch_norm'] = [True] * 2
** params['num_epochs'] = 25  //  300
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: 5e-1
** params['K'] = [5] * 2  
** optimizer: SGD
** average pooling before fully-connected
** params['M'] = [55] Fully connected
* nparams = weights + bias = ~60k
* train on perturbed dataset, augmentation=3, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of training part: (87.16, 87.11, 0.87)  //  
* accuracy, F1, loss of validation part: (78.75, 78.28, 0.873)  //  (79.99, 79.70, 1.07)
* accuracy, F1, loss of test part: (75.87, 75.83, 0.996)  //  (76.37, 76.46, 1.29)
* test on val_perturbed dataset: P@N 0.690, R@N 0.701, F1@N 0.686, mAP 0.662, NDCG 0.688  //  0.692 0.700 0.688 0.662 0.692
* test on test_perturbed dataset: P@N 0., R@N 0., F1@N 0., mAP 0., NDCG 0.  //  0.674 0.669 0.664 0.631 0.714
* time per batch: 0.12 s
* Remarks: 2419 MiB on GPU, // 1 jour 22h

## experiment 6 equiangular
* git commit: c2b14ccd96be7e5f4810a949ba8ae3cc822565fb
* similar parameters as Cohen simple:
** bandwidths = [64, 16, 10] 
** params['F'] = [100, 100]
** params['batch_norm'] = [True] * 2
** params['num_epochs'] = 71
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: 5e-1
** params['K'] = [bw] * 2  
** optimizer: SGD
** average pooling before fully-connected
** params['M'] = [55] Fully connected
* nparams = weights + bias = ~204k
* train on perturbed dataset, augmentation=3, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of training part: (84.63, 84.40, 0.54)
* accuracy, F1, loss of validation part: (74.42, 73.38, 1.26)
* accuracy, F1, loss of test part: (69.43, 69.35, 1.49)
* test on val_perturbed dataset: P@N 0.621, R@N 0.647, F1@N 0.617, mAP 0.590, NDCG 0.624
* test on test_perturbed dataset: P@N 0.597, R@N 0.612, F1@N 0.587, mAP 0.548, NDCG 0.634
* time per batch: 1.0 s
* Remarks: 4419 MiB on GPU, 

## experiment 7.1 best
* git commit: xxxx
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//16]
** params['F'] = [32, 32, 64, 64, n_classes]
** params['batch_norm'] = [True] * 5
** params['num_epochs'] = 300
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0.1
** params['dropout'] = 1
** params['scheduler'] = lambda step: 1e-3
** params['K'] = [5] * 5  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~55k
* train on perturbed dataset, augmentation=1, random translation (object not on the center of the sphere)
* accuracy, F1, loss of training part: (90.12, 89.85, 0.52)
* accuracy, F1, loss of validation part: (78.26, 77.62, 1.20)
* accuracy, F1, loss of test part: (73.94, 73.66, 1.33)
* time per batch: 0.12 s
* Remarks: 883 MiB on GPU, 11h to train

## experiment 7.2 best
* git commit: 36c03cef0b262575aa9d0d9b85bd6ff2ae1ffaa6
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//32, Nside//32]
** params['F'] = [32, 32, 64, 64, 128, n_classes]
** params['batch_norm'] = [True] * 6
** params['num_epochs'] = 100
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0.5
** params['dropout'] = 1
** params['scheduler'] = lambda step: 1e-3
** params['K'] = [5] * 6  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~190k
* train on perturbed dataset, augmentation=1, random translation (object not on the center of the sphere)
* accuracy, F1, loss of training part: (91.18, 90.99, 0.61)
* accuracy, F1, loss of validation part: (79.52, 78.65, 1.25)
* accuracy, F1, loss of test part: (74.76, 74.48, 1.40)
* test on val_perturbed dataset: P@N 0.680, R@N 0.703, F1@N 0.682, mAP 0.662, NDCG 0.690
* test on test_perturbed dataset: P@N 0.648, R@N 0.656, F1@N 0.643, mAP 0.619, NDCG 0.705
* time per batch: 0.14 s
* Remarks: 1395 MiB on GPU, 4h to train

## experiment 8 128Nsides
* git commit: 
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//32, Nside//32]
** params['F'] = [16, 32, 64, 128, 256, n_classes]
** params['batch_norm'] = [True] * 6
** params['num_epochs'] = 100
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0.5
** params['dropout'] = 1
** params['scheduler'] = lambda step: 1e-3
** params['K'] = [5] * 6  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~233k
* train on perturbed dataset, augmentation=1, random translation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (79.45, 78.31, 0.92)
* accuracy, F1, loss of test part: (75.94, 75.09, 1.04)
* test on val_perturbed dataset: P@N 0.682, R@N 0.715, F1@N 0.687, mAP 0.670, NDCG 0.691
* test on test_perturbed dataset: P@N 0.670, R@N 0.672, F1@N 0.661, mAP 0.630, NDCG 0.715
* time per batch: 1.15 s
* Remarks: 8623 MiB on GPU (4491 MiB on notebook), 1j 20h to train

## experiment 9 64 Nsides
* git commit: 20f5aa17a40cafa6e846628cb51598122be5cdeb
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//32, Nside//32]
** params['F'] = [16, 32, 64, 128, 256, n_classes]
** params['batch_norm'] = [True] * 6
** params['num_epochs'] = 100 (30 is sufficient)
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0.5
** params['dropout'] = 0.7
** params['scheduler'] = lambda step: 1e-2
** params['K'] = [5] * 6  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~233k
* train on perturbed dataset, augmentation=1, random translation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (80.69, 80.12, 1.07)
* accuracy, F1, loss of test part: (75.99, 76.07, 1.24)
* test on val_perturbed dataset: P@N 0.699, R@N 0.716, F1@N 0.700, mAP 0.678, NDCG 0.702
* test on test_perturbed dataset: P@N 0.661, R@N 0.665, F1@N 0.654, mAP 0.626, NDCG 0.706
* time per batch: 0.26 s
* Remarks: 2443 MiB on GPU (1395 MiB on notebook), 9h34m to train

## experiment 9.2 64 Nsides
* git commit: 20f5aa17a40cafa6e846628cb51598122be5cdeb
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//32, Nside//32]
** params['F'] = [16, 32, 64, 128, 256, n_classes]
** params['batch_norm'] = [True] * 6
** params['num_epochs'] = 100 (30 is sufficient)
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0.5
** params['dropout'] = 0.7
** params['scheduler'] = lambda step: 1e-2
** params['K'] = [2] * 6  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~102k
* train on perturbed dataset, augmentation=1, random translation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (79.84, 78.69, 1.00)
* accuracy, F1, loss of test part: (75.23, 74.41, 1.16)
* test on val_perturbed dataset: P@N 0.680, R@N 0.712, F1@N 0.685, mAP 0.670, NDCG 0.689
* test on test_perturbed dataset: P@N 0.667, R@N 0.673, F1@N 0.661, mAP 0.631, NDCG 0.709
* time per batch: 0.11 s
* Remarks: xxx MiB on GPU (891 MiB on notebook), 5h09m to train

## experiment 10 better graph
* git commit: 20f5aa17a40cafa6e846628cb51598122be5cdeb
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//32, Nside//32]
** params['F'] = [16, 32, 64, 128, 256, n_classes]
** params['batch_norm'] = [True] * 6
** params['num_epochs'] = 50
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: 1e-2
** params['K'] = [4] * 6  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~189k
* train on perturbed dataset, augmentation=1, random translation (object not on the center of the sphere)
* training set loss: 9.28e-3
* accuracy, F1, loss of validation part: (81.86, 81.47, 0.97)
* accuracy, F1, loss of test part: (78.81, 78.83, 1.19)
* test on val_perturbed dataset: P@N 0.715, R@N 0.728, F1@N 0.715, mAP 0.690, NDCG 0.716
* test on test_perturbed dataset: P@N 0.699, R@N 0.695, F1@N 0.690, mAP 0.662, NDCG 0.743
* time per batch: 0.05 s
* Remarks: 875 MiB on GPU, 1h18m to train (40 min would have been sufficient)

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
* testing part: accuracy 75.93, f1 76.19
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
* validation part: accuracy 82.99, f1 82.73
* testing part: accuracy 78.09, f1 78.45
* test on val_perturbed dataset: P@N 0.733, R@N 0.740, F1@N 0.730, mAP 0.702, NDCG 0.725
* test on test_perturbed dataset: P@N 0.699, R@N 0.693, F1@N 0.690, mAP 0.657, NDCG 0.740
* time per batch: 0.46 s  //  0.38 s
* time to run: 32 hours