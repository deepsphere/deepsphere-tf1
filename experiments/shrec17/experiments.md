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

## try to get rid of overfitting
* parameters
** L2 regularization: training loss worsens, validation loss worsens, validation accuracy drops 1 point
** drop filters in convolution layers: training loss worsens, validation loss same, validation accuracy same
** dropout in fully connected layer: training loss same, validation loss same, validation accuracy increases 1 point
** reduce number of features: training loss same, validation loss same, validation accuracy drops 1 point
** add triplet_loss (see Esteved paper): training loss same, validation loss same, validation accuracy increases 1 point
** augmentation with noise: training loss same, validation loss same, validation accuracy same
** augmentation with noise and translation: 
* best_val (triplet_loss)
** accuracy, F1, loss of validation part: (82.21, 81.80, 0.92)
** accuracy, F1, loss of test part: (78.66, 78.81, 1.46)
** test on val_perturbed dataset: P@N 0.737, R@N 0.742, F1@N 0.734, mAP 0.708, NDCG 0.733
** test on test_perturbed dataset: P@N 0.703, R@N 0.703, F1@N 0.697, mAP 0.668, NDCG 0.748

## True better graph
### first layer full
* accuracy, F1, loss of validation part: (82.19, 81.76, 0.92)
* accuracy, F1, loss of test part: (77.80, 77.98, 1.17)
* Remarks: overfit sooner and at a greater pace
* 27 epochs, 54 min
### all layers full
* accuracy, F1, loss of validation part: (82.12, 81.66, 0.90)
* accuracy, F1, loss of test part: (78.39, 78.37, 1.14)
* Remarks: overfit sooner and at a greater pace, even slower
### no layer full
* accuracy, F1, loss of validation part: (82.45, 82.03, 0.98)
* accuracy, F1, loss of test part: (78.13, 78.15, 1.26)
### no better graph
* accuracy, F1, loss of validation part: (82.45, 82.01, 1.02)
* accuracy, F1, loss of test part: (78.26, 78.50, 1.25)

## Augmentation test
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//32, Nside//32]
** params['F'] = [16, 32, 64, 128, 256, n_classes]
** params['batch_norm'] = [True] * 6
** params['num_epochs'] = 50
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: 2e-2
** params['K'] = [4] * 6  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~189k
###### all perturbations
* train on perturbed dataset, 3 random rotations + 3 random translations (object not on the center of the sphere)
* training set loss: 6.3e-3
** rotated dataset
* accuracy, F1, loss of validation part: (84.08, 83.80, 1.08)
* accuracy, F1, loss of test part: (79.61, 79.91, 1.45)
* test on val_perturbed dataset: P@N 0.745, R@N 0.755, F1@N 0.745, mAP 0.723, NDCG 0.747
                           macro: P@N 0.522, R@N 0.553, F1@N 0.520, mAP 0.480, NDCG 0.536
* test on test_perturbed dataset: P@N 0.719, R@N 0.710, F1@N 0.708, mAP 0.680, NDCG 0.758
                           macro: P@N 0.457, R@N 0.490, F1@N 0.449, mAP 0.410, NDCG 0.470
** non-rotated dataset
* accuracy, F1, loss of validation part: (84.53, 84.18, 1.08)
* accuracy, F1, loss of test part: (80.42, 80.65, 1.40)
* test on val_perturbed dataset: P@N 0.750, R@N 0.760, F1@N 0.749, mAP 0.728, NDCG 0.750
                           macro: P@N 0.530, R@N 0.557, F1@N 0.525, mAP 0.486, NDCG 0.543
* test on test_perturbed dataset: P@N 0.725, R@N 0.717, F1@N 0.715, mAP 0.686, NDCG 0.764
                           macro: P@N 0.475, R@N 0.508, F1@N 0.468, mAP 0.428, NDCG 0.486
* time per batch: 0.05 s
* Remarks: 881 MiB on GPU, 5h04m to train
###### only rot
* train on perturbed dataset, 3 random rotations (object not on the center of the sphere)
* training set loss: 3.9e-3
** rotated dataset
* accuracy, F1, loss of validation part: (83.85, 83.35, 1.06)
* accuracy, F1, loss of test part: (79.25, 79.26, 1.39)
* test on val_perturbed dataset: P@N 0.742, R@N 0.754, F1@N 0.742, mAP 0.720, NDCG 0.741
                           macro: P@N 0.500, R@N 0.540, F1@N 0.502, mAP 0.463, NDCG 0.517
* test on test_perturbed dataset: P@N 0.709, R@N 0.706, F1@N 0.701, mAP 0.674, NDCG 0.753
                           macro: P@N 0.454, R@N 0.502, F1@N 0.453, mAP 0.420, NDCG 0.479
** non-rotated dataset
* accuracy, F1, loss of validation part: (83.81, 83.39, 1.05)
* accuracy, F1, loss of test part: (79.26, 79.30, 1.38)
* test on val_perturbed dataset: P@N 0.740, R@N 0.751, F1@N 0.740, mAP 0.717, NDCG 0.740
                           macro: P@N 0.508, R@N 0.544, F1@N 0.507, mAP 0.466, NDCG 0.521
* test on test_perturbed dataset: P@N 0.712, R@N 0.538, F1@N 0.591, mAP 0.503, NDCG 0.594
                           macro: P@N 0.416, R@N 0.512, F1@N 0.427, mAP 0.415, NDCG 0.466
* time per batch: 0.05 s
* Remarks: 881 MiB on GPU, 2h43m to train
###### no rot
* train on perturbed dataset, 3 random translations (object not on the center of the sphere)
* training set loss: 5.04e-3
** rotated dataset
* accuracy, F1, loss of validation part: (83.71, 83.30, 1.04)
* accuracy, F1, loss of test part: (79.82, 80.10, 1.32)
* test on val_perturbed dataset: P@N 0.737, R@N 0.749, F1@N 0.737, mAP 0.716, NDCG 0.737
                           macro: P@N 0.514, R@N 0.544, F1@N 0.511, mAP 0.475, NDCG 0.532
* test on test_perturbed dataset: P@N 0.715, R@N 0.707, F1@N 0.705, mAP 0.675, NDCG 0.755
                           macro: P@N 0.448, R@N 0.495, F1@N 0.448, mAP 0.412, NDCG 0.474
** non-rotated dataset
* accuracy, F1, loss of validation part: (84.06, 83.78, 1.02)
* accuracy, F1, loss of test part: (79.57, 79.76, 1.33)
* test on val_perturbed dataset: P@N 0.745, R@N 0.754, F1@N 0.744, mAP 0.723, NDCG 0.745
                           macro: P@N 0.520, R@N 0.552, F1@N 0.518, mAP 0.484, NDCG 0.540
* test on test_perturbed dataset: P@N 0.709, R@N 0.704, F1@N 0.700, mAP 0.671, NDCG 0.751
                           macro: P@N 0.448, R@N 0.492, F1@N 0.447, mAP 0.411, NDCG 0.471
* time per batch: 0.05 s
* Remarks: 881 MiB on GPU, 2h58m to train
###### add triplet loss and better graph
* train on perturbed dataset, 3 random translations (object not on the center of the sphere)
* training set loss: 0.03 + 0.02
** rotated dataset
* accuracy, F1, loss of validation part: (84.28, 84.10, 1.12)
* accuracy, F1, loss of test part: (80.12, 80.52, 1.48)
* test on val_perturbed dataset: P@N 0.749, R@N 0.753, F1@N 0.746, mAP 0.719, NDCG 0.743
                           macro: P@N 0.529, R@N 0.556, F1@N 0.524, mAP 0.486, NDCG 0.544
* test on test_perturbed dataset: P@N 0.723, R@N 0.707, F1@N 0.705, mAP 0.675, NDCG 0.755
                           macro: P@N 0.465, R@N 0.505, F1@N 0.460, mAP 0.422, NDCG 0.482
** non-rotated dataset
* accuracy, F1, loss of validation part: (84.06, 83.89, 1.11)
* accuracy, F1, loss of test part: (79.88, 80.28, 1.48)
* test on val_perturbed dataset: P@N 0.748, R@N 0.753, F1@N 0.746, mAP 0.722, NDCG 0.746
                           macro: P@N 0.518, R@N 0.542, F1@N 0.513, mAP 0.473, NDCG 0.531
* test on test_perturbed dataset: P@N 0.719, R@N 0.713, F1@N 0.710, mAP 0.679, NDCG 0.755
                           macro: P@N 0.468, R@N 0.509, F1@N 0.464, mAP 0.426, NDCG 0.486
* time per batch: 0.16 s
* Remarks: xxxx MiB on GPU, 7h52m to train


## Augmented Dataset no rotation / on rotated dataset
* accuracy, F1, loss of validation part: (81.86, 81.47, 0.97)
* accuracy, F1, loss of test part: (77.78, 77.98, 1.17)
* test on val_perturbed dataset: P@N 0.723, R@N 0.736, F1@N 0.723, mAP 0.699, NDCG 0.726
* test on test_perturbed dataset: P@N 0.704, R@N 0.695, F1@N 0.693, mAP 0.663, NDCG 0.747

## Equiangular graph with correct pooling
* git commit: c279685d6e7258a6d4f12e98c680230ba3b6c754
** bw = [64, 32, 16, 8, 4, 2, 2]
** params['F'] = [16, 32, 64, 128, 256, n_classes]
** params['batch_norm'] = [True] * 6
** params['num_epochs'] = 20
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: 5e-1
** params['K'] = [4] * 6  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~189k
* train on perturbed dataset, augmentation=5, random translations and rotations (object not on the center of the sphere)
* training set loss: 0.22
* accuracy, F1, loss of validation part: (83.05, 82.61, 0.81)
* accuracy, F1, loss of test part: (79.25, 79.36, 1.02)
* test on val_perturbed dataset: P@N 0.730, R@N 0.745, F1@N 0.730, mAP 0.708, NDCG 0.731
                          macro: P@N 0.497, R@N 0.537, F1@N 0.492, mAP 0.453, NDCG 0.505
* test on test_perturbed dataset: P@N 0.709, R@N 0.700, F1@N 0.698, mAP 0.665, NDCG 0.748
                          macro: P@N 0.439, R@N 0.489, F1@N 0.439, mAP 0.403, NDCG 0.459
* time per batch: 0.033 s
* Remarks: 889 MiB on GPU, 2h30m to train

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
* augmented with 6 fixed translation
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

# Esteves model
## paper experiment
* parameters:
** bandwidth = [64, 64, 64, 32, 32, 16, 16, 8, 8]
** features (main branche) = [16, 16, 32 (x2), 32, 64 (x2), 64, 128 (x2), 128, n_classes]
** batch norm 3D = yes
** num epoch = 32
** batch_size = 32
** activation = prelu
** no regularization
** gap and next a fully connected to nclasses
** learning rate = {0: 0.001, 16: 0.0002, 24: 0.00004}
** localized filter with 8 parameters in frequential space
* nparams = ~500k
* augmented with 16 random SO3 rotations
* P@N 0.717, R@N 0.737, mAP 0.685
* macro P@N 0.450, R@N 0.550, mAP 0.444
* train time: wall time: 10319.81s, per epoch: 309.79, per epoch process: 382
* train acc = 0.9122, val acc = 0.9122, test acc = 0.7918, test F1 = 0.7936
* 2449 MiB on GPU