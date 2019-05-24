# Deepsphere model
* Nside = 32 to be similar to Esteves models
## experiment 1 no rotation
* git commit: xxxx
* shrec17 optim configuration
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//16]
** params['F'] = [32, 32, 64, 64, n_classes]
** params['batch_norm'] = [True] * 5
** params['num_epochs'] = 50
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: 1e-3
** params['K'] = [5] * 5  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~185k
* train on perturbed dataset, augmentation=1, random translation (object not on the center of the sphere)
* accuracy, F1 of training part: 96.53, 96.54
* accuracy, F1 of test part: 86.71, 86.77
* rotated dataset
* accuracy, F1 of test part: 63.52, 62.66
* time per batch: 0.052 s
* 50 epoch = 29m

## experiment 2 no rotation - augmentation
* git commit: xxxx
* shrec17 optim configuration
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//16]
** params['F'] = [32, 32, 64, 64, n_classes]
** params['batch_norm'] = [True] * 5
** params['num_epochs'] = 50
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: 1e-3
** params['K'] = [5] * 5  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~185k
* train on perturbed dataset, augmentation=3, random translation (object not on the center of the sphere)
* accuracy, F1 of training part (rot): 79.58, 79.42
* accuracy, F1 of training part (tr): 99.90, 99.90 ##
* accuracy, F1 of training part (rot+tr): 78.57, 78.35
* accuracy, F1 of test part (rot): 69.68, 69.67
* accuracy, F1 of test part (tr): 87.48, 87.58
* accuracy, F1 of test part (rot+tr): 69.71, 69.51
* time per batch: 0.052 s
* 50 epoch = 56m

## experiment 3 rotation - augmentation
* git commit: xxxx
* shrec17 optim configuration
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//16]
** params['F'] = [32, 32, 64, 64, n_classes]
** params['batch_norm'] = [True] * 5
** params['num_epochs'] = 50
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: 1e-3
** params['K'] = [5] * 5  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~185k
* train on perturbed dataset, augmentation=3, random rotation (object center of the sphere)
* accuracy, F1 of training part (rot): 99.43, 99.32 ##
* accuracy, F1 of training part (tr): 93.78, 93.70
* accuracy, F1 of training part (rot+tr): 94.95, 94.92
* accuracy, F1 of test part (rot): 85.25, 85.36
* accuracy, F1 of test part (tr): 80.67, 81.03
* accuracy, F1 of test part (rot+tr): 83.00, 83.15
* time per batch: 0.047 s
* 20 epoch = 16 m

## experiment 4 all augmentation
* git commit: xxxx
* shrec17 optim configuration
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//16]
** params['F'] = [32, 32, 64, 64, n_classes]
** params['batch_norm'] = [True] * 5
** params['num_epochs'] = 50
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: 1e-3
** params['K'] = [5] * 5  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~185k
* train on perturbed dataset, augmentation=3, random rotation (object center of the sphere)
* accuracy, F1 of training part (rot): 99.67, 99.67 ##
* accuracy, F1 of training part (tr):  99.54,  99.54 ##
* accuracy, F1 of training part (rot+tr): 99.57, 99.57 ##
* accuracy, F1 of test part (rot): 85.98, 86.09
* accuracy, F1 of test part (tr): 86.82, 86.96
* accuracy, F1 of test part (rot+tr): 85.80, 85.94
* time per batch: 0.060 s
* 10 epoch = 40 m

## experiment 5 no perturbation
* git commit: xxxx
* shrec17 optim configuration
** nsides = [Nside, Nside//2, Nside//4, Nside//8, Nside//16, Nside//16]
** params['F'] = [32, 32, 64, 64, n_classes]
** params['batch_norm'] = [True] * 5
** params['num_epochs'] = 50
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 0
** params['dropout'] = 1
** params['scheduler'] = lambda step: 1e-3
** params['K'] = [5] * 5  
** optimizer: Adam
** average pooling at the end
* nparams = weights + bias = ~185k
* train on perturbed dataset, augmentation=1, no rotation (object center of the sphere)
* accuracy, F1 of training part (rot): 74.15, 73.85     74.90, 75.36
* accuracy, F1 of training part (tr):  95.12, 95.08     95.63, 95.63
* accuracy, F1 of training part (rot+tr): 70.60, 70.40  71.52, 72.20
* accuracy, F1 of training part (): 99.40, 99.39 ##     99.40, 99.40
* accuracy, F1 of training part (Z): ?, ?               86.14, 86.34
* accuracy, F1 of test part (rot): 65.82, 65.98         64.79, 65.94
* accuracy, F1 of test part (tr): 85.35, 85.44          84.89, 85.19
* accuracy, F1 of test part (rot+tr): 63.21, 62.97      62.24, 63.39
* accuracy, F1 of test part (): 87.20, 87.26            86.14, 86.45
* accuracy, F1 of test part (Z): ?, ?                   67.79, 68.44
* time per batch: 0.060 s
* 30 epoch = 10 m

### Many problems arise
Seems to have problem with the preprocessing

## experiment 6 only z rotation (reversed)
* train on perturbed dataset, augmentation=3, random z rotation (equator) (object center of the sphere)
* accuracy, F1 of training part (Z): 99.39, 99.39 ##
* accuracy, F1 of training part (rot): 86.43, 86.26 
* accuracy, F1 of training part (tr):  95.01, 95.02 
* accuracy, F1 of training part (rot+tr): 82.90, 82.70 
* accuracy, F1 of training part (): 97.72, 97.73
* accuracy, F1 of test part (Z): 86.75, 87.08
* accuracy, F1 of test part (rot): 76.86, 77.81
* accuracy, F1 of test part (tr): 84.21, 84.65
* accuracy, F1 of test part (rot+tr): 73.66, 74.82
* accuracy, F1 of test part (): 85.66, 86.06
* time per batch: 0.060 s
* 10 epoch = 10.25 m

## experiment 7 rotation
* train on perturbed dataset, augmentation=3, random ZYZ rotation (object center of the sphere)
* accuracy, F1 of training part (Z): 85.90, 87.12 
* accuracy, F1 of training part (rot): 98.95, 98.93 ## 
* accuracy, F1 of training part (tr):  92.85, 92.88 
* accuracy, F1 of training part (rot+tr): 94.42, 94.44 
* accuracy, F1 of training part (): 95.09, 95.07
* accuracy, F1 of test part (Z): 56.13, 58.64
* accuracy, F1 of test part (rot): 84.09, 84.62
* accuracy, F1 of test part (tr): 79.85, 80.44
* accuracy, F1 of test part (rot+tr): 82.44, 83.16
* accuracy, F1 of test part (): 81.00, 81.61
* time per batch: 0.060 s
* 10 epoch = 10 m

## experiment no perturbation
* train on perturbed dataset, augmentation=15, all random rotation (object not on center of the sphere)
* accuracy, F1 of training part (Z): 87.12, 87.02 
* accuracy, F1 of training part (rot): 74.83, 74.86 
* accuracy, F1 of training part (tr):  95.83, 95.81 
* accuracy, F1 of training part (rot+tr): 71.46, 71.39
* accuracy, F1 of training part (): 99.40, 99.40 ##
* accuracy, F1 of test part (Z): 73.55, 73.19
* accuracy, F1 of test part (rot): 64.78, 65.12
* accuracy, F1 of test part (tr): 86.17, 86.36
* accuracy, F1 of test part (rot+tr): 63.36, 64.16
* accuracy, F1 of test part (): 87.8, 87.97
* time per batch: 0.064 s
* 10 epoch = 50 m

## final
* x/x: 87.8, z/z: 86.8, z/SO3: 76.9, SO3/SO3: 84.09

## experiment all perturbation
* train on perturbed dataset, augmentation=1, non random rotation (object center of the sphere)
* accuracy, F1 of training part (Z): 99.71, 99.70 ##
* accuracy, F1 of training part (rot): 99.79, 99.79 ##
* accuracy, F1 of training part (tr):  99.78, 99.78 ##
* accuracy, F1 of training part (rot+tr): 99.80, 99.80 ##
* accuracy, F1 of training part (): 87.07, 87.16 
* accuracy, F1 of test part (Z): 87.45, 87.52
* accuracy, F1 of test part (rot): 86.75, 86.89
* accuracy, F1 of test part (tr): 87.56, 87.60
* accuracy, F1 of test part (rot+tr): 86.49, 86.57
* accuracy, F1 of test part (): 87.8, 87.97
* time per batch: 0.063 s
* 30 epoch = 11 m

## deeper model
### no rotation
* train on dataset, augmentation=1, no random rotation (object center of the sphere)
* change std
* accuracy, F1 of training part (Z):  
* accuracy, F1 of training part (rot): 
* accuracy, F1 of training part (tr):   
* accuracy, F1 of training part (rot+tr): 
* accuracy, F1 of training part (): 97.29, 97.29 ##
* accuracy, F1 of test part (Z): 
* accuracy, F1 of test part (rot): 68.23, 68.28 
* accuracy, F1 of test part (tr): 
* accuracy, F1 of test part (rot+tr): 
* accuracy, F1 of test part (): 88.45, 88.59
* time per batch: 1.030 s
* 30 epoch = 45 m

## Nsides 128
### train no rotation
* train on dataset, augmentation=1, no random rotation (object center of the sphere)
* change std
* accuracy, F1 of training part (Z):  
* accuracy, F1 of training part (rot): 
* accuracy, F1 of training part (tr):   
* accuracy, F1 of training part (rot+tr): 
* accuracy, F1 of training part (): 97.29, 97.29 ##
* accuracy, F1 of test part (Z): 
* accuracy, F1 of test part (rot): 65.80, 74.04 
* accuracy, F1 of test part (tr): 
* accuracy, F1 of test part (rot+tr): 
* accuracy, F1 of test part (): 83.20, 89.03 (best 87.6)
* time per batch: 1.030 s
* 30 epoch = 2h44

# Cohen model
* done by jiang, similar experiment
* results: x/x 85.0

# Esteves model
* bw: 64
* complicated model using two branches, see report
* Results: z/z 88.9, SO3/SO3 86.9, z/SO3 78.6
* params: 0.5M

# Jiang model
* no augmentation, no random translation nor rotation
* results: x/x 90.5