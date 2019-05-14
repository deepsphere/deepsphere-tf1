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
* train on perturbed dataset, augmentation=3, random rotation (object center of the sphere)
* accuracy, F1 of training part (rot): 74.15, 73.85 
* accuracy, F1 of training part (tr):  95.12, 95.08 
* accuracy, F1 of training part (rot+tr): 70.60, 70.40 
* accuracy, F1 of training part (): 99.40, 99.39 ##
* accuracy, F1 of test part (rot): 65.82, 65.98
* accuracy, F1 of test part (tr): 85.35, 85.44
* accuracy, F1 of test part (rot+tr): 63.21, 62.97
* accuracy, F1 of test part (): 87.20, 87.26
* time per batch: 0.060 s
* 10 epoch = 40 m

### Many problems arise
Seems to have problem with the preprocessing

# Cohen model

* results: x/x 85.0

# Esteves model
* bw: 64
* complicated model using two branches, see report
* Results: z/z 88.9, SO3/SO3 86.9, z/SO3 78.6
* params: 0.5M

# Jiang model

* results: x/x 90.5