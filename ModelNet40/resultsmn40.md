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
* accuracy, F1 of test part: 87.56, 87.63
* rotated dataset
* accuracy, F1 of test part: 69.71, 69.51
* time per batch: 0.052 s
* 50 epoch = 29m

# Cohen model

* results: x/x 85.0

# Esteves model
* bw: 64
* complicated model using two branches, see report
* Results: z/z 88.9, SO3/SO3 86.9, z/SO3 78.6
* params: 0.5M

# Jiang model

* results: x/x 90.5