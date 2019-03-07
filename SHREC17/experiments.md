# Deepsphere model
## experiment 1
* git commit:
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
* nparams = weights (500+50000+27500+580800) + bias (100 + 100 + 55) = ~600k
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (65.82, 64.05, 1.86+e03)
* test on val_perturbed dataset: P@N 0.154, R@N 0.159, F1@N 0.155, mAP 0.144, NDCG 0.214
* test on test_perturbed dataset: P@N 0.520, R@N 0.566, F1@N 0.527, mAP 0.483, NDCG 0.530

## experiment 2

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
** frequential space, convolution ???  kernel, grid is a ring around equator of size (2*bandwidth, 1) ==> whole space?
* nparams = ~1.4M
* augmented with fixed translation
* P@N 0.701, R@N 0.711, F1@N 0.699, mAP 0.676, NDCG 0.756

## experiment simple
* git commit
** bandwidth = [64, 16, 10] ==> npixel [16384, 32768(1024), 8000(400)]
** features = [100, 100, n_classes]
** batch norm 3D = yes
** num epoch = 300
** batch_size = 32
** activation = relu
** no regularization
** no fully connected: so3 integration
** learning rate = 0.5
** frequential space, convolution ???  kernel, grid is a ring around equator of size (2*bandwidth, 1) ==> whole space?
* nparams = ~400k
* no augmentation, random translation and rotation