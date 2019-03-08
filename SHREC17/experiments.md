# Deepsphere model
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
* nparams = weights (500+50000+27500+580800) + bias (100 + 100 + 55) = ~600k
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (65.82, 64.05, 1.86+e03)
* test on val_perturbed dataset: P@N 0.154, R@N 0.159, F1@N 0.155, mAP 0.144, NDCG 0.214
* test on test_perturbed dataset: P@N 0.520, R@N 0.566, F1@N 0.527, mAP 0.483, NDCG 0.530
* time per batch: N/A

## experiment 2
* git commit: 
* similar parameters as Cohen simple:
** nsides = [Nside, Nside//4, Nside//8] ==> npixel [12'228, 768, 192]
** params['F'] = [100, 100]
** params['M'] = [nclasses]
** params['batch_norm'] = [True] * 2
** params['num_epochs'] = 16
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 1
** params['dropout'] = 1
** params['scheduler'] = lambda step: tf.train.exponential_decay(2e-2, step, decay_steps=5, decay_rate=0.999)   # peut être changer
** params['K'] = sqrt(3) * nsides  Pas sur de ça, mais pas encore trouvé moyen de faire un truc similaire
** average pooling before fully-connected
* nparams = weights (33600+130000+5500) + bias (100 + 100) = ~170k
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (71.12, 68.18, 1.23)
* test on val_perturbed dataset: P@N 0., R@N 0., F1@N 0., mAP 0., NDCG 0.
* test on test_perturbed dataset: P@N 0., R@N 0., F1@N 0., mAP 0., NDCG 0.
* time per batch: 0.75 s

## experiment 0
* git commit: 
* similar parameters as Cohen simple:
** nsides = [Nside, Nside//4, Nside//8, Nside//8] ==> npixel [12'228, 768, 192]
** params['F'] = [100, 100]
** params['batch_norm'] = [True] * 2
** params['num_epochs'] = 20
** params['batch_size'] = 32
** params['activation'] = 'relu'
** params['regularization'] = 1
** params['dropout'] = 1
** params['scheduler'] = lambda step: tf.train.exponential_decay(2e-2, step, decay_steps=5, decay_rate=0.999)   # peut être changer
** params['K'] = [5] * 2  Pas sur de ça, mais pas encore trouvé moyen de faire un truc similaire
** average pooling before fully-connected
* nparams = weights (500+50000+5500) + bias (100 + 100) = ~56k
* train on perturbed dataset, no augmentation, random translation and rotation (object not on the center of the sphere)
* accuracy, F1, loss of validation part: (, , )
* test on val_perturbed dataset: P@N 0., R@N 0., F1@N 0., mAP 0., NDCG 0.
* test on test_perturbed dataset: P@N 0., R@N 0., F1@N 0., mAP 0., NDCG 0.
* time per batch: N/A

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
* test on val_perturbed dataset: P@N 0., R@N 0., F1@N 0., mAP 0., NDCG 0.
* test on test_perturbed dataset: P@N 0., R@N 0., F1@N 0., mAP 0., NDCG 0.
* time per batch: 