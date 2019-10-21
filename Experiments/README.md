# Experiments
The different experiments

## Reproduce the experiments
### SHREC'17
`python preprocessing.py 32 3 ../../data/shrec17/ deepsphere_rot False`
or
`python preprocessing.py 64 3 ../../data/shrec17/ equiangular False`

then

`python run_experiment_healpix.py 32 3 deepsphere_rot 6`
or
`python run_experiment_equiangular.py`

### ModelNet40
```sh
python preprocessing.py 32 3 ../../data/modelnet40/ deepsphere_rot False
python run_experiment.py
```

### GHCN
```sh
python run_experiment.py 2010 2015 ../../data/ghcn/ True False
```

### Cosmo
```sh
python experiment_deepsphere.py FCN 3 1 3.5 True 8
```

### Climate
```sh
python run_experiment_ico.py
```

## Data exploration
Run the jupyter notebook named Sandbbox to play with the data and the architecture of the network.

