# Experiments

Directory grouping scripts to reproduce our experiments.

## Data exploration

In each experiment's folder, you can run the jupyter notebook named `Sandbox_*.ipynb` to play with the data and the network architecture.

## Reproduce the experiments

### SHREC'17

```sh
python preprocessing.py 32 3 ../../data/shrec17/ deepsphere_rot False
python run_experiment_healpix.py 32 3 deepsphere_rot 6
```

```sh
python preprocessing.py 64 3 ../../data/shrec17/ equiangular False
python run_experiment_equiangular.py
```

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
