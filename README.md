# An empirical study of spherical convolutional neural networks

Frédérick Gusset, [Nathanaël Perraudin][nath], [Michaël Defferrard][mdeff]

[nath]: https://perraudin.info
[mdeff]: http://deff.ch

The code in this repository is based on [DeepSphere](https://github.com/SwissDataScienceCenter/DeepSphere) and regroups all experiments performed in the master thesis ["An empirical study of spherical convolutional neural networks"][thesis]. This master project was performed in the LTS2 lab at EPFL, during Spring semester 2019.

[thesis]: https://infoscience.epfl.ch/record/267531?&ln=fr

## Installation

For a local installation, follow the below instructions.

1. Clone this repository.
   ```sh
   git clone https://github.com/Droxef/PDMdeepsphere.git
   cd PDMdeepSphere
   ```

2. Install the dependencies.
   ```sh
   pip install -r requirements.txt
   ```

   **Note**: if you will be working with a GPU, comment the
   `tensorflow==1.6.0` line in `requirements.txt` and uncomment the
   `tensorflow-gpu==1.6.0` line.

   **Note**: the code has been developed and tested with Python 3.5.

3. Play with the Jupyter notebooks.
   ```sh
   jupyter notebook
   ```

## Experiments

The different benchmarks are regrouped in the [Experiment](Experiments) folder, and each has at least one notebook to rerun the experiment and reproduce the results in the report.

1. SHREC17
    * [demo_sphere_SHREC17][cached data]
      Shrec17 experiment with TF dataset pipeline
    * [demo_sphere_SHREC17_equiangular][equiangular]
      SHREC17 experiment using an equiangular sampling similar as [Cohen et al.](https://arxiv.org/abs/1801.10130)

2. ModelNet40
    * [demo](Experiments/ModelNet40/demo_sphere_ModelNet40.ipynb)
     MN40 experiment
    * [analyze rotation](Experiments/ModelNet40/Sphere_ModelNet40_rotation.ipynb)
    Analyze the behaviour when adding different rotation perturbations
    
3. GHCN
    * [test](Experiments/GHCN/sphere_GHCN_test.ipynb)
    Analyze of the dataset
    * [demo](Experiments/GHCN/sphere_GHCN.ipynb)
    GHCN diffent taks

4. Climate

5. Graphs
    * [equiangular_and_other_graphs](Experiments/Graphs/equiangular_and_other_graphs.ipynb)
    Construct an equiangular graph and analyze its properties

6. Irregular pooling
    * [Irregular_pooling](Experiments/Irregular_pooling/Irregular_pooling.ipynb) 
    Find ways to use pooling on random part of sphere

[cached data]: (Experiments/SHREC17/demo_sphere_SHREC17-Cached_data.ipynb)
[equiangular]: (Experiments/SHREC17/demo_sphere_SHREC17_equiangular.ipynb)


## License & co

The content of this repository is released under the terms of the [MIT license](LICENCE.txt).
