# DeepSphere: a graph-based spherical CNN

Frédérick Gusset, Martino Milani, [Michaël Defferrard][mdeff], [Nathanaël Perraudin][nath]

[nath]: https://perraudin.info
[mdeff]: http://deff.ch

The code in this repository is based on [DeepSphere](https://github.com/SwissDataScienceCenter/DeepSphere) and regroups all experiments performed in the paper ["DeepSphere: a graph-based spherical CNN"][paper]. 

[paper]: http://localhost

## Installation
[![Binder](https://mybinder.org/badge_logo.svg)][binder_lab]

&nbsp; Click the binder badge to play with the notebooks from your browser without installing anything.

[binder_lab]: http://localhost


For a local installation, follow the below instructions.

1. Clone this repository.
   ```sh
   git clone https://github.com/Droxef/deepsphere_v2_code.git
   cd deepsphere_v2_code
   ```

2. Install the dependencies.
   ```sh
   conda env create -f environment.yml
   ```

   **Note**: the code has been developed and tested with Python 3.5.

## Experiments

The different benchmarks are regrouped in the [Experiment](Experiments) folder,
and each has at least a run_experiment script to rerun the experiment and reproduce the results in the report,
and a sandbox notebook to explore the data and hyperparameters.

1. SHREC17

2. ModelNet40
    
3. GHCN

4. Climate

### Reproducing the results
Follow the below steps to reproduce the paper's results. The steps are essantially the same for each experiment, and additional instructions are present in the [data](data/README.md) and [experiments](Experiments/README.md) README.
1. Download the dataset
Run the download script

```python data/{experiment_name}/download_dataset.py```

2. Preprocess the dataset (if necessary)

```python Experiments/{experiment_name}/preprocessing.py```

3. Run the experiment

```python Experiments/{experiment_name}/run_experiment.py```


## Notebooks
Various notebooks are grouped in the [Notebooks](Notebooks) folder, such as code for the proof of the theorem and tests with different sampling scheme.


## License & co

The content of this repository is released under the terms of the [MIT license](LICENSE.txt).

The code, based on the original [TensorFlow implementation of DeepSphere for cosmology](https://github.com/deepsphere/deepsphere-cosmo-tf1), was mostly developed during the master theses of Frédérick Gusset ([code][frédérick_code], [report][frédérick_report], [slides][frédérick_slides]) and Martino Milani ([code][martino_code], [report][martino_report], [slides][martino_slides], [poster][martino_poster]).

[martino_code]: https://github.com/MartMilani/PDM
[martino_report]: https://infoscience.epfl.ch/record/268192/files/Graph%20Laplacians%20on%20the%20Sphere%20for%20Rotation%20Equivariant%20Neural%20Networks.pdf
[martino_slides]: https://infoscience.epfl.ch/record/268192/files/Presentation.pdf
[martino_poster]: https://infoscience.epfl.ch/record/268192/files/Poster.pdf
[frédérick_code]: https://github.com/Droxef/PDMdeepsphere
[frédérick_report]: https://infoscience.epfl.ch/record/267531/files/Spherical%20Convolutional%20Neural%20Networks.pdf
[frédérick_slides]: https://infoscience.epfl.ch/record/267531/files/Final%20Presentation.pdf

Please consider citing our papers if you find this repository useful.

```
@inproceedings{deepsphere_iclr,
  title = {{DeepSphere}: a graph-based spherical {CNN}},
  author = {Defferrard, Michaël and Milani, Martino and Gusset, Frédérick and Perraudin, Nathanaël},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2020},
  url = {https://openreview.net/forum?id=B1e3OlStPB},
}
```

```
@inproceedings{deepsphere_rlgm,
  title = {{DeepSphere}: towards an equivariant graph-based spherical {CNN}},
  author = {Defferrard, Micha\"el and Perraudin, Nathana\"el and Kacprzak, Tomasz and Sgier, Raphael},
  booktitle = {ICLR Workshop on Representation Learning on Graphs and Manifolds},
  year = {2019},
  archiveprefix = {arXiv},
  eprint = {1904.05146},
  url = {https://arxiv.org/abs/1904.05146},
}
```
