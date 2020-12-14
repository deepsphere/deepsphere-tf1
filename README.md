# DeepSphere: a graph-based spherical CNN

Frédérick Gusset, Martino Milani, [Michaël Defferrard][mdeff], [Nathanaël Perraudin][nath]

[nath]: https://perraudin.info
[mdeff]: http://deff.ch

The code in this repository is based on [DeepSphere](https://github.com/SwissDataScienceCenter/DeepSphere) and regroups all experiments performed in the paper ["DeepSphere: a graph-based spherical CNN"][paper]. 

[paper]: http://localhost

## Installation

For a local installation, follow the below instructions.

1. Clone this repository.
    ```sh
    git clone https://github.com/deepsphere/deepsphere-tf1.git
    cd deepsphere-tf1
    ```

2. Install the dependencies.
    ```sh
    conda env create -f environment.yml
    ```

## Reproducing the results of the paper

Each experiment has a folder in [data](data) with scripts to download the data, and a folder in [experiments](experiments) with scripts to preprocess and reproduce the experiment.
Additionally, there is a notebook to explore the data and hyper-parameters.

The steps are essentially the same for each experiment (`climate`, `cosmo`, `ghcn`, `modelnet40`, `shrec17`), as follows:

```sh
python data/{experiment_name}/download_dataset.py
python experiments/{experiment_name}/preprocessing.py
python experiments/{experiment_name}/run_experiment.py
```

Additional instructions are contained in the [data](data) and [experiments](experiments) READMEs.

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
