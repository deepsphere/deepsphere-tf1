# DeepSphere: a graph-based spherical CNN

[Michaël Defferrard](https://deff.ch),
[Martino Milani](https://www.linkedin.com/in/martino-milani-11258350),
[Frédérick Gusset](https://www.linkedin.com/in/frédérick-gusset-a42485191),
[Nathanaël Perraudin](https://perraudin.info)

The code in this repository implements a generalization of Convolutional Neural Networks (CNNs) to the sphere.
We here model the discretized sphere as a graph of connected pixels.
The resulting convolution is efficient (especially when data doesn't span the whole sphere) and mostly equivariant to rotation (small distortions are due to the non-existence of a regular sampling of the sphere).
The pooling strategy exploits hierarchical pixelizations of the sphere to analyze the data at multiple scales.
The performance of DeepSphere is demonstrated on four problems: the recognition of 3D objects, the discrimination of cosmological models, the segmentation of extreme events in climate simulations, and the identification of trends in historical weather.

## Resources

Code:
* [deepsphere-cosmo-tf1](https://github.com/deepsphere/deepsphere-cosmo-tf1): original repository, implemented in TensorFlow v1. Use to reproduce [`arxiv:1810.12186`][paper_cosmo].
* [deepsphere-cosmo-tf2](https://github.com/deepsphere/deepsphere-cosmo-tf2): reimplementation in TFv2. Use for new developments in TensorFlow targeting HEALPix.
* [deepsphere-tf1](https://github.com/deepsphere/deepsphere-tf1): extended to other samplings and experiments, implemented in TFv1. Use to reproduce [`arxiv:2012.15000`][paper_iclr].
* [deepsphere-pytorch](https://github.com/deepsphere/deepsphere-pytorch): reimplementation in PyTorch. Use for new developments in PyTorch.

Papers:
* DeepSphere: Efficient spherical CNN with HEALPix sampling for cosmological applications, 2018.\
  [[paper][paper_cosmo], [blog](https://datascience.ch/deepsphere-a-neural-network-architecture-for-spherical-data), [slides](https://doi.org/10.5281/zenodo.3243380)]
* DeepSphere: towards an equivariant graph-based spherical CNN, 2019.\
  [[paper][paper_rlgm], [poster](https://doi.org/10.5281/zenodo.2839355)]
* DeepSphere: a graph-based spherical CNN, 2020.\
  [[paper][paper_iclr], [slides](https://doi.org/10.5281/zenodo.3777976), [video](https://youtu.be/NC_XLbbCevk)]

[paper_cosmo]: https://arxiv.org/abs/1810.12186
[paper_rlgm]: https://arxiv.org/abs/1904.05146
[paper_iclr]: https://arxiv.org/abs/2012.15000

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

The below notebooks contain some experiments.

1. [From sampled spheres to graphs.][sphere_to_graph]
   Experiments showing how to build a graph from a sampled sphere.

   **Note:** The tested and recommended implementations are now available in the [PyGSP].

1. [Irregular pooling.][irregular_pooling]
   Experiments with pooling on non-uniformly and partially sampled spheres.

Explanatory notebooks about the method are available [here](https://github.com/deepsphere/deepsphere-cosmo-tf1/#notebooks) and [there](https://github.com/deepsphere/deepsphere-cosmo-tf2/#notebooks).

[sphere_to_graph]: https://nbviewer.jupyter.org/github/deepsphere/deepsphere-tf1/blob/master/notebooks/sphere_to_graph.ipynb
[irregular_pooling]: https://nbviewer.jupyter.org/github/deepsphere/deepsphere-tf1/blob/master/notebooks/irregular_pooling.ipynb
[PyGSP]: https://github.com/epfl-lts2/pygsp

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
