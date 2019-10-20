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
   conda env create -f environment_conda.yml
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


## License & co

The content of this repository is released under the terms of the [MIT license](LICENCE.txt).
