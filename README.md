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

The content of this repository is released under the terms of the [MIT license](LICENCE.txt).
