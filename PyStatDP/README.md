# PyStatDP
This is a fork of [cmla-psu/statdp Statistical Counterexample Detector for Differential Privacy](https://github.com/cmla-psu/statdp) that is based on [this paper](https://arxiv.org/pdf/1805.10277.pdf), created to explore the possibility of integrating it into the CI workflow of projects with differentially private elements.

[![Build Status](https://travis-ci.org/OpenMined/PyStatDP.svg?branch=generic-feature)](https://travis-ci.org/OpenMined/PyStatDP)

## Usage

We assume your algorithm implementation has the folllowing signature: `(queries, epsilon, ...)` (list of queries, privacy budget and extra arguments).  

Then you can simply call the detection tool within three extra lines of code, that automatically performs the tasks of database generation and event selection.   
```python
from pystatdp import pystatdp
#import your privacy preserving algorithm
from pydp.algorithms.laplacian import BoundedMean

pystatdp = pystatdp()

#Currently, only mechanisms with the class and call structure of [PyDP](https://github.com/openmined/PyDP) are supported.
# All mechanisms of PyDP are supported.

if __name__ == '__main__':
    psd.main(BoundedMean, tuple((-15, 15)), tuple((0.9,)), e_iter=1000,
            d_iter=5000, test_range= 0.5)
```
### Arguments
```bash
algo         algorithm
param        parameters for the algorithm
privacy      privacy budget
```
### Optional arguments
```bash
e_iter       event iterations; default: 100000
d_iter       detection iterations; default: 500000
test_range   absolute difference between two consecutive `test privacy budgets`; default = 0.1
n_checks     Number of `test privacy budgets` to validate for, with mean at `privacy`; default: 3
```

### Base implementation
The `detect_counterexample` accepts multiple extra arguments to customize the process, check the signature and notes of `detect_counterexample` method to see how to use.  
The `main` method from `pystatdp` calls `detect_counterexample` internally.

```python
def detect_counterexample(algorithm, test_epsilon, default_kwargs=None, databases=None, num_input=(5, 10),
                          event_iterations=100000, detect_iterations=500000, cores=None, sensitivity=ALL_DIFFER,
                          quiet=False, loglevel=logging.INFO):
    """
    :param algorithm: The algorithm to test for.
    :param test_epsilon: The privacy budget to test for, can either be a number or a tuple/list.
    :param default_kwargs: The default arguments the algorithm needs except the first Queries argument.
    :param databases: The databases to run for detection, optional.
    :param num_input: The length of input to generate, not used if database param is specified.
    :param event_iterations: The iterations for event selector to run.
    :param detect_iterations: The iterations for detector to run.
    :param cores: The number of max processes to set for multiprocessing.Pool(), os.cpu_count() is used if None.
    :param sensitivity: The sensitivity setting, all queries can differ by one or just one query can differ by one.
    :param quiet: Do not print progress bar or messages, logs are not affected.
    :param loglevel: The loglevel for logging package.
    :return: [(epsilon, p, d1, d2, kwargs, event)] The epsilon-p pairs along with databases/arguments/selected event.
    """
```

## Install
For the best performance we recommend installing `pystatdp` in a `conda` virtual environment (or `venv` if you prefer; the setup is similar):

```bash
# we use python 3.8, but 3.6 and above should work fine
conda create -n pystatdp anaconda python=3.8
conda activate pystatdp
# install dependencies from conda for best performance
conda install jsonpickle matplotlib numpy numba pip scipy sympy tqdm
# install icc_rt compiler for best performance with numba, this requires using intel's channel
conda install -c intel icc_rt
# install the remaining non-conda dependencies and pystatdp
pip install .
```
Then you can run `examples/benchmark.py` to run the experiments we conducted.


## Visualizing the results
A nice python library `matplotlib` is recommended for visualizing your result.

There's a python code snippet within class `pystatdp`(`plot_result` method) to show an example of plotting the results.

Then you can generate a figure like the BoundedMean method of PyDP (see [here](https://github.com/OpenMined/PyStatDP/blob/master/examples/generic_method.pdf).)   

![iSVT4](https://raw.githubusercontent.com/yxwangcs/StatDP/master/examples/iSVT4.svg?sanitize=true)

## Customizing the detection
Our tool is designed to be modular and components are fully decoupled. You can write your own `input generator`/`event selector` and apply them to `hypothesis test`.

In general the detection process is

`test_epsilon --> generate_databases --((d1, d2, kwargs), ...), epsilon--> select_event --(d1, d2, kwargs, event), epsilon--> hypothesis_test --> (d1, d2, kwargs, event, p-value), epsilon`

You can checkout the definition and docstrings of the functions respectively to define your own generator/selector. Basically the `detect_counterexample` function in `pystatdp.core` module is just shortcut function to take care of the above process for you.

`statistics_test` function in `hypotest` module can be used universally by all algorithms (this function is to calculate p-value based on the observed statistics). However, you may need to design your own generator or selector for your own algorithm, since our input generator and event selector are designed to work with numerical queries on databases.

## Citing this work

You are encouraged to cite the orginal StatDp [paper](https://arxiv.org/pdf/1805.10277.pdf) if you use this tool for academic research:

```bibtex
@inproceedings{ding2018detecting,
  title={Detecting Violations of Differential Privacy},
  author={Ding, Zeyu and Wang, Yuxin and Wang, Guanhong and Zhang, Danfeng and Kifer, Daniel},
  booktitle={Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security},
  pages={475--489},
  year={2018},
  organization={ACM}
}
```

## License
[MIT](https://github.com/yxwangcs/statdp/blob/master/LICENSE).
