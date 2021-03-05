## Privacy Guarantees of Machine Learning Pipelines

### Install
For the best performance we recommend installing `private-pipelines` in a `conda` virtual environment (or `venv` if you prefer; the setup is similar):

```bash
# we use python 3.7 because conda doesn't support keras for python > 3.7, but 3.6 and above should work fine
conda create -n private-pipelines anaconda python=3.7
conda activate private-pipelines
# install dependencies from conda for best performance
conda install jsonpickle matplotlib numpy numba pip scipy sympy tqdm 
# install icc_rt compiler for best performance with numba, this requires using intel's channel
conda install -c intel icc_rt
# install the remaining non-conda dependencies and pystatdp
pip3 install python-dp opacus 
```
Then you can run `experiments/demo.py` to run the experiments we conducted.