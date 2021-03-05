# MIT License
#
# Copyright (c) 2018 Yuxin Wang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import torch 
import random
from sklearn.datasets import load_iris

sys.path.append(os.path.abspath('../PyStatDP/'))
from pystatdp import pystatdp
from pystatdp.generators import ML_DIFFER

sys.path.append(os.path.abspath('../initial_hypothesis/iris'))
from iris import PredictIris
from data import likely_misclassified_points, outlier_indices


# We need to enclose in a main() to avoid freeze_support runtime error.
if __name__ == "__main__": 
    psd = pystatdp()

    # Load data
    iris = load_iris()
    x_data = iris.data
    rand_point = random.choice(outlier_indices)
    x_test = x_data[rand_point]
    print("x_test (element {}): {}".format(rand_point, x_test))

    # def main(self, algo, param, privacy, e_iter=100000, d_iter=500000, test_range=0.1, n_checks=3):
    psd.main(PredictIris, torch.Tensor(x_test), tuple((2.0,)), e_iter=1000, d_iter=5000, test_range=0.5, n_checks=8, sensitivity=ML_DIFFER)
