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
import argparse
import json
import random
import numpy as np

from sklearn.datasets import load_iris
from jsonpickle import encode
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from keras.utils import to_categorical

sys.path.append(os.path.abspath('../../PyStatDP/'))
from pystatdp import pystatdp
from pystatdp.generators import ML_DIFFER

sys.path.append(os.path.abspath('../../initial_hypothesis/iris'))
from iris import PredictIris, absolute_model_path
from data import likely_misclassified_points


# We need to enclose in a main() to avoid freeze_support runtime error.
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description='Experiment outlier D and D\' models')
    parser.add_argument('--plot_all', action='store_true',
                        help='whether or not to plot all figures')
    parser.add_argument('--privacy', type=float, default=4.0, help='claimed privacy budget')
    parser.add_argument('--test_range', type=float, default=0.5, help='test range')
    parser.add_argument('--n_checks', type=int, default=3, help='number of tests to run')
    parser.add_argument('--batch_number', type=int, default=1, help='trained model batch number')
    parser.add_argument('--threshold', type=float, default=0.9, help='threshold to determine measured epsilon')
    parser.add_argument('--e_iter', type=int, default=500, help='number of iterations to run event selection')
    parser.add_argument('--d_iter', type=int, default=500, help='number of iterations to run detection algorithm')
    args = parser.parse_args()

    psd = pystatdp()

    # Load data
    iris = load_iris()
    x_data = iris.data
    y_data = iris.target
    y_data = to_categorical(y_data)
    indices = np.arange(len(x_data)) 

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(x_data, y_data, indices, test_size=0.2, random_state=42)

    # Check that indices line up
    assert(np.all((x_train == np.take(x_data, idx_train, axis=0))))

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train)), batch_size=8, shuffle=True, num_workers=0, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test)), batch_size=8, num_workers=0, drop_last=True)

    print("test_loader length: {}".format(len(test_loader.dataset)))

    # Create experiment directory.
    experiment_path = '/homes/al5217/private-pipelines/experiments/iris/'
    experiment_number = len(os.listdir(experiment_path))
    print("experiment_number: {}".format(experiment_number))
    try:
        os.makedirs('{}experiment-{}'.format(experiment_path, experiment_number))
        print("Created directory.")
    except FileExistsError:
        print('error creating file :( current path: {}'.format(Path.cwd()))
        pass

    # Run algorithm.
    results = psd.main(PredictIris, tuple((test_loader, args.batch_number)), tuple((args.privacy,)), e_iter=args.e_iter, d_iter=args.d_iter, test_range=args.test_range, n_checks=args.n_checks, sensitivity=ML_DIFFER)

    if args.plot_all:
        # plot and save to file
        plot_file = "{}experiment-{}/test_result.pdf".format(experiment_path, experiment_number)

        psd.plot_result(results, r'Test $\epsilon$', 'P Value', 'PredictIris', plot_file)

    # Get measured epsilon and format results.
    # results[test_budget] = [(epsilon, p, d1, d2, kwargs, event)]
    results = results[args.privacy]
    measured_epsilon = -1.0
    p_values = []
    for (epsilon, p, d1, d2, kwargs, event) in reversed(results):
        p_values.append({'epsilon': epsilon, 'p_value': p})
        if p >= args.threshold:
            measured_epsilon = epsilon

    training_info = ""
    with open(absolute_model_path + '/batch-' + str(args.batch_number) + '/training_info.json') as json_file:
        training_info = json.load(json_file)

    results_json = {
        'experiment_args': vars(args),
        'training_info': training_info,
        'measured_epsilon': measured_epsilon,
        'batch_number': args.batch_number,
        'p_values': p_values,
        'results': encode(results, unpicklable=False)
    }

    # dump the results to file
    json_file = Path.cwd() / f'experiment-{experiment_number}/test_results.json'
    with json_file.open('w') as f:
        json.dump(results_json, f, indent="  ")