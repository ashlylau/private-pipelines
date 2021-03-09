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
from sklearn.datasets import load_iris
from jsonpickle import encode
from pathlib import Path

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
    args = parser.parse_args()

    psd = pystatdp()

    # Load data
    iris = load_iris()
    x_data = iris.data

    measured_epsilons = []

    test_points = likely_misclassified_points[:10]
    for test_point in test_points:
        x_test = x_data[test_point]
        y = iris.target[test_point]
        print("x_test (element {}): {} (class = {})".format(test_point, x_test, y))

        # Run algorithm.
        results = psd.main(PredictIris, tuple((torch.Tensor(x_test), args.batch_number)), tuple((args.privacy,)), e_iter=500, d_iter=5000, test_range=args.test_range, n_checks=args.n_checks, sensitivity=ML_DIFFER)

        if args.plot_all:
            # plot and save to file
            plot_file = Path.cwd() / f'outlier_test_{test_point}.pdf'

            psd.plot_result(results, r'Test $\epsilon$', 'P Value', 'PredictIris', plot_file)
        
        # Get measured epsilon and format results.
        # results[test_budget] = [(epsilon, p, d1, d2, kwargs, event)]
        results = results[args.privacy]
        measured_epsilon = -1.0
        for (epsilon, p, d1, d2, kwargs, event) in reversed(results):
            if p >= args.threshold:
                measured_epsilon = epsilon

        measured_epsilons.append(measured_epsilon)

        training_info = ""
        with open(absolute_model_path + '/batch-' + str(args.batch_number) + '/training_info.json') as json_file:
            training_info = json.load(json_file)

        results_json = {
            'training_info': training_info,
            'measured_epsilon': measured_epsilon,
            'batch_number': args.batch_number,
            'x_test': {
                'index': test_point,
                'x': x_test.tolist(),
                'y': y.tolist()
            },
            'results': encode(results, unpicklable=False)
        }

        # dump the results to file
        json_file = Path.cwd() / f'outlier_test_{test_point}.json'
        with json_file.open('w') as f:
            json.dump(results_json, f, indent="  ")

    experiment_results = {
        'experiment_args': vars(args),
        'x_test': test_points,
        'measured_epsilon': measured_epsilons,
        'avg_epsilon': sum(measured_epsilons)/len(measured_epsilons),
    }
    print("x_test: {}".format(test_points))
    print("measured epsilon: {}".format(measured_epsilons))
    print("avg epsilon: {}".format(sum(measured_epsilons)/len(measured_epsilons)))
    json_file = Path.cwd() / f'outlier_test_results.json'
    with json_file.open('w') as f:
        json.dump(experiment_results, f, indent="  ")