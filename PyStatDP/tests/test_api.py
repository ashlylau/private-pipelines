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
import logging
import pytest
from flaky import flaky
from pystatdp.algorithms import generic_method_pydp
from pystatdp import pystatdp,  ALL_DIFFER, ONE_DIFFER
from pydp.algorithms.laplacian import BoundedMean, BoundedStandardDeviation, BoundedSum,\
                                    BoundedVariance, Max, Min, Median,Percentile

correct_algorithms = (
    (generic_method_pydp, {'algorithm': BoundedMean, 'param_for_algorithm': (-15,15), 'privacy': 0.7}, 5, ALL_DIFFER),
    (generic_method_pydp, {'algorithm': BoundedStandardDeviation, 'param_for_algorithm': (-15,15), 'privacy': 0.7}, 5, ALL_DIFFER),
    (generic_method_pydp, {'algorithm': BoundedSum, 'param_for_algorithm': (-15,15), 'privacy': 0.7}, 5, ALL_DIFFER),
    (generic_method_pydp, {'algorithm': BoundedVariance, 'param_for_algorithm': (-15,15), 'privacy': 0.7}, 5, ALL_DIFFER),
    (generic_method_pydp, {'algorithm': Max, 'param_for_algorithm': (-15,15), 'privacy': 0.7}, 5, ALL_DIFFER),
    (generic_method_pydp, {'algorithm': Min, 'param_for_algorithm': (-15,15), 'privacy': 0.7}, 5, ALL_DIFFER),
    (generic_method_pydp, {'algorithm': Median, 'param_for_algorithm': (-15,15), 'privacy': 0.7}, 5, ALL_DIFFER),
    (generic_method_pydp, {'algorithm': Percentile, 'param_for_algorithm': (0.75, -15, 15), 'privacy': 0.7}, 5, ALL_DIFFER)
)

psd = pystatdp()

@pytest.mark.parametrize('algorithm', correct_algorithms, ids=[algorithm[0].__name__ for algorithm in correct_algorithms])
# due to the statistical and randomized nature, use flaky to allow maximum 5 runs of failures
@flaky(max_runs=5)
def test_correct_algorithm(algorithm):
    func, kwargs, num_input, sensitivity = algorithm
    result = psd.detect_counterexample(func, (0.6, 0.7, 0.8), kwargs, num_input=num_input,
                event_iterations=100, detect_iterations=500, loglevel=logging.DEBUG, sensitivity=sensitivity)
    assert isinstance(result, list) and len(result) == 3
    epsilon, p, *extras = result[0]
    assert p <= 0.05, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(
        epsilon, p, extras)
    epsilon, p, *extras = result[1]
    assert p >= 0.05, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(
        epsilon, p, extras)
    epsilon, p, *extras = result[2]
    assert p >= 0.95, 'epsilon: {}, p-value: {} is not expected. extra info: {}'.format(
        epsilon, p, extras)
