# MIT License
#
# Copyright (c) 2018-2019 Yuxin Wang
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
from functools import partial
import multiprocessing as mp

import numpy as np
from numba import njit

from pystatdp.core import run_algorithm
import pystatdp._hypergeom as hypergeom


@njit
def statistics_test(cx, cy, epsilon, iterations):
    """ Calculate p-value based on observed results.
    :param cx: The observed count of running algorithm with database 1 that falls into the event
    :param cy:The observed count of running algorithm with database 2 that falls into the event
    :param epsilon: The epsilon to test for.
    :param iterations: The total iterations for running algorithm.
    :return: p-value
    """
    # average p value
    sample_num = 200
    p_value = 0
    for new_cx in np.random.binomial(cx, 1.0 / (np.exp(epsilon)), sample_num):
        p_value += hypergeom.sf(new_cx - 1, 2 * iterations,
                                iterations, new_cx + cy)
    return p_value / sample_num


def hypothesis_test(algorithm, d1, d2, kwargs, event, epsilon, iterations, process_pool, report_p2=True):
    """ Run hypothesis tests on given input and events.
    :param algorithm: The algorithm to run on.
    :param kwargs: The keyword arguments the algorithm needs.
    :param d1: Database 1.
    :param d2: Database 2.
    :param event: The event set.
    :param iterations: Number of iterations to run.
    :param epsilon: The epsilon value to test for.
    :param process_pool: The multiprocessing.Pool() to use.
    :param report_p2: The boolean to whether report p2 or not.
    :return: p values.
    """
    # use undocumented mp.Pool._processes to get the number of max processes for the pool, this is unstable and
    # may break in the future, therefore we fall back to mp.cpu_count() if it is not accessible
    core_count = process_pool._processes if process_pool._processes and isinstance(process_pool._processes, int) \
        else mp.cpu_count()
    process_iterations = np.full(core_count, int(
        np.floor(float(iterations) / core_count)), dtype=int)
    # process_iterations = [int(np.floor(float(iterations) / core_count)) for _ in range(core_count)]
    # add the remaining iterations to the last index
    process_iterations[core_count -
                       1] += iterations % process_iterations[core_count - 1]

    # start the pool to run the algorithm and collects the statistics
    cx, cy = 0, 0
    # fill in other arguments for running the algorithm, leaving `iterations` to be filled
    runner = partial(run_algorithm, algorithm, d1, d2, kwargs, event)
    for ((local_cx, local_cy), *_), _ in process_pool.imap_unordered(runner, process_iterations):
        cx += local_cx
        cy += local_cy
    cx, cy = (cx, cy) if cx > cy else (cy, cx)

    # calculate and return p value
    if report_p2:
        return statistics_test(cx, cy, epsilon, iterations), statistics_test(cy, cx, epsilon, iterations)
    else:
        return statistics_test(cx, cy, epsilon, iterations)
