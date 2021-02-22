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
import multiprocessing as mp
from numpy.testing import assert_almost_equal
import pytest
from pydp.algorithms.laplacian import BoundedMean
from pystatdp.algorithms import generic_method_pydp
from pystatdp.hypotest import hypothesis_test, statistics_test as pystatdp_statistics_test


@pytest.mark.parametrize('process_pool', (mp.Pool(1), mp.Pool()), ids=('SingleCore', 'MultiCore'))
def test_hypothesis_test(process_pool):
    with process_pool:
        kwargs = {'algorithm': BoundedMean,
                'param_for_algorithm': tuple((-15, 15)),
                'privacy': 0.9}
        d1, d2 = [0] + [2 for _ in range(4)], [1 for _ in range(5)]
        event = (0,)

        # TODO; returns p1 = p2 = 1 for privacy_budget = 0.25
        # kwargs['privacy'] = 0.25
        # p1, p2 = hypothesis_test(generic_method_pydp, d1, d2, kwargs, event, 0.25, 1000, process_pool)
        # print(p1, p2)
        # assert 0 <= p1 <= 0.05
        # assert 0.95 <= p2 <= 1.0

        kwargs['privacy'] = 0.5
        p1, p2 = hypothesis_test(generic_method_pydp, d1, d2, kwargs, event, 0.5, 1000, process_pool)
        print(p1, p2)
        assert 0.05 <= p1 <= 1.0
        assert 0.95 <= p2 <= 1.0

        kwargs['privacy'] = 0.75
        p1, p2 = hypothesis_test(generic_method_pydp, d1, d2, kwargs, event, 0.75, 1000, process_pool)
        print(p1, p2)
        assert 0.95 <= p1 <= 1.0
        assert 0.95 <= p2 <= 1.0


def test_statistics_test():
    # test both JIT'ed version and original python version of statistics_test function
    for func in (pystatdp_statistics_test, pystatdp_statistics_test.py_func):
        assert_almost_equal(func(1000, 1000, 1, 2000), 1)
        assert_almost_equal(func(1999, 1, 1, 2000), 0)
