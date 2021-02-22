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
import pytest
from pydp.algorithms.laplacian import BoundedMean
from pystatdp.algorithms import generic_method_pydp
from pystatdp.selectors import select_event


@pytest.mark.parametrize('process_pool', (mp.Pool(1), mp.Pool()), ids=('SingleCore', 'MultiCore'))
def test_select_event(process_pool):
    with process_pool:
        d1 = [0] + [2 for _ in range(4)]
        d2 = [1 for _ in range(5)]
        kwargs = {'algorithm': BoundedMean,
                'param_for_algorithm': tuple((-15, 15)),
                'privacy': 0.9}
        _, _, _, event = select_event(generic_method_pydp, ((d1, d2, kwargs),), 0.9, 1000, process_pool)
        assert event[0][0] < event[0][1]
