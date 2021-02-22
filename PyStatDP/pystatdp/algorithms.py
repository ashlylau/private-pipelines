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

algo_dict = {
    'bounded_functions': {
        'BoundedMean': 'example call with parameters: dp.BoundedMean(epsilon, -15, 15)',
        'BoundedStandardDeviation': 'example call with parameters: dp.BoundedStandardDeviation(epsilon, 0, 15)',
        'BoundedSum': 'example call with parameters: dp.BoundedSum(epsilon, 0, 10)',
        'BoundedVariance': 'example call with parameters: dp.BoundedVariance(epsilon, 0, 16)'
    },
    'order_statistics': {
        'Max': 'example call with parameters: dp.Max(epsilon)',
        'Min': 'example call with parameters: dp.Min(epsilon)',
        'Median': 'example call with parameters: dp.Median(epsilon)',
        'Percentile': 'example call with parameters: dp.Percentile(epsilon)'
    }
}


def generic_method_pydp(queries, privacy, algorithm, param_for_algorithm):
    '''
    A generic method to route incoming tasks.
    param queries: queries to the algorithm
    param epsilon: privacy budget
    param algorithm: The algorithm to be tested; (e.g dp.BoundedMean, dp.BoundedSum)
    param param_to_algorithm (a tuple): inputs to the algortihm.

    queries = [1,2,3,4,5]
    print(generic_method_pydp(prng, queries, 1.0, dp.BoundedMean, (-15,15)))
    >>> example call with parameters: dp.BoundedMean(epsilon, -15, 15)
        0.0
    '''
    if type(queries) != list:
        queries = queries.tolist()
    # print(algo_dict[str(algorithm)[13:-2]])
    if str(algorithm)[13:-2] in algo_dict['order_statistics'].keys():
        return algorithm(privacy).quick_result(queries, privacy)
    else:
        return algorithm(privacy, *param_for_algorithm).quick_result(queries)
