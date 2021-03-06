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
import enum
import numpy as np
from logging import getLogger

sys.path.append(os.path.abspath('../../initial_hypothesis/iris'))
from iris import absolute_model_path
iris_model_path = absolute_model_path

sys.path.append(os.path.abspath('../../initial_hypothesis/adult'))
from adult import absolute_model_path
adult_model_path = absolute_model_path

logger = getLogger(__name__)


class Sensitivity(enum.Enum):
    ALL_DIFFER = 0
    ONE_DIFFER = 1
    ML_DIFFER = 2


ALL_DIFFER = Sensitivity.ALL_DIFFER
ONE_DIFFER = Sensitivity.ONE_DIFFER
ML_DIFFER = Sensitivity.ML_DIFFER


def generate_arguments(algorithm, d1, d2, default_kwargs):
    """
    :param algorithm: The algorithm to test for.
    :param d1: The database 1
    :param d2: The database 2
    :param default_kwargs: The default arguments that are given or have a default value.
    :return: Extra argument needed for the algorithm besides Q and epsilon.
    """
    arguments = algorithm.__code__.co_varnames[:algorithm.__code__.co_argcount]
    if arguments[1] not in default_kwargs:
        logger.error(
            f'The third argument {arguments[2]} (privacy budget) is not provided!')
        return None

    return default_kwargs


def generate_databases(algorithm, num_input, default_kwargs, sensitivity=ALL_DIFFER):
    """
    :param algorithm: The algorithm to test for.
    :param num_input: The number of inputs to be generated
    :param default_kwargs: The default arguments that are given or have a default value.
    :param sensitivity: The sensitivity setting, all queries can differ by one or just one query can differ by one.
    :return: List of (d1, d2, args) with length num_input
    """
    if not isinstance(sensitivity, Sensitivity):
        raise ValueError(
            'sensitivity must be pystatdp.ALL_DIFFER or pystatdp.ONE_DIFFER or pystatdp.ML_DIFFER')
    if sensitivity == ML_DIFFER:
        d1 = -1  # This will be the model trained with the full dataset.

        algo_name = str(default_kwargs['algorithm'])[8:-2]
        if algo_name == "adult.PredictAdult":
            absolute_model_path = adult_model_path
        else:
            absolute_model_path = iris_model_path
        print("model path: {}".format(absolute_model_path))

        # Get valid models from given batch.
        _, batch_number = default_kwargs["param_for_algorithm"]
        model_names = os.listdir(absolute_model_path + "/batch-" + str(batch_number))
        model_indices = []
        for model_name in model_names:
            if model_name == 'training_info.json':
                continue
            model_index = int(str(model_name)[6:])
            if model_index != -1:
                model_indices.append(model_index)
        
        # Tuples will represent model number pairs to test.
        candidates = [([d1], [d2]) for d2 in model_indices]
    else:
        # assume maximum distance is 1
        d1 = np.ones(num_input, dtype=int)
        candidates = [
            (d1, np.concatenate((np.array([0]), d1[1:]), axis=0)),  # one below
            (d1, np.concatenate((np.array([2]), d1[1:]), axis=0)),  # one above
        ]

        if sensitivity == ALL_DIFFER:
            dzero = np.zeros(num_input, dtype=int)
            dtwo = np.full(num_input, 2, dtype=int)
            candidates.extend([
                # one above rest below
                (d1, np.concatenate((np.array([2]), dzero[1:]), axis=0)),
                # one below rest above
                (d1, np.concatenate((np.array([0]), dtwo[1:]), axis=0)),
                # half half
                (d1, np.concatenate((dtwo[:int(num_input/2.0) + 1], dzero[:num_input - int(num_input / 2.0) + 1]), axis=0)),  # [0 for _ in range(num_input - int(num_input / 2))]),
                # all above
                (d1, dtwo),
                # all below
                (d1, dzero),
                # x shape
                (np.concatenate((d1[:int(np.floor(num_input / 2.0))+1], dzero[:int(np.ceil(num_input / 2.0))+1]), axis=0),
                np.concatenate((dzero[:int(np.floor(num_input / 2.0))+1], d1[:int(np.ceil(num_input / 2.0))+1]), axis=0))
            ])

    return tuple((d1, d2, generate_arguments(algorithm, d1, d2, default_kwargs)) for d1, d2 in candidates)