from pystatdp import pystatdp
from pydp.algorithms.laplacian import BoundedMean, Max

psd = pystatdp()

def test_main_bounded_mech_few_iter():
    assert psd.main(BoundedMean, tuple((-15, 15)), tuple((0.9,)), e_iter=100, d_iter=500) == 0

def test_main_order_mech_few_iter():
    assert psd.main(Max, tuple((-15, 15)), tuple((0.9,)), e_iter=100, d_iter=500) == 0

def test_main_bounded_mech_longer_range_few_iter():
    assert psd.main(BoundedMean, tuple((-15, 15)), tuple((0.9,)), e_iter=100, d_iter=500, test_range=0.5) == 0

def test_main_order_mech_longer_range_few_iter():
    assert psd.main(Max, tuple((-15, 15)), tuple((0.9,)), e_iter=100, d_iter=500, test_range=0.5) == 0

def test_main_bounded_mech_longer_range_few_iter():
    assert psd.main(BoundedMean, tuple((-15, 15)), tuple((0.9,)), e_iter=100, d_iter=500, test_range=0.5) == 0

def test_main_bounded_mech_longer_range_more_iter():
    assert psd.main(BoundedMean, tuple((-15, 15)), tuple((0.9,)), e_iter=20000, d_iter=100000, test_range=0.5) == 0
