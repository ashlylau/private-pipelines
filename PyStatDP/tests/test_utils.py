from pystatdp.utils import arr_n_check

def test_negative_privacy():
    assert arr_n_check(-0.9, 0.5, 3) == (0.4,0.9,1.4)

def test_callable():
    assert callable(arr_n_check)

def test_inappropriats_n_checks():
    assert arr_n_check(0.9, 0.5, -3) == (0.4,0.9,1.4)

def test_inappropriats_n_checks1():
    assert arr_n_check(0.9, 0.5, 4) == (0.4,0.9,1.4)

def test_very_large_test_range():
    assert arr_n_check(0.9, 1, 5) == (0.9,)
