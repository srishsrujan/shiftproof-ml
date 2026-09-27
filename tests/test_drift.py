import numpy as np
import pandas as pd

from src.shiftproof.drift import categorical_shift, psi


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(1)
    x = rng.normal(size=1000)
    y = x.copy()
    assert psi(x, y) < 1e-6


def test_categorical_shift_is_positive_for_changed_distribution():
    a = pd.Series(["a"] * 90 + ["b"] * 10)
    b = pd.Series(["a"] * 20 + ["b"] * 80)
    assert categorical_shift(a, b) > 0.2
