import numpy as np

from src.shiftproof.calibration import PlattCalibrator, choose_abstention_threshold, confidence_from_probability


def test_calibrator_preserves_shape_and_range():
    p = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    y = np.array([0, 0, 0, 1, 1])
    c = PlattCalibrator().fit(p, y)
    out = c.transform(p)
    assert out.shape == p.shape
    assert np.all(out > 0)
    assert np.all(out < 1)


def test_confidence_is_at_least_half():
    c = confidence_from_probability(np.array([0.1, 0.4, 0.6, 0.9]))
    assert np.all(c >= 0.5)


def test_abstention_threshold_is_reasonable():
    p = np.linspace(0.02, 0.98, 100)
    y = (p > 0.6).astype(int)
    threshold = choose_abstention_threshold(y, p)
    assert 0.5 <= threshold <= 0.99
