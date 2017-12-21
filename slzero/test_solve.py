import numpy as np
from slzero import solve


def test_solve():
    np.random.seed(215)
    A = np.random.randn(100,  10)
    s = np.random.randn(10, 20)
    s[(-0.3 < s) & (s < 0.3)] = 0
    y = np.dot(A, s)
    x = y + np.random.randn(*y.shape) * 1e-5
    s_hat = solve(A, x)
    np.testing.assert_almost_equal(s, s_hat, decimal=5)
