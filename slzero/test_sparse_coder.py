import numpy as np
from slzero import SLZeroSparseCoder


def test_fit():
    np.random.seed(215)
    dic = np.random.randn(100,  10)
    comp = np.random.randn(10, 20)
    comp[(-0.3 < comp) & (comp < 0.3)] = 0
    Y = np.dot(dic, comp)
    X = Y + np.random.randn(*Y.shape) * 1e-5
    coder = SLZeroSparseCoder(dic)
    coder.fit(X)
    np.testing.assert_almost_equal(
        comp, coder.components_, decimal=5)
