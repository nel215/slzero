import numpy as np


def solve(A, x):
    sigma_min = 1e-6
    L = 5
    max_iter = 10
    c = 0.75
    mu = 2
    A_inv = np.linalg.pinv(A)
    s_hat = np.dot(A_inv, x)
    sigma = 4.0*np.max(np.abs(s_hat), axis=1)
    sigma = sigma[:, np.newaxis]

    for i in range(max_iter):
        s = s_hat
        for l in range(L):
            delta = s * np.exp(-np.power(s, 2) / 2 / np.power(sigma, 2))
            s = s - mu * delta
            rhs = np.dot(A, s) - x
            s = s - np.dot(A_inv, rhs)
        s_hat = s
        sigma *= c
        if np.all(sigma < sigma_min):
            break

    return s_hat
