import numpy as np


def f1(x):
    n, _ = x.shape
    fx = (np.arange(1, n + 1) * (x ** 2)).sum()
    return fx


def grad_f1(x):
    n, _ = x.shape
    grad_fx = 2 * np.arange(1, n + 1).reshape(n, 1) * x
    return grad_fx


def hessian_f1(x):
    n, _ = x.shape
    hessian_fx = np.diag(2 * np.arange(1, n + 1))
    return hessian_fx
