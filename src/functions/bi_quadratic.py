import numpy as np


def f3(x):
    x = x.ravel()
    fx = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
    return fx


def grad_f3(x):
    x = x.ravel()
    grad_fx = np.array([
        - 400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]),
        200 * (x[1] - x[0] ** 2)
    ])
    return grad_fx.reshape(-1, 1)


def hessian_f3(x):
    x = x.ravel()
    hessian = np.array([
        [- 400 * (x[1] - 3 * x[0] ** 2) * 2 * x[0] + 2, - 400 * x[0]],
        [-400 * x[0], 200]
    ])
    return hessian
