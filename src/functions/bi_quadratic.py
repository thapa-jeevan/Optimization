import numpy as np


class BiQuadraticFunction:
    @staticmethod
    def f(x):
        x = x.ravel()
        fx = 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2
        return fx

    @staticmethod
    def grad(x):
        x = x.ravel()
        grad_fx = np.array([
            - 400 * (x[1] - x[0] ** 2) * x[0] - 2 * (1 - x[0]),
            200 * (x[1] - x[0] ** 2)
        ])
        return grad_fx.reshape(-1, 1)

    @staticmethod
    def hessian(x):
        x = x.ravel()
        hessian = np.array([
            [- 400 * (x[1] - 3 * x[0] ** 2) + 2, - 400 * x[0]],
            [-400 * x[0], 200]
        ])
        return hessian

    @staticmethod
    def initialize(x0=None, n=2):
        if x0:
            x0 = np.array(x0).reshape(-1, 1)
        else:
            x0 = np.random.randint(2, 20, (n, 1)) + .1
        return x0
