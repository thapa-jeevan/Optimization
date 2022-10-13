import numpy as np


expds = [1, 10]

class Quadratic2Function:
    @staticmethod
    def f(x):
        global expds
        n, _ = x.shape
        expds = np.array(expds)
        fx = expds @ (x ** 2)
        return np.squeeze(fx)

    @staticmethod
    def grad(x):
        global expds
        n, _ = x.shape
        grad_fx = 2 * np.array(expds).reshape(-1, 1) * x
        return grad_fx

    @staticmethod
    def hessian(x):
        global expds
        n, _ = x.shape
        hessian_fx = np.diag(2 * np.array(expds))
        return hessian_fx

    @staticmethod
    def initialize(x0=None, n=2):
        if x0:
            x0 = np.array(x0).reshape(-1, 1)
        else:
            x0 = np.array([10, 30]).reshape(-1, 1)
        return x0


if __name__ == '__main__':
    qf = QuadraticFunction()
    x = np.arange(1, 5).reshape(-1, 1)
    print(qf.f(x))
    assert qf.f(x) == 100