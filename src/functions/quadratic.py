import numpy as np


class QuadraticFunction:
    @staticmethod
    def f(x):
        n, _ = x.shape
        fx = np.arange(1, n + 1) @ (x ** 2)
        return fx

    @staticmethod
    def grad(x):
        n, _ = x.shape
        grad_fx = 2 * np.arange(1, n + 1).reshape(n, 1) * x
        return grad_fx

    @staticmethod
    def hessian(x):
        n, _ = x.shape
        hessian_fx = np.diag(2 * np.arange(1, n + 1))
        return hessian_fx

    @staticmethod
    def initialize(x0=None, n=100):
        if x0:
            x0 = np.array(x0).reshape(-1, 1)
        else:
            x0 = np.random.randint(2, 5, (n, 1))
        return x0


# if __name__ == '__main__':
#     qf = QuadraticFunction()
#     x = np.arange(1, 5).reshape(-1, 1)
#     print(qf.f(x))
#     assert qf.f(x) == 100