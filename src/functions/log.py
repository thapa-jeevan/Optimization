import ast
import os

import numpy as np

from src.settings import CONFIGS_DIR


class LogTransformFunction:
    def __init__(self):
        self.m = 500
        self.n = 100
        function2_config_dir = os.path.join(CONFIGS_DIR, "function2")
        for t_ in ["A", "b", "c"]:
            file_path = os.path.join(function2_config_dir, f"fun2_{t_}.txt")
            assert os.path.exists(file_path)

            with open(file_path, "r") as f:
                val = f.read()
                val = '[' + val.strip().replace('  ', ' ').replace(' ', ', ') + ']'
                val = ast.literal_eval(val)
                val = np.array(val)
                setattr(self, t_, val)

        self.A = self.A.reshape(self.m, self.n)
        self.b = self.b.reshape(self.m, 1)
        self.c = self.c.reshape(self.n, 1)

    def f(self, x):
        t = self.b - self.A @ x
        # print(f"\r {(t < 0).any()}\t\t", np.abs(x).max(), end="")
        # if np.isnan(np.abs(x).max()):
        #     exit(0)
        fx = self.c.T @ x - (np.log(self.b - self.A @ x)).sum()
        # if (t < 0).any():
        #     # print(fx)
        #     import pdb
        #     pdb.set_trace()

        # if np.isnan(fx).any():
        #     import pdb
        #     pdb.set_trace()
        #     exit(0)
        # print(np.abs(fx).max())

        return np.squeeze(fx)

    def grad(self, x):
        grad_fx = self.c + self.A.T @ (1 / (self.b - self.A @ x))
        return grad_fx

    def hessian(self, x):
        hessian = self.A.T @ np.diag((1. / ((self.b - self.A @ x) ** 2)).ravel()) @ self.A
        return hessian

    def initialize(self, x0=None):
        x0 = np.linalg.inv(self.A.T @ self.A) @ self.A.T @ (self.b - np.ones_like(self.b) * 20)
        return x0

    def get_solution(self):
        y = - 1 / (- np.linalg.inv(self.A @ self.A.T) @ self.A @ self.c)
        return np.linalg.inv(self.A.T @ self.A) @ self.A.T @ (self.b + y)


if __name__ == '__main__':
    func_ = LogTransformFunction()
    soln = func_.get_solution()
    print(soln)
    print(func_.f(soln))
    # x0 = func_.initialize()
    #
    # check = func_.b - func_.A @ x0
    # print((check < 0).any())
    #
    # x_ = np.linalg.inv(func_.A.T @ func_.A) @ func_.A.T @ func_.b
    # # print(x_)
    #
    # import pdb
    # pdb.set_trace()
