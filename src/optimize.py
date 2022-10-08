import numpy as np

from functions import *
from optimizers import *

max_iters = 1000
x0 = np.array([4, 1, 100]).reshape(-1, 1)

alpha = 3e-1

delta1 = 1e-7
delta2 = 1e-7

search_space = [[f1, grad_f1, hessian_f1],
                [f2, grad_f2, hessian_f2],
                [f3, grad_f3, hessian_f3]]

for (f, grad_f, hessian_f) in search_space:
    gradient_descent(
        f, x0,
        alpha=alpha, max_iters=max_iters, delta1=delta1, delta2=delta2,
        display=False
    )

    quasi_newton_method(
        f, x0,
        alpha=alpha, max_iters=max_iters, delta1=delta1, delta2=delta2,
        display=False
    )

    newton_method(
        f, x0, hessian_f, grad_f_=grad_f,
        alpha=alpha, max_iters=max_iters, delta1=delta1, delta2=delta2,
        display=False
    )
