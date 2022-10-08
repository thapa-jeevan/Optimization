import numpy as np
import numpy.linalg as lalg

from src.utils import grad_f_estimate


def newton_method(f, x0, hessian_f, alpha=1e-3, max_iters=1000,
                  grad=None, delta1=1e-6, delta2=1e-6,
                  display=False):
    x_k = x0

    grad_f = lambda inp: grad_f_estimate(f, inp) if grad is None else grad(inp)

    for i in range(max_iters):
        grad = grad_f(x_k)
        hessian = hessian_f(x_k)
        p = - np.linalg.inv(hessian) @ grad

        if display:
            print(f"x_k: {x_k}, \nfunction: {f(x_k)}, \ngrad: {grad}\n")

        x_k_next = x_k + alpha * p

        if (lalg.norm(x_k_next - x_k) < delta1) or (lalg.norm(f(x_k_next) - f(x_k)) < delta2):
            break

        x_k = x_k_next

    return i, x_k
