import numpy as np
import numpy.linalg as lalg

from src.utils import grad_f_estimate


def newton_method(f, x0, hessian, alpha=1e-4, max_iters=1000,
                  grad=None, delta1=1e-6, delta2=1e-6,
                  display=False, **kwargs):
    x_k = x0

    if not grad:
        grad = lambda inp: grad_f_estimate(f, inp)

    for k in range(max_iters):
        grad_k = grad(x_k)
        hessian_k = hessian(x_k)
        p = - np.linalg.inv(hessian_k) @ grad_k

        if display:
            print(f"x_k: {x_k}, \nfunction: {f(x_k)}, \ngrad: {grad_k}\n")

        x_k_next = x_k + alpha * p

        if (lalg.norm(x_k_next - x_k) < delta1) or (lalg.norm(f(x_k_next) - f(x_k)) < delta2):
            break

        x_k = x_k_next

    return k, x_k
