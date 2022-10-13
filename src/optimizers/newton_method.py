import numpy as np
import numpy.linalg as lalg

from src.utils import grad_f_estimate, line_search


def newton_method(f, x0, hessian, alpha=1, max_iters=1000,
                  grad=None, delta1=1e-6, delta2=1e-6,
                  display=False, **kwargs):
    x_k = x0
    fx_ls = []
    xk_ls = []

    if not grad:
        grad = lambda inp: grad_f_estimate(f, inp)

    fx_ls = []

    for k in range(max_iters):
        fx_ls.append(f(x_k))
        xk_ls.append(x_k)

        grad_k = grad(x_k)
        hessian_k = hessian(x_k)
        p_k = - np.linalg.inv(hessian_k) @ grad_k

        if display:
            print(f"x_k: {x_k}, \nfunction: {f(x_k)}, \ngrad: {grad_k}\n")

        alpha_k = line_search(x_k, p_k, f, grad, alpha)
        x_k_next = x_k + alpha_k * p_k

        if (np.linalg.norm(x_k_next - x_k) < delta1):
            print("No x movement")
            break

        if (np.linalg.norm(f(x_k_next) - f(x_k)) < delta2):
            print("No fx movement")
            break
        x_k = x_k_next

    if k == max_iters - 1:
        print("Max Iter Reached")

    return k, x_k, f(x_k), fx_ls, xk_ls
