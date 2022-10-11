import numpy as np

from src.utils import grad_f_estimate, line_search


def adam_optimizer(f, x0, max_iters=1000,
                     grad=None, delta1=1e-6, delta2=1e-6,
                     display=False, **kwargs):
    x_k = x0
    fx_ls = []

    if not grad:
        grad = lambda inp: grad_f_estimate(f, inp)

    s = 0
    r = 0

    rho1 = 0.9
    rho2 = 0.999
    delta = 1e-8

    for k in range(max_iters):
        fx_ls.append(f(x_k))

        g = grad(x_k)

        s = rho1 * s + (1 - rho1) * g
        r = rho2 * r + (1 - rho2) * g ** 2

        r = r + g ** 2

        s_ = s / (1 - rho1 ** (k + 1))
        r_ = r / (1 - rho2 ** (k + 1))

        if display:
            print(f"x_k: {x_k}, \nfunction: {f(x_k)}, \ngrad: {g}\n")

        p_k = - s_ / (np.sqrt(r_) + delta)
        p_k = - g / (np.sqrt(r) + delta)

        alpha_k = line_search(x_k, p_k, f, grad)
        x_k_next = x_k + alpha_k * p_k

        if (np.linalg.norm(x_k_next - x_k) < delta1) or (np.linalg.norm(f(x_k_next) - f(x_k)) < delta2):
            break

        x_k = x_k_next

    return k, x_k, f(x_k), fx_ls
