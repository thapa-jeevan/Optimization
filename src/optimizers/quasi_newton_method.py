import numpy as np
import numpy.linalg as lalg

from src.utils import grad_f_estimate


def quasi_newton_method(f, x0, alpha=1e-3, max_iters=1000,
                        grad_f_=None, delta1=1e-6, delta2=1e-6,
                        display=False):
    x_k = x0

    x_k = x_k.reshape(-1, 1)
    n, _ = x_k.shape

    F_k = np.eye(n)

    grad_f = lambda inp: grad_f_estimate(f, inp) if grad_f_ is None else grad_f_(inp)

    for i in range(max_iters):
        grad_k = grad_f(x_k)
        p = - F_k @ grad_k

        if display:
            print(f"x_k: {x_k}, \nfunction: {f(x_k)}, \ngrad: {grad_k}\n")

        x_k_next = x_k + alpha * p

        if (lalg.norm(x_k_next - x_k) < delta1) or (lalg.norm(f(x_k_next) - f(x_k)) < delta2):
            break

        s_k = x_k_next - x_k
        y_k = grad_f(x_k_next) - grad_k

        F_k = F_k + (y_k.T @ (F_k @ y_k + s_k)).squeeze() * s_k @ s_k.T / ((y_k.T @ s_k).squeeze() ** 2) \
              - (s_k @ y_k.T @ F_k + F_k @ y_k @ s_k.T) / (y_k.T @ s_k).squeeze()

        x_k = x_k_next

    return i, x_k
