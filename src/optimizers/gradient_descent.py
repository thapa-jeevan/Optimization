import numpy.linalg as lalg

from src.utils import grad_f_estimate, line_search


def gradient_descent(f, x0, alpha=None, max_iters=1000,
                     grad=None, delta1=1e-6, delta2=1e-6,
                     display=False, **kwargs):
    x_k = x0

    if not grad:
        grad = lambda inp: grad_f_estimate(f, inp)

    for k in range(max_iters):
        p_k = - grad(x_k)

        if display:
            print(f"x_k: {x_k}, \nfunction: {f(x_k)}, \ngrad: {p_k}\n")

        alpha_k = line_search(x_k, p_k, f, grad)
        x_k_next = x_k + alpha_k * p_k

        if (lalg.norm(x_k_next - x_k) < delta1) or (lalg.norm(f(x_k_next) - f(x_k)) < delta2):
            break

        x_k = x_k_next

    return k, x_k, f(x_k)
