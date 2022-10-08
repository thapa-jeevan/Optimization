import numpy as np


def grad_f_estimate(f, x, h=1e-5, order=1):
    hs = h * np.eye(len(x))

    if order == 1:
        x_ = (x + hs).T
        x_ = np.expand_dims(x_, axis=-1)
        f_x_h = [f(t) for t in x_]
        grad = (f_x_h - f(x)) / h

    elif order == 2:
        f_x_h_r = [f(x + t) for t in hs]
        f_x_h_l = [f(x - t) for t in hs]
        f_x_h_r = np.array(f_x_h_r)
        grad = (f_x_h_r - f_x_h_l) / (2 * h)
    return grad.reshape(-1, 1)