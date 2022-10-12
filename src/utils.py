import os

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .settings import REPORTS_DIR, CONFIGS_DIR


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


def line_search(x_k, p_k, f, grad, alpha0=1, c_=0.1, decrement_ratio=0.5, max_iters=100):
    alpha_k = alpha0

    slope = c_ * p_k.T @ grad(x_k)

    y_intercept = f(x_k)
    y_line = lambda x__: slope * alpha_k + y_intercept

    for _ in range(max_iters):
        x_k1 = x_k + alpha_k * p_k

        if f(x_k1) < y_line(x_k1):
            break
        else:
            alpha_k = alpha_k * decrement_ratio
    return alpha_k


def process_config(opt_func, config):
    x0 = config["x0"]
    config["x0"] = opt_func.initialize(x0)

    config["f"] = opt_func.f

    if config["grad"]:
        config["grad"] = opt_func.grad

    if config["hessian"]:
        config["hessian"] = opt_func.hessian

    return config


def read_config():
    cfg_path = os.path.join(CONFIGS_DIR, "config.yml")
    with open(cfg_path, "r") as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    return cfg


def dump_result(func_, opt_, result):
    iter_, x_final, fx_final = result

    file_path = os.path.join(REPORTS_DIR, f"{func_}_{opt_}_result.txt")
    with open(file_path, "w") as f:
        f.write(f"iter: {iter_}\n\n")
        f.write(f"fx_final:{fx_final}\n\n")
        f.write(f"min_x:{x_final.min()}  max_x:{x_final.max()}\n\n")
        f.write(f"final_x:{x_final}")


def visualize(fx_ls, func_):
    font = {'size': 20}
    plt.rc('font', **font)
    file_path = os.path.join(REPORTS_DIR, "graphs", f"{func_}_graph.jpg")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    plt.figure(figsize=(10, 7))

    for opt_, fx_ls_ in fx_ls.items():
        plt.plot(fx_ls_, linewidth=3)
    for opt_, fx_ls_ in fx_ls.items():
        plt.scatter(len(fx_ls_) - 1, fx_ls_[-1], linewidth=5, marker="o")

    plt.title(func_)
    plt.xlabel("#Iteration (k)")
    plt.ylabel("f(x_k)")
    plt.legend(fx_ls.keys())
    # plt.yscale("log")
    plt.savefig(file_path)
    plt.show()
