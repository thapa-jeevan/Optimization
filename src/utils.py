import os

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
    iter_, x_final = result

    file_path = os.path.join(REPORTS_DIR, f"{func_}_{opt_}_result.txt")
    with open(file_path, "w") as f:
        f.write(f"iter: {iter_}\n\n")
        f.write(f"final_x:{x_final}")
