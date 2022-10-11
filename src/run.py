import numpy as np

from .func_opt_factory import get_function, get_optimizer
from .utils import process_config, dump_result, read_config, visualize

SEED = 1476585

if __name__ == '__main__':
    cfg = read_config()
    for func_, opt_config in cfg.items():
        fx_ls = {}
        for opt_, config in opt_config.items():
            np.random.seed(SEED)

            opt_func = get_function(func_)()

            print(func_, opt_, config)
            config = process_config(opt_func, config)

            optimizer = get_optimizer(opt_)
            *result, fx_ls_ = optimizer(**config)
            dump_result(func_, opt_, result)

            fx_ls[opt_] = fx_ls_
        visualize(fx_ls, func_)
