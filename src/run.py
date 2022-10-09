from .func_opt_factory import get_function, get_optimizer
from .utils import process_config, dump_result, read_config

if __name__ == '__main__':
    cfg = read_config()
    for func_, opt_config in cfg.items():
        for opt_, config in opt_config.items():
            opt_func = get_function(func_)()

            print(func_, opt_, config)
            config = process_config(opt_func, config)

            optimizer = get_optimizer(opt_)
            result = optimizer(**config)
            # print(result, "\n")
            dump_result(func_, opt_, result)
