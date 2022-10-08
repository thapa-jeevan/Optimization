import yaml

from .func_opt_factory import get_function, get_optimizer

if __name__ == '__main__':
    with open("config.yml", "r") as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

    for func_, opt_config in cfg.items():
        for opt_, config in opt_config.items():
            print(func_, opt_, config)
            optimizer = get_optimizer(opt_)
            optimizer(
                get_function(func_),
                **config,
            )
