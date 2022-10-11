from .functions import *
from .optimizers import *


def get_optimizer(optimizer_name):
    if optimizer_name == "gradient_descent":
        return gradient_descent

    elif optimizer_name == "newton_method":
        return newton_method

    elif optimizer_name == "quasi_newton_method":
        return quasi_newton_method

    elif optimizer_name == "adam_optimizer":
        return adam_optimizer

    else:
        raise NotImplementedError(optimizer_name + " is not implemented!")


def get_function(func_name):
    if func_name == "quadratic":
        return QuadraticFunction

    elif func_name == "log":
        return LogTransformFunction

    elif func_name == "bi_quadratic":
        return BiQuadraticFunction

    else:
        raise NotImplementedError(func_name + " is not implemented!")
