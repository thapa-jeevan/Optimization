from .functions import *
from .optimizers import *


def get_optimizer(optimizer_name):
    if optimizer_name == "gradient_descent":
        return gradient_descent

    elif optimizer_name == "newton_method":
        return newton_method

    elif optimizer_name == "quasi_newton_method":
        return quasi_newton_method

    else:
        raise NotImplementedError(optimizer_name + " is not implemented!")


def get_function(func_name):
    if func_name == "quadratic":
        return f1, grad_f1, hessian_f1

    elif func_name == "log":
        return f2, grad_f2, hessian_f2

    elif func_name == "bi_quadratic":
        return f3, grad_f3, hessian_f3

    else:
        raise NotImplementedError(func_name + " is not implemented!")
