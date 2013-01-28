from expfun2 import fun2

def fun(x):
    if x.ndim == 1:
        x = x.reshape((1,-1))
    result = fun2(x)
    return result