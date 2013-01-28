from numpy import exp

def fun2(x):
    return exp(-(x**2).sum(axis=0))