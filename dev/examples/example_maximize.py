from playdoh import *

def fun(x):
    import numpy
    if x.ndim == 1:
        x = x.reshape((1,-1))
    result = numpy.exp(-(x**2).sum(axis=0))
    return result

if __name__ == '__main__':
    from numpy import *
    dimension = 4
    initrange = tile([-10,10], (dimension,1))
    results = maximize(fun,
                       popsize = 10000,
                       maxiter = 10,
                       cpu = 1,
                       codedependencies = [],
                       returninfo = True,
                       initrange = initrange)
    print_table(results)