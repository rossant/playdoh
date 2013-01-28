from playdoh import *

def fun(x,y,shared_data,groups):
    import numpy
    n = len(x)
    result = numpy.zeros(n)
    x0 = numpy.kron(shared_data['x0'], numpy.ones(n/groups))
    y0 = numpy.kron(shared_data['y0'], numpy.ones(n/groups))
    result = numpy.exp(-(x-x0)**2-(y-y0)**2)
    return result

if __name__ == '__main__':    
    results = maximize(fun,
                       popsize = 10000,
                       maxiter = 10,
                       groups = 3,
                       cpu = 2,
                       shared_data={'x0': [0,1,2], 'y0':[3,4,5]},
                       x_initrange = [-10,10],
                       y_initrange = [-10,10])
    print_table(results)
    