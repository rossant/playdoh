"""
Simple example of :func:`maximize`.
"""
from playdoh import *
import numpy


# The fitness function to maximize
def fun(x):
    return numpy.exp(-x ** 2)


if __name__ == '__main__':
    # Maximize the fitness function in parallel
    results = maximize(fun,
                       popsize=10000,  # size of the population
                       maxiter=10,  # maximum number of iterations
                       cpu=1,  # number of CPUs to use on the local machine
                       x_initrange=[-10, 10])  # initial interval for
                                               # the ``x`` parameter

    # Display the final result in a table
    print_table(results)
