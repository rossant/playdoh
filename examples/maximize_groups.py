"""
Example of :func:`maximize` with several groups. Groups allow to optimize
a fitness function with different parameters in parallel but by
vectorizing the fitness evaluation for all groups.
"""
from playdoh import *
import numpy


# The fitness function is a Gaussian with different centers for
# different groups ``shared_data`` contains the different centers.
def fun(x, y, nodesize, shared_data, groups):
    # Expand ``x0`` and ``y0`` to match the total population size
    x0 = numpy.kron(shared_data['x0'], numpy.ones(nodesize / groups))
    y0 = numpy.kron(shared_data['y0'], numpy.ones(nodesize / groups))
    # Compute the Gaussian for all centers in a vectorized fashion
    result = numpy.exp(-(x - x0) ** 2 - (y - y0) ** 2)
    return result


if __name__ == '__main__':
    # Maximize the fitness function in parallel
    results = maximize(fun,
                       popsize=50,  # size of the population for each group
                       maxiter=10,  # maximum number of iterations
                       cpu=1,  # number of CPUs to use on the local machine
                       groups=3,  # number of groups
                       algorithm=CMAES,  # optimization algorithm, can be PSO,
                                         # GA or CMAES
                       shared_data={'x0': [0, 1, 2],  # centers of the Gaussian
                                                      # for each group
                                    'y0': [3, 4, 5]},
                       x_initrange=[-10, 10],  # initial interval for the
                                                 # ``x`` parameter
                       y_initrange=[-10, 10])  # initial interval for the
                                                 # ``y`` parameter

    # Display the final result in a table
    print_table(results)
