"""
Example of :func:`maximize` by using a fitness function implemented in a class.
Using a class allows to have an initialization at the beginning of the
optimization.
"""
from playdoh import *
from numpy import exp, tile


# The class must derive from the ``Fitness`` class.
class FitnessTest(Fitness):
    # This method allows to initialize some data. Parameters
    # can be passed using the ``initargs`` and ``initkwds``
    # arguments of ``maximize``.
    def initialize(self, a):
        self.a = a

    # This method is called at every iteration.
    def evaluate(self, x):
        return exp(-((x - self.a) ** 2))


if __name__ == '__main__':
    # Maximize the fitness function in parallel
    results = maximize(FitnessTest,
                       popsize=10000,  # size of the population
                       maxiter=10,  # maximum number of iterations
                       cpu=1,  # number of CPUs to use on the local machine
                       args=(3,),  # parameters for the ``initialize`` method
                       x_initrange=[-10, 10])  # initial interval for the
                                               # ``x`` parameter

    # Display the final result in a table
    print_table(results)
