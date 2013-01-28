"""
Example usage of the :func:`map` function with shared data.
"""
from playdoh import *
from numpy.random import rand


# The function to parallelize. The extra argument ``shared_data`` is a
# read-only dictionary
# residing is shared memory on the computer. It can contain large NumPy arrays
# used by
# all the CPUs to execute the function.
def fun(x, shared_data):
    return x + shared_data['x0']


# This line is required on Windows, any call to a Playdoh function
# must be done after this line on this OS.
# See http://docs.python.org/library/multiprocessing.html#windows
if __name__ == '__main__':
    # Execute two function evaluations with a large NumPy array in shared data.
    map(fun,
        [rand(100000, 2), rand(100000, 2)],
        cpu=2,
        shared_data={'x0': rand(100000, 2)})
