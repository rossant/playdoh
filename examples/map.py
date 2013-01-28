"""
Simple example of the :func:`map` function.
"""
from playdoh import *


# The function to parallelize
def fun(x):
    return x ** 2


# This line is required on Windows, any call to a Playdoh function
# must be done after this line on this OS.
# See http://docs.python.org/library/multiprocessing.html#windows
if __name__ == '__main__':
    # Execute ``fun(1)`` and ``fun(2)`` in parallel on two CPUs on this machine
    # and return the result.
    print map(fun, [1, 2], cpu=2)
