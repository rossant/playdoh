"""
Simple example of the :func:`map` function with a lambda function.
"""
from playdoh import *


# This line is required on Windows, any call to a Playdoh function
# must be done after this line on this OS.
# See http://docs.python.org/library/multiprocessing.html#windows
if __name__ == '__main__':
    # Execute ``lambda(1)`` and ``lambda(2)`` in parallel on two CPUs
    # on this machine
    # and return the result.
    print map(lambda x: x * x, [1, 2], cpu=2)
