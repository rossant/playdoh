from playdoh import *
from test import *
import numpy
import sys
import time
from numpy import int32, ceil


def fitness_fun(x):
    return numpy.exp(-x * x)


class GATest(unittest.TestCase):
    def test_1(self):
        result = maximize(fitness_fun,
                          algorithm=GA,
                          maxiter=3,
                          popsize=1000,
                          cpu=3,
                          x_initrange=[-5, 5])
        self.assertTrue(abs(result[0]['x']) < .1)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(GATest)


if __name__ == '__main__':
    unittest.main()
