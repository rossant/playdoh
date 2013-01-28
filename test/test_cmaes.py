from playdoh import *
from test import *
import numpy
import sys
import time
from numpy import int32, ceil


def fitness_fun(x):
    return numpy.exp(-x * x)


class CMAESTest(unittest.TestCase):
    def test_1(self):
        time.sleep(.2)
        result = maximize(fitness_fun,
                          algorithm=CMAES,
                          maxiter=3,
                          popsize=1000,
                          cpu=1,
                          x_initrange=[-5, 5])
        self.assertTrue(abs(result[0]['x']) < .1)

    def test_2(self):
        p1 = Process(target=open_server, args=(2718, 1, 0))
        p1.start()
        time.sleep(.2)

        p2 = Process(target=open_server, args=(2719, 1, 0))
        p2.start()
        time.sleep(.2)

        machines = [('localhost', 2718), ('localhost', 2719)]
        result = maximize(fitness_fun,
                          algorithm=CMAES,
                          maxiter=3,
                          popsize=1000,
                          machines=machines,
                          x_initrange=[-5, 5])
        self.assertTrue(abs(result[0]['x']) < .1)

        close_servers(machines)
        time.sleep(.2)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(CMAESTest)


if __name__ == '__main__':
    unittest.main()
