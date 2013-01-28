from playdoh import *
from test import *
import numpy
import sys
import time
from numpy import int32, ceil


def fitness_fun(x_0):
    # test parameter name with '_'
    return numpy.exp(-x_0 * x_0)


def fitness_gpu(x):
    code = '''
    __global__ void test(double *x, int n)
    {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i>=n) return;
     x[i] *= x[i];
    }
    '''
    n = len(x)

    mod = pycuda.compiler.SourceModule(code)
    f = mod.get_function('test')
    x2 = pycuda.gpuarray.to_gpu(x)
    f(x2, int32(n), block=(128, 1, 1), grid=(int(ceil(float(n) / 128)), 1))
    y = x2.get()
    return y


class PsoTest(unittest.TestCase):
    def test_1(self):
        result = maximize(fitness_fun,
                          algorithm=PSO,
                          maxiter=3,
                          popsize=1000,
                          cpu=1,
                          x_0_initrange=[-5, 5])
        self.assertTrue(abs(result[0]['x_0']) < .1)

    def test_gpu(self):
        if CANUSEGPU:
            result = minimize(fitness_gpu,
                              algorithm=PSO,
                              maxiter=3,
                              popsize=1024,
                              gpu=1,
                              x_initrange=[-5, 5])

            self.assertTrue(abs(result[0]['x']) < .1)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(PsoTest)


if __name__ == '__main__':
    unittest.main()
