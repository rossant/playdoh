# -*- coding: utf-8 -*-
from playdoh import *
from test import *
from multiprocessing import Process
from numpy import int32, ones, hstack
import time


code = '''
__global__ void test(double *x, int n)
{
 int i = blockIdx.x * blockDim.x + threadIdx.x;
 if(i>=n) return;
 x[i] *= 2.0;
}
'''


def run(index=0):
    set_gpu_device(index)

    n = 100
    mod = pycuda.compiler.SourceModule(code)
    f = mod.get_function('test')
    x = pycuda.gpuarray.to_gpu(ones(n))
    f(x, int32(n), block=(n, 1, 1))
    y = x.get()
    close_cuda()

    if max(y) == min(y) == 2:
        log_info("GPU test #%d PASSED" % (index + 1))
    else:
        log_warn("GPU test #%d FAILED" % (index + 1))

    return y


def pycuda_fun(n=100):

    import pycuda
    from numpy import ones, int32
    code = '''
    __global__ void test(double *x, int n)
    {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i>=n) return;
     x[i] *= 2.0;
    }
    '''

    mod = pycuda.compiler.SourceModule(code)
    f = mod.get_function('test')
    x = pycuda.gpuarray.to_gpu(ones(n))
    f(x, int32(n), block=(n, 1, 1))
    y = x.get()
    return y


class GpuTest(unittest.TestCase):
    def test_1(self):

        if not CANUSEGPU:
            log_warn("PyCUDA is not installed, can't use GPU")
            return

        p1 = Process(target=run, args=(0,))
        p1.start()
        time.sleep(.2)

        p2 = Process(target=run, args=(1,))
        p2.start()
        time.sleep(.2)

        p1.join()
        p2.join()

        log_info("GPU tests passed")

    def test_2(self):
        log_info("test2")
        if not CANUSEGPU:
            log_warn("PyCUDA is not installed, can't use GPU")
            return

        r = map(pycuda_fun, [100, 100], gpu=1)
        r = hstack(r)
        b = r.max() == r.min() == 2
        if b:
            log_info("GPU jobs passed")
        else:
            log_warn("GPU jobs failed")
        self.assertTrue(b)


def test_suite():
    return unittest.TestLoader().loadTestsFromTestCase(GpuTest)


if __name__ == '__main__':
    unittest.main()
