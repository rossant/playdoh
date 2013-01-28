"""
Example of :func:`map` with a function loading CUDA code and running on GPUs.
"""
from playdoh import *
from numpy import *
import pycuda


# The function loading the CUDA code
def fun(scale):
    # The CUDA code, which multiplies a vector by a scale factor.
    code = '''
    __global__ void test(double *x, int n)
    {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     if(i>=n) return;
     x[i] *= %d;
    }
    ''' % scale

    # Compile the CUDA code to GPU code
    mod = pycuda.compiler.SourceModule(code)

    # Transform the CUDA function into a Python function
    f = mod.get_function('test')

    # Create a vector on the GPU filled with 8 ones
    x = pycuda.gpuarray.to_gpu(ones(8))

    # Start the function on the GPU
    f(x, int32(8), block=(8, 1, 1))

    # Load the result from the GPU to the CPU
    y = x.get()

    # Finally, return the result
    return y

# This line is required on Windows, any call to a Playdoh function
# must be done after this line on this OS.
# See http://docs.python.org/library/multiprocessing.html#windows
if __name__ == '__main__':
    # Execute ``fun(2)`` and ``fun(3)`` on 1 GPU on this machine
    # and return the result.
    if CANUSEGPU:
        print map(fun, [2, 3], gpu=1)
