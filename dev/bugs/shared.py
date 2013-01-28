import multiprocessing, sys, os, time, ctypes, numpy, cPickle
from multiprocessing import Process, sharedctypes, Pipe
from numpy import ctypeslib, mean

def make_common_item(v):
    shape = v.shape
    mapping = {
        numpy.dtype(numpy.float64):ctypes.c_double,
        numpy.dtype(numpy.int32):ctypes.c_int,
        }
    ctype = mapping.get(v.dtype, None)
    if ctype is not None:
        v = v.flatten()
        v = sharedctypes.Array(ctype, v, lock=False)
    return v, shape

def make_numpy_item((v, shape)):
    try:
        v = ctypeslib.as_array(v)
        v.shape = shape
    except:
        pass
    return v

def fun(conn, x):
    y = make_numpy_item(x)
    print y.shape, mean(mean(y))
    sys.stdout.flush()
    time.sleep(5)

if __name__ == '__main__':
    cols = 2 # BUG: cols=2
    x = numpy.random.rand(2000000,cols)
    print x.shape, mean(mean(x))
    
    x2 = x
    x2 = make_common_item(x) # comment this to check that the 2 subprocesses use much less memory
    
    parent_conn1, child_conn1 = Pipe()
    p1 = Process(target=fun, args=(child_conn1,x2))
    p1.start()
    
    parent_conn2, child_conn2 = Pipe()
    p2 = Process(target=fun, args=(child_conn2,x2))
    p2.start()
    
    p1.join()
    p1.terminate()
    
    p2.join()
    p2.terminate()
    
    print "Done"
