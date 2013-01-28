__all__ = ['initialise_cuda', 'pycuda', 'MAXGPU', 'CANUSEGPU',
           'set_gpu_device', 'close_cuda', 'get_gpu_count']

try:
    from debugtools import *
    import atexit
    import pycuda
    import pycuda.compiler
    import pycuda.gpuarray
    import pycuda.driver as drv
    import multiprocessing
    pycuda.context = None
    pycuda.isinitialised = False
    MAXGPU = 0
    CANUSEGPU = True

    def get_gpu_count():
        """
        Return the total number of GPUs without initializing PyCUDA on the
        current process.
        """
        pool = multiprocessing.Pool(1)
        result = pool.apply(initialise_cuda)
        return result

    def initialise_cuda():
        """
        Initialize PyCUDA on the current process.
        """
        global MAXGPU
        if not pycuda.isinitialised:
#            log_debug("drvinit")
            drv.init()
            pycuda.isinitialised = True
#            log_debug("drvdevicecount")
            MAXGPU = drv.Device.count()
            log_debug("PyCUDA initialized, %d GPU(s) found" % MAXGPU)
        return MAXGPU

    def set_gpu_device(n):
        """
        Make PyCUDA use GPU number n in the system.
        """
#        log_debug(inspect.stack()[1][3])
        initialise_cuda()
        log_debug("Setting PyCUDA context number %d" % n)
        try:
            pycuda.context.detach()
        except:
            log_debug("Couldn't detach PyCUDA context")
            pass
        if n < MAXGPU:
            pycuda.context = drv.Device(n).make_context()
        else:
            pycuda.context = drv.Device(MAXGPU - 1).make_context()
            log_warn("Unable to set GPU device %d, setting device %d instead" %
                (n, MAXGPU - 1))

    def close_cuda():
        """
        Closes the current PyCUDA context. MUST be called at the end of the
        script.
        """
        log_debug("Trying to close current PyCUDA context")
        if pycuda.context is not None:
            try:
                log_debug("Closing current PyCUDA context")
                pycuda.context.pop()
                pycuda.context = None
            except:
                log_warn("A problem occurred when closing PyCUDA context")

    atexit.register(close_cuda)

except:
    MAXGPU = 0
    CANUSEGPU = False
    pycuda = None

    def get_gpu_count():
        """
        Return the total number of GPUs, 0 here because PyCUDA doesn't \
        appear to be installed.
        """
        return 0

    def initialise_cuda():
        return 0

    def set_gpu_device(n):
        log_warn("PyCUDA not available")
        pass

    def close_cuda():
        pass
