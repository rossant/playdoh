"""
Custom Pool class, allowing to process tasks in a queue and change dynamically
the number of CPUs to use.
"""
from gputools import *
from debugtools import *
import multiprocessing
import threading
import sys
import os
import time
import gc
import ctypes
import numpy
import traceback
import cPickle
import zlib
import math
from multiprocessing import Process, Pipe, sharedctypes, Lock
from threading import Thread
from numpy import array, nonzero, ctypeslib

try:
    MAXCPU = multiprocessing.cpu_count()
except:
    MAXCPU = 0


__all__ = ['MAXCPU', 'CustomPool', 'Pool', 'make_common', 'make_numpy',
           'get_cpu_count']


def get_cpu_count():
    return multiprocessing.cpu_count()


class CustomConnection(object):
    """
    Handles chunking and compression of data.
    """
    def __init__(self, conn, index, lock=None, chunked=False,
                 compressed=False):
        self.conn = conn
        self.chunked = chunked
        self.compressed = compressed
        self.index = index
        self.lock = lock
        self.BUFSIZE = 2048

    def send(self, obj):
        s = cPickle.dumps(obj, -1)

        log_debug("acquiring lock")
        if self.lock is not None:
            self.lock.acquire()

        if self.compressed:
            s = zlib.compress(s)
        if self.chunked:
#            l = int(math.ceil(float(len(s))/self.BUFSIZE))
#            n = l*self.BUFSIZE
            # len(s) is a multiple of BUFSIZE, padding right with spaces
#            s = s.ljust(n)
#            l = "%08d" % l
#            try:
            # length of the message
            n = len(s)
            l = int(math.ceil(float(n) / self.BUFSIZE))
            log_debug("pipe %d: %d bytes to send in %d packet(s)" %
                (self.index, n, l))
            self.conn.send(n)
            time.sleep(.001)
            for i in xrange(l):
                log_debug("pipe %d: sending packet %d/%d" %
                    (self.index, i + 1, l))
                data = s[i * self.BUFSIZE:(i + 1) * self.BUFSIZE]
#                ar = arr.array('c', data)
#                self.conn.send_bytes(ar)
                self.conn.send_bytes(data)
                time.sleep(.001)
            log_debug("pipe %d: sent %d bytes" % (self.index, n))
#            except:
#                log_warn("Connection error")
        else:
            self.conn.send(s)

        log_debug("releasing lock")
        if self.lock is not None:
            self.lock.release()

    def recv(self):
        if self.chunked:
            # Gets the first 8 bytes to retrieve the number of packets.
#            l = ""
#            n = 8
#            while n > 0:
#                l += self.conn.recv(n)
#                n -= len(l)
            # BUG: sometimes l is filled with spaces??? setting l=1 in this
            # case (not a terrible solution)
#            try:
#                l = int(l)
#            except:
#                log_warn("transfer error, the paquet size was empty")
#                l = 1

            n = int(self.conn.recv())
            log_debug("pipe %d: %d bytes to receive" % (self.index, n))
#            length = l*self.BUFSIZE
            s = ""
            # Ensures that all data have been received
#            for i in xrange(l):# len(s) < length:
            while len(s) < n:
#                ar = arr.array('c', [0]*self.BUFSIZE)
                log_debug("pipe %d: receiving packet..." % (self.index))
#                n = self.conn.recv_bytes_into(ar)
#                s += ar.tostring()
                s += self.conn.recv_bytes()
#                if not self.conn.poll(.01): break
                time.sleep(.001)
            log_debug("pipe %d: received %d bytes" % (self.index, len(s)))
        else:
            s = self.conn.recv()
        # unpads spaces on the right
#        s = s.rstrip()
        if self.compressed:
            s = zlib.decompress(s)
        return cPickle.loads(s)

    def close(self):
        self.conn.close()

#class CustomPipe(object):
#    def __init__(self):
#        # TODO: try to fix the linux bug
##        log_debug("calling Pipe()")
#        parent_conn, child_conn = Pipe()
#        self.parent_conn = CustomConnection(parent_conn)
#        self.child_conn = CustomConnection(child_conn)


def getCustomPipe(index):
    parent_conn, child_conn = Pipe()
    parent_lock = Lock()
    child_lock = Lock()
    return (CustomConnection(parent_conn, index, parent_lock),
            CustomConnection(child_conn, index, child_lock))
#    cpipe = CustomPipe()
#    return cpipe.parent_conn, cpipe.child_conn


class Task(object):
    def __init__(self, fun, do_redirect=None, *args, **kwds):
        self.fun = fun
        self.args = args
        self.kwds = kwds
        self.do_redirect = do_redirect
        self.set_queued()

    def set_queued(self):
        self._status = 'queued'

    def set_processing(self):
        self._status = 'processing'

    def set_finished(self):
        self._status = 'finished'

    def set_crashed(self):
        self._status = 'crashed'

    def set_killed(self):
        self._status = 'killed'

    def get_status(self):
        return self._status
    status = property(get_status)


def eval_task(index, child_conns, parent_conns, shared_data, task, type):
    """
    Evaluates the task on unit index of given type ('CPU' or 'GPU')
    """
    if type == 'GPU':
        set_gpu_device(index)
        if task.do_redirect:
            sys.stdin = file(os.devnull)
            sys.stdout = file(os.devnull)
        if task.do_redirect is None and os.name == 'posix':
            log_warn("WARNING: specify do_redirect=True if CUDA code is not\
                compiling. see \
                <http://playdoh.googlecode.com/svn/docs/playdoh.html#gpu>")
    log_info("Evaluating task on %s #%d" % (type, index + 1))
    # shared data: if there is shared data, pass it in the task's kwds
    # task fun must have fun(..., shared_data={})
    if len(shared_data) > 0:
        task.kwds['shared_data'] = shared_data
    result = task.fun(*task.args, **task.kwds)
#    log_debug("Task successfully evaluated on %s #%d..." % (type, index))
    if type == 'GPU':
#        set_gpu_device(0)
        close_cuda()  # ensures that the context specific to the process is
                      # closed at the process termination
    child_conns[index].send(result)


def make_common_item(v):
    if isinstance(v, numpy.ndarray):
        shape = v.shape
        mapping = {
            numpy.dtype(numpy.float64): ctypes.c_double,
            numpy.dtype(numpy.int32): ctypes.c_int,
            }
        ctype = mapping.get(v.dtype, None)
        if ctype is not None:
            log_debug('converting numpy array to common array')
            v = v.flatten()
            v = sharedctypes.Array(ctype, v, lock=False)
    else:
        # shape = None means that v is not an array and should not be converted
        # back as numpy item
        shape = None
    return v, shape


def make_numpy_item((v, shape)):
    if shape is not None:
        try:
            v = ctypeslib.as_array(v)
            v.shape = shape
            log_debug('converting common array to numpy array')
        except:
            log_debug('NOT converting common array to numpy array')
            pass
    return v


def make_common(shared_data):
    shared_data = dict([(k, make_common_item(v)) for k, v in \
                        shared_data.iteritems()])
    return shared_data


def make_numpy(common_shared_data):
#    shared_args = common_shared_data[1]
#    shared_kwds = common_shared_data[2]
#    args = [make_numpy_item(v) for v in shared_args]
    shared_data = dict([(k, make_numpy_item(v)) for k, v in \
                        common_shared_data.iteritems()])
#    return (common_shared_data[0], args, kwds)
    return shared_data


def process_fun(index, child_conns, parent_conns, shared_data):
    """
    This function is executed on a child process.
    conn is a connection to the parent
    """
    shared_data = make_numpy(shared_data)
    conn = child_conns[index]
    while True:
        log_debug('process_fun waiting...')
        try:
            fun, args, kwds = conn.recv()
        except Exception:
            log_warn(traceback.format_exc())
        log_debug('process_fun received function <%s>' % fun)
        if fun is None:
            break
        # the manager can poll for the process status by sending
        # '_process_status',
        # if it answers 'idle' then it's idle, otherwise it means that it
        # is busy.
#        if fun == '_process_status':
#            log_debug('sending <idle>')
#            conn.send('idle')
#            continue
        try:
            result = fun(index, child_conns, parent_conns,
                         shared_data, *args, **kwds)
        except Exception:
            log_warn(str(fun))
            log_warn(traceback.format_exc())
        del fun, args, kwds, result
        gc.collect()
    log_debug('process_fun finished')

#    close_cuda()

    conn.close()


class Pool(object):
    def __init__(self, workers, npipes=2, globs=None):
        self.workers = workers
        if globs is None:
            globs = globals()
        self.globals = globs
        self.npipes = npipes  # number of pipes/unit

        self.parent_conns = [None] * (workers * npipes)
        self.child_conns = [None] * (workers * npipes)
        self.processes = [None] * (workers)
        self.pids = [None] * (workers)
        self.status = [None] * (workers)

        self.launch_workers()

    def launch_workers(self, units=None, shared_data={}):
        log_debug("launching subprocesses...")
        if units is None:
            units = range(self.workers)
        elif type(units) is int:
            units = range(units)

        # create pipes
        for i in xrange(self.npipes):
            for unit in units:
                pconn, cconn = getCustomPipe(unit + i * self.workers)
                self.parent_conns[unit + i * self.workers] = pconn
                self.child_conns[unit + i * self.workers] = cconn
#        log_debug((self.npipes, units, self.parent_conns))

        # create processes
        for unit in units:
            p = Process(target=process_fun, args=(unit,
                                                  self.child_conns,
                                                  self.parent_conns,
                                                  shared_data))
            p.start()
            self.pids[unit] = p.pid
            self.processes[unit] = p
            self.status[unit] = 'idle'
            time.sleep(.01)

    def close_workers(self, units):
        log_debug("closing the subprocesses...")
        # close pipes
        for i in xrange(self.npipes):
            for unit in units:
                self.parent_conns[unit + i * self.workers].close()
                self.child_conns[unit + i * self.workers].close()

        # close processes
        for unit in units:
            p = self.processes[unit]
            p.terminate()
            del p

    def restart_workers(self, units=None, shared_data={}):
        shared_data = make_common(shared_data)
        if units is None:
            units = range(self.workers)
        elif type(units) is int:
            units = range(units)

        log_debug("restarting the subprocesses...")
        self.close_workers(units)
        time.sleep(.5)
        self.launch_workers(units, shared_data)
        log_debug("restarted the subprocesses!")

#        for unit in units:
#            pconn, cconn = Pipe()
#            self.parent_conns[unit]= pconn
#            self.child_conns[unit] = cconn
#
#            p = Process(target=process_fun, args=(unit,
#                                                  self.child_conns,
#                                                  self.parent_conns,
#                                                  shared_data))
#            p.start()
#            self.pids[unit] = p.pid
#            self.processes[unit] = p

    def set_status(self, index, status):
        self.status[index] = status

    def get_status(self):
        return self.status

    def get_idle_units(self, n):
        """
        Returns the indices of n idle units
        """
        status = self.get_status()
        unitindices = []
        for i in xrange(len(status)):
            if status[i] == 'idle':
                unitindices.append(i)
            if len(unitindices) == n:
                break
        return unitindices

    def execute(self, index, fun, *args, **kwds):
        self.parent_conns[index].send((fun, args, kwds))

    def __getitem__(self, index):
        """
        Allows to use the following syntax:
            pool[i].fun(*args, **kwds)
        instead of
            pool.execute(i, fun, *args, **kwds)
        """
        class tmp(object):
            def __init__(self, obj):
                self.obj = obj

            def __getattr__(self, name):
                if name == 'recv':
                    #self.obj.parent_conns[index].recv()
                    return lambda: self.obj.recv(unitindices=[index])
                if name == 'send':
                    #self.obj.parent_conns[index].send(data)
                    return lambda data:\
                                    self.obj.send(data, unitindices=[index])
                function = self.obj.globals[name]
                return lambda *args, **kwds: self.obj.execute(index, function,
                                                              *args, **kwds)
        t = tmp(self)
        return t

    def send(self, data, unitindices=None, confirm=False):
        """
        Sends data to all workers on this machine or only to the specified
        workers (unitindices).
        data must be a tuple (fun, args, kwds). Numpy arrays in args and kwds
        will be shared
        to save memory.
        """

        # TEST
        confirm = False

        if unitindices is None:
            unitindices = xrange(self.workers)
        if not confirm:
            [self.parent_conns[i].send(data) for i in unitindices]
        else:
            while True:
                [self.parent_conns[i].send(data) for i in unitindices]
                # if send OK, terminate, otherwise, send again to the units
                # which returned False
                unitindices = nonzero(array([not self.parent_conns[i].recv()
                    for i in unitindices]))[0]
                if len(unitindices) == 0:
                    break
                log_debug("pickling error, sending data again")

    def recv(self, unitindices=None):  # ,discard = None, keep = None):
        """
        Receives data from all workers on this machine or only to the specified
        workers (unitindices).
        Discard allows to discard sent data as soon as data==discard
        keep is the opposite of discard: it allows to recv a specific data
        """
        if unitindices is None:
            unitindices = xrange(self.workers)
        result = []
        i = 0
#        discarded = []
#        kept = []
#        log_debug((unitindices, self.parent_conns))
        while i < len(unitindices):
            ui = unitindices[i]
            r = self.parent_conns[ui].recv()
#            log_debug('recv <%s, %s>' % (str(r), str(type(r))))
#            if (discard is not None) and (r == discard):
#                discarded.append(ui)
#                continue
#            if (keep is not None) and (r != keep):
#                kept.append(ui)
#                continue
            result.append(r)
            i += 1
#        if discard is not None:
#            return result, discarded
#        elif kept is not None:
#            return result, kept
#        else:
        return result

    def map(self, fun, args, unitindices=None):
        """
        Maps a function with a list of arguments. We always have:
            len(args) == self.workers
        NOTE: fun must accept 3 arguments before 'args':
            index, child_conns, parent_conns
        """
        if unitindices is None:
            unitindices = xrange(self.workers)
        for i in xrange(len(args)):
            arg = args[i]
#            self.set_status(i, 'busy')
            self.execute(unitindices[i], fun, *arg)
        results = []
        for i in xrange(len(args)):
            results.append(self.parent_conns[unitindices[i]].recv())
#            self.set_status(i, 'idle')
        return results

    def _close(self, i):
        try:
            log_debug("sending shutdown connection to subprocess %d" % i)
            self.parent_conns[i].send((None, (), {}))
        except:
            log_warn("unable to send shutdown connection to subprocess %d" % i)
#        self.parent_conns[i].close()
        self.processes[i].join(.1)
        self.processes[i].terminate()

    def close(self):
        threads = []
        for i in xrange(self.workers):
            t = Thread(target=self._close, args=(i,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join(.1)

#    def join(self):
#        self.close()
#
#    def terminate(self):
#        self.close()


class CustomPool(object):
    """
    Custom Pool class. Allows to evaluate asynchronously any number of tasks
    over any number of CPUs/GPUs. The number of CPUs/GPUs can be changed
    dynamically.
    Tasks statuses can be retrieved at any time.

    Transparent support for both CPUs and GPUs.

    No exception handling here : the function to be mapped is assumed to handle
    exceptions, and return (not raise) an Exception object if any exception
    occured.
    """
    def __init__(self, units, unittype='CPU', do_redirect=None):
        """
        units is either a list of unit indices, or an integer (number of
        units)
        """
        if type(units) is not int:
            self.unitindices = units
            units = len(units)
        else:
            self.unitindices = range(units)
        self.units = units  # number of CPUs/GPUs
        self.type = unittype
        self.pool = None
        self.running = True  # set to False to stop the pool properly
        self.tasks = []  # task queue
        self.current_step = 0  # current iteration step
        self.thread = None
        self.results = {}
        self.do_redirect = do_redirect

    def _run(self):
#        prevunits = self.units

        # IMPORTANT
        # self.pool can be set by an external class, so that the same
        # pool can be associated to multiple CustomPools
        if self.pool is None:
            log_debug("The multiprocessing pool wasn't initialized,\
                       initializing now")
            self.pool = self.create_pool(self.units)

        # the pool is now busy
        [self.pool.set_status(i, 'busy') for i in self.unitindices]

        if len(self.shared_data) > 0:
            self.pool.restart_workers(self.unitindices,
                                      shared_data=self.shared_data)

        i = self.current_step  # current step, task index is i+j

        while (i < len(self.tasks)) and self.running:
            units = self.units
            if units == 0:
                log_warn("No units available, terminating now")
                return
#            if (prevunits is not units):
#                log_info("Changing the number of units from %d to %d"\
#                               % (prevunits, units))
#                self.pool.close()
#                self.pool.join()
#                self.pool = multiprocessing.Pool(units)

            # Selects tasks to evaluate on the units
            local_tasks = self.tasks[i:i + units]
            if len(local_tasks) == 0:
                log_warn("There is no task to process")
                return

            log_debug("Processing tasks #%d to #%d..." % (i + 1,
                                                         i + len(local_tasks)))

            # Sends the tasks to the pool of units
            [task.set_processing() for task in local_tasks]
            local_results = self.pool.map(eval_task,
                                          [(local_tasks[j], self.type) for j in
                                                xrange(len(local_tasks))],
                                          unitindices=self.unitindices)

            for j in xrange(len(local_results)):
                result = local_results[j]
                self.results[i + j] = result
                if isinstance(result, Exception):
                    self.tasks[i + j].set_crashed()
                else:
                    self.tasks[i + j].set_finished()
            log_info("Tasks #%d to #%d finished" %
                        (i + 1, i + len(local_tasks)))

#            prevunits = units
            i += len(local_tasks)
            self.current_step = i

        # the pool is now available
        [self.pool.set_status(i, 'idle') for i in self.unitindices]

        log_debug("All tasks finished")

    def _create_tasks(self, fun, *argss, **kwdss):
        """
        Creates tasks from a function and a list of arguments
        """
        tasks = []
        k = len(argss)  # number of non-named arguments
        keys = kwdss.keys()  # keyword arguments

        i = 0  # task index
        while True:
            try:
                args = [argss[l][i] for l in xrange(k)]
                kwds = dict([(key, kwdss[key][i]) for key in keys])
            except:
                break
            task = Task(fun, self.do_redirect, *args, **kwds)  # do_redirect
            tasks.append(task)
            i += 1

        return tasks

    def set_units(self, units):
        log_debug("setting units to %d" % units)
        # TODO
#        log_debug(self.unitindices)
        self.unitindices = self.pool.get_idle_units(units)
#        log_debug(self.unitindices)
        self.units = units

    def add_tasks(self, fun, *argss, **kwdss):
        new_tasks = self._create_tasks(fun, *argss, **kwdss)
        log_debug("Adding %d tasks" % len(new_tasks))
        n = len(self.tasks)  # current number of tasks
        self.tasks.extend(new_tasks)
        ids = range(n, n + len(new_tasks))
        for id in ids:
            self.results[id] = None
        return ids

    def run_thread(self):
        log_debug("running thread")
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def submit_tasks(self, fun, shared_data, *argss, **kwdss):
        """
        Evaluates fun on a list of arguments and keywords over a pool of CPUs.
        Same as Pool.map, except that the number of CPUs can be changed during
        the processing of the queue. Also, it is a non-blocking call (separate
        thread).

        Exception handling must be performed inside fun : if an Exception
        occurs,
        fun must return the Exception object.
        """
        self.shared_data = shared_data
        ids = self.add_tasks(fun, *argss, **kwdss)
        # Launches the new tasks if the previous ones have stopped.
        if self.thread is None or not self.thread.is_alive():
            self.run_thread()
        return ids

    def clear_tasks(self):
        log_debug("clearing all pool tasks")
        self.running = False
        self.join()
        self.tasks = []
        self.current_step = 0
        self.thread = None
        self.results = {}
        self.running = True

    @staticmethod
    def create_pool(units=MAXCPU):
        if units > 0:
            log_debug("Creating multiprocessing pool with %d units" % units)
#            return multiprocessing.Pool(units)
            return Pool(units)  # Pool class defined above instead of the
                                # multiprocessing.Pool
        else:
            log_debug("NOT creating multiprocessing pool with 0 unit!")
            return None

    @staticmethod
    def close_pool(pool):
        log_debug("Closing multiprocessing pool")
        pool.terminate()
        pool.join()

    def map(self, fun, shared_data, *argss, **kwdss):
        ids = self.submit_tasks(fun, shared_data, *argss, **kwdss)
        return self.get_results(ids)

    def get_status(self, ids):
        """
        Returns the status of each task. Non-blocking call.
        """
        return [self.tasks[id].status for id in ids]

    def get_results(self, ids):
        """
        Returns the result. Blocking call.
        """
        self.join()
        return [self.results[id] for id in ids]

    def join(self):
        if self.thread is not None:
            log_debug("joining CustomPool thread")
            self.thread.join()
        else:
            log_debug("tried to join CustomPool thread but it wasn't active")
        [self.pool.set_status(i, 'idle') for i in self.unitindices]

    def has_finished(self, ids):
        for s in self.get_status(ids):
            if s is not "finished":
                return False
        return True

    def close(self):
        """
        Waits for the current processings to terminate then terminates the job
        processing.
        """
        self.running = False
        self.join()

        for i in xrange(len(self.tasks)):
            if (self.tasks[i].status == 'queued'):
                self.tasks[i].set_killed()

        if self.pool is not None:
            log_debug("closing pool of processes")
            self.pool.close()

    def kill(self):
        """
        Kills immediately the task processing.
        """
        self.running = False

        if self.pool is not None:
            self.pool.close()
#            self.pool.terminate()
#            self.pool.join()
            self.pool = None

        del self.thread

        for i in xrange(len(self.tasks)):
            if (self.tasks[i].status == 'queued' or
                    self.tasks[i].status == 'processing'):
                self.tasks[i].set_killed()
