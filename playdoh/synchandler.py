from baserpc import *
from rpc import *
from gputools import *
from resources import *
from debugtools import *
from codehandler import *
from threading import Thread, Lock
from Queue import Queue

import sys
import time
import traceback
import os

__all__ = ['SyncHandler', 'Tubes', 'Node',
           'ParallelTask', 'TaskRun']


class Node(object):
    def __init__(self, index, machine, type, unitidx=0):
        """
        A Node represents an abstract working unit in the graph. It is uniquely
        identified by an integer between 0 and N-1 if there are N nodes in the
        graphs.
        A Node can be mapped to a CPU/GPU or to a machine (type='CPU', 'GPU' or
        'machine').
        'unitidx' is then the CPU/GPU index between 0 and K-1 (K-core machine),
        and machine is
        a Machine object.
        """
        self.index = index
        self.machine = Machine(machine)
        self.type = type
        self.unitidx = unitidx

    def __repr__(self):
        return '<Node %d on %s, %s %d>' % (self.index,
                                           self.machine,
                                           self.type,
                                           self.unitidx + 1)


class TubeIn(object):
    def __init__(self, name):
        """
        Represents a one-way communication between two Nodes.
        Node A sends an object to Node B and B waits until it receives the
        object.
        Objects are stored in a FIFO queue. This object is *stored on the
        target* (Node B).

        The name is used by the source to identify the tube on the target Node.
        """
#        log_debug('creating Tube <%s>' % name)
        self.name = name
        self._queue = None

    def get_queue(self):
        if self._queue is None:
            self._queue = Queue()
        return self._queue

    queue = property(get_queue)

    def pop(self):
        """
        Removes and returns an item from the queue. If the queue is empty,
        blocks
        until it is not.
        """
#        return self.child_conns[self.unitidx].recv()
        log_debug("popping tube <%s>..." % self.name)
        data = self.queue.get(True)
        log_debug("tube <%s> popped!" % self.name)
        return data

    def local_push(self, obj):
        """
        Puts an object into the queue. Not blocking.
        Is called remotely by the source machine.
        """
        log_debug('local pushing tube <%s>' % self.name)
        self.queue.put(obj)

    def empty(self):
        return self.queue.empty()

    def __repr__(self):
        return "<TubeIn '%s'>" % (self.name)


class TubeOut(object):
    def __init__(self, name, to_node, task_id, this_machine):
#        log_debug('creating Tube <%s> to %s' % (name, str(to_node)))
        self.name = name
        self.to_node = to_node
        self.task_id = task_id
        self.this_machine = this_machine

    def push(self, obj):
        # different machine
        if self.to_node.machine == self.this_machine:
            log_debug('pushing tube <%s> (same machine)' % self.name)
#            while True:
            self.parent_conns[self.to_node.unitidx].send(('_push',
                                                          (self.name, obj),
                                                           {}))
#                if self.parent_conns[self.to_node.unitidx].recv():
#                    break
#                log_debug("pickling error, sending data again")
        # same machine
        else:
#            log_debug(str((self.to_node.machine, self.this_machine)))
            log_debug('pushing tube <%s> (different machine)' % self.name)

            # send distant push to the server using the second channel
            args = (self.to_node,
                    self.name,
                    obj)
            i = self.unitidx + (len(self.child_conns) / 2)
            self.child_conns[i].send(('_distant_push', args, {}))

    def __repr__(self):
        return "<TubeOut '%s' connected to %s>" % (self.name,
                                                   str(self.to_node))


class Tubes(object):
    def __init__(self, tubes_in, tubes_out):
        self.tubes_in = tubes_in
        self.tubes_out = tubes_out

    def get_tubes_in(self):
        names = self.tubes_in.keys()
        names.sort()
        return names

    def get_tubes_out(self):
        names = self.tubes_out.keys()
        names.sort()
        return names

    def pop(self, name):
#        log_debug('in'+name+str(self.tubes_in))
        obj = self.tubes_in[name].pop()
        # BUG FIX FOR LINUX
        time.sleep(.01)
        return obj

    def push(self, name, obj):
#        log_debug('out'+name+str(self.tubes_out))
        self.tubes_out[name].push(obj)
        # BUG FIX FOR LINUX
        time.sleep(.01)

    def register(self, index, child_conns, parent_conns):
        for tube in self.tubes_out.itervalues():
            tube.unitidx = index
            tube.child_conns = child_conns
            tube.parent_conns = parent_conns

    def __repr__(self):
        return '<Tubes object, tubes_in=%s, tubes_out=%s>' % \
                (str(self.tubes_in), str(self.tubes_out))


#class Result(object):
#    def __init__(self, result):
#        self.result = result


def process_serve(unitindex, child_conns, parent_conns, shared_data,
                  task_class, all_nodes, index, tubes, unit_type, do_redirect):
    """
    Process server.
    This function is called on the processes of the current machine. It
    handles task initializing and launching, and allows to poll for the task's
    status asynchronously.
    shared_data is a read-only dict in shared memory (only numpy arrays)
    index is the Node index
    """
    # allows the tubes to access the connections to other processes
    tubes.register(unitindex, child_conns, parent_conns)

    # Opens the PyCUDA context at the beginning
#    if unit_type == 'GPU':
#        set_gpu_device(unitindex)
#        if task.do_redirect:
#            sys.stdin = file(os.devnull)
#            sys.stdout = file(os.devnull)

#    if unit_type == 'GPU':
#        set_gpu_device(unitindex)

    task = task_class(index, unitindex, all_nodes, tubes,
                      shared_data, unit_type)
    thread = None
#    node = all_nodes[index]
    conn = child_conns[unitindex]

    def execute(m, a, k):
        log_debug('executing method <%s>' % m)
        if hasattr(task, m):
#            result = getattr(task, m)(*a, **k)
            try:
                result = getattr(task, m)(*a, **k)
            except:
                log_warn("An exception occurred in sub-process #%d" %
                    (unitindex))
                exc = traceback.format_exc()
                log_warn(exc)
                #log_warn("An exception occurred in the task: "+str(e))
                result = None
        else:
            log_warn('Task object has no method <%s>' % m)
            result = None
#        log_debug('execute function finished with result <%s>' % str(result))
        return result

    def start(m, a, k):
        if unit_type == 'GPU':
            set_gpu_device(unitindex)
            if do_redirect:
                sys.stdin = file(os.devnull)
                sys.stdout = file(os.devnull)
            if do_redirect is None and os.name == 'posix':
                log_warn("WARNING: specify do_redirect=True if CUDA code is \
                    not compiling. see \
                    <http://playdoh.googlecode.com/svn/docs/playdoh.html#gpu>")
        execute(m, a, k)

        # get the result
#        log_debug("get_result")
        result = execute('get_result', (), {})

        # send it to the local server
#        log_debug("sending result to the local server")
        i = unitindex + (len(child_conns) / 2)
        child_conns[i].send(('send_result', (result,), {}))

        if unit_type == 'GPU':
            close_cuda()

    while True:
        log_debug('process_serve %d waiting...' % unitindex)
#        received = conn.recv()
#        log_debug('process_serve received <%s>' % str(received))
#        try:
        # HACK: sometimes, pickling error on Linux
#        while True:
#            try:
        method, args, kwds = conn.recv()
#                conn.send(True) # OK
#                break
#            except:
#                conn.send(False) # NOT OK, send again

#        except:
#            log_warn("connection has been closed")
#            continue
        log_debug('process_serve %d received method <%s>' % (unitindex,
                                                             method))
        if method is None:
#            log_debug("SENDING FINISHED TO LOCAL SERVER")
            conn.send(None)
            break
#        elif method == '_process_status':
#            continue
#        elif method is '_join':
#            thread.join()
        elif method == '_push':
            tubes.tubes_in[args[0]].local_push(args[1])  # args = (name, obj)
#            conn.send(None)
        elif method == 'start':
            thread = Thread(target=start, args=(method, args, kwds))
#            log_debug('starting thread...')
            thread.start()
#        elif method == '_get_status':
#            if thread is None:
#                status = 'not started'
#            elif thread.is_alive():
#                status = 'running'
#            else:
#                status = 'finished'
#            log_debug('status is <%s>' % status)
#            conn.send(status)
        elif method == '_pass':
            log_debug('pass!')
            time.sleep(.2)
        else:
            result = execute(method, args, kwds)
            conn.send(result)
    log_debug('process_serve finished')


class SyncHandler(object):
    def __init__(self):
        self.tubes_in = {}
        self.tubes_out = {}
        self.thread = None
        self.result = None
        self.unitindices = []
        self.resultqueue = None
        self.pool = None
        self.clients = None

        # used to wait for process_serve before sending initialize
        self.initqueue = Queue()

    def create_tubes(self):
        for i in xrange(len(self.nodes)):
            node = self.nodes[i]
            # change the unitindex to match idle units
            node.unitidx = self.unitindices[i]
            tubes_in = {}
            tubes_out = {}
            for (name, i1, i2) in self.topology:
                if i1 == node.index:
                    to_node = [n for n in self.all_nodes if n.index == i2][0]
                    tubes_out[name] = TubeOut(name,
                                              to_node,
                                              self.task_id,
                                              self.local_machine)
                if i2 == node.index:
                    tubes_in[name] = TubeIn(name)
            self.tubes[node.unitidx] = Tubes(tubes_in, tubes_out)
            log_debug('Tubes created for node %s: %s' %
                (str(node), str(self.tubes[node.unitidx])))

    def submit(self, task_class, task_id, topology, all_nodes, local_nodes,
               type='CPU', shared_data={}, do_redirect=None):
        """
        Called by the launcher client, submits the task, the
        topology of the graph
        (with abstract nodes indexed by an integer), and a mapping
        assigning each node
        to an actual computing unit in the current machine.
        topology is a dict {pipe_name: (index1, index2)}
        mapping is a dict {index: Node}
        WARNING: Nodes in mapping should contain the Process indices
        that are idle
        """
        # If no local nodes
        if len(local_nodes) == 0:
            log_info("No nodes are being used on this machine")
            return
        self.topology = topology
        self.all_nodes = all_nodes
        self.nodes = local_nodes  # local nodes
        self.local_machine = local_nodes[0].machine
        self.type = type
        self.task_id = task_id
        self.task_class = task_class
        self.tubes = {}  # dict {unitidx: Tubes object}

        # computing the number of units to use on this computer
        unitindices = {}
        for n in self.nodes:
            unitindices[n.unitidx] = None
        self.units = len(unitindices)

        # finds idle units
        self.pool = self.pools[type]
        self.pool.globals = globals()
        # gets <self.units> idle units
        self.unitindices = self.pool.get_idle_units(self.units)
        for n in self.nodes:
            # WARNING: reassign idle units to nodes,
            # unitidx=0 ==> unitidx = first idle unit
            # Node unitidxs must be a consecutive list
#            log_debug(self.unitindices)
#            log_debug(n.unitidx)
            n.unitidx = self.unitindices[n.unitidx]

        self.create_tubes()

        # relaunch the needed units and give them the shared data
        # which must be specified in submit
        if len(shared_data) > 0:
            self.pool.restart_workers(self.unitindices, shared_data)

        # launch the process server on every unit
        for node in self.nodes:
            self.pool.set_status(node.unitidx, 'busy')
            self.pool[node.unitidx].process_serve(task_class,
                                                  self.all_nodes,
                                                  node.index,
                                                  self.tubes[node.unitidx],
                                                             type,
                                                             do_redirect)

        # now initialize can begin
        log_debug("<submit> has finished!")

        # used to wait for the task to finish
        self.resultqueue = dict([(i, Queue()) for i in self.unitindices])

        self.initqueue.put(True)

    def distant_push(self, node, tube, data):
        log_debug('distant push to %s for %s' % (str(node), str(tube)))
        self.pool.send(('_push', (tube, data), {}),
                                                unitindices=[node.unitidx],
                                                             confirm=True)

    def initialize(self, *argss, **kwdss):
        # argss[i] is a list of arguments of rank i for initialize,
        # one per node on this machine

        log_debug("Waiting for <submit> to finish...")
        self.initqueue.get()
        time.sleep(.1)

        log_info("Initializing task <%s>" % self.task_class.__name__)

        k = len(argss)  # number of non-named arguments
        keys = kwdss.keys()  # keyword arguments

        for i in xrange(len(self.unitindices)):
            args = [argss[l][i] for l in xrange(k)]
            kwds = dict([(key, kwdss[key][i]) for key in keys])
            self.pool.send(('initialize', args, kwds),
                            unitindices=[self.unitindices[i]], confirm=True)
        return self.pool.recv(unitindices=self.unitindices)

    def initialize_clients(self):
        # get the list of individual machines
        _machines = [node.machine.to_tuple() for node in self.all_nodes]
        self.distant_machines = []
        for m in _machines:
            if (m not in self.distant_machines) and \
                    (m != self.local_machine.to_tuple()):
                self.distant_machines.append(m)

        # initialize RpcClients
#        log_info("CONNECTING")
        self.clients = RpcClients(self.distant_machines,
                             handler_class=SyncHandler,
                             handler_id=self.task_id)
        self.clients.connect()
        self.clients_lock = Lock()

    def clients_push(self, to_node, name, obj):
        machine = to_node.machine.to_tuple()
        index = self.distant_machines.index(machine)
        log_debug("server: relaying subprocess for distant push \
                    to node <%s>" % str(to_node))

#        log_debug("ACQUIRING CLIENT LOCK")
        self.clients_lock.acquire()
        self.clients.set_client_indices([index])
#        log_debug("DISTANT PUSH")
        self.clients.distant_push(to_node, name, obj)
#        log_debug("RELEASING CLIENT LOCK")
        self.clients_lock.release()

    def close_clients(self):
        if self.clients is None:
            return
        self.clients.set_client_indices(None)
        self.clients.disconnect()

    def recv_from_children(self, index):
        while True:
#            log_debug("LOCAL SERVER WAITING %d" % index)
            r = self.pool.recv(unitindices=[self.pool.workers + index])[0]
#            log_debug("LOCAL SERVER RECV")# % str(r))

#            if r is None:
#                log_server("LOCAL SERVER exiting")
#                break

            result, args, kwds = r

            # distant push
            if result == '_distant_push':
                to_node, name, obj = args

                self.clients_push(to_node, name, obj)
#                client = RpcClient(machine, handler_class=SyncHandler,
#                           handler_id=task_id)
#                client.distant_push(to_node, name, obj)

            # retrieve result and quit the loop on the server
            elif result == 'send_result':
#                self.task_result = args[0]
                log_debug("server: received result from subprocesses")
                self.resultqueue[index].put(args[0])
                break

#        if index == self.unitindices[0]:
#            time.sleep(1)
#            log_info("CLEARING POOL")
#            self.clear_pool()
#            log_debug("CLOSING RPCCLIENTS")
#            self.close_clients()
#        log_debug("RECV FROM CHILDREN FINISHED")

    def start(self):
        log_info("Starting task")
        self.pool.send(('start', (), {}), unitindices=self.unitindices,
                                      confirm=True)

        self.initialize_clients()

        # listen for children subprocesses
        self.recv_thread = {}  # dict([(i, None) for i in self.unitindices])
        for i in self.unitindices:
            self.recv_thread[i] = Thread(target=self.recv_from_children,
                                         args=(i,))
            self.recv_thread[i].start()

    def clear_pool(self):
        if self.pool is None:
            return
        [self.pool.set_status(i, 'idle') for i in self.unitindices]
        # closes process_serve
        log_info("Terminating task")
        self.pool.send((None, (), {}), unitindices=self.unitindices,
                                   confirm=True)

    def get_info(self, *args, **kwds):
        """
        Obtains information about the running task
        """
        log_info("Getting task information...")

        k = len(args)  # number of non-named arguments
        keys = kwds.keys()  # keyword arguments

        for i in xrange(len(self.unitindices)):
            args = [argss[l][i] for l in xrange(k)]
            kwds = dict([(key, kwdss[key][i]) for key in keys])
            self.pool.send(('get_info', args, kwds),
                            unitindices=[self.unitindices[i]],
                            confirm=True)

        info = self.pool.recv(unitindices=self.unitindices)
        return info

#    def wait(self):
#        """
#        Once the task has been started, waits until it has finished
#        """
#        # HACK: unblocks the connection on the pool (cannot send() while
#        # another thread recv() in process_serve)
##        self.pool.send(('_pass', (), {}), unitindices=self.unitindices)
##        log_debug((self.unitindices, self.pool.workers))
#        pipes = [i+self.pool.workers for i in self.unitindices]
#        log_debug('waiting on pipes %s...' % str(pipes))
#        # unit indices of units which have not finished yet
##        leftindices = list(set(self.unitindices).difference(self.finished))
#
#        # receives "task finished" on the second channel
#        self.pool.recv(unitindices=pipes)

#    def get_status(self):
#        log_debug("Getting task status...")
#        self.pool.send(('_get_status', (), {}), unitindices=self.unitindices)
#        return self.pool.recv(unitindices=self.unitindices, discard=
#        'task finished')

    def get_result(self):
        """
        Blocks until the task has completed
        """
        # wait
        # TODO: implement WAIT with a Queue instead of a long recv in a
        # dedicated channel
#        pipes = [i+self.pool.workers for i in self.unitindices]
#        log_debug('waiting on pipes %s...' % str(pipes))
        # receive "task finished" on the second channel
#        self.pool.recv(unitindices=pipes)
#        log_debug("joining local server thread")
        if self.resultqueue is not None:
            log_debug("server: waiting for the task to finish")
            results = [self.resultqueue[i].get() \
                            for i in self.resultqueue.keys()]

#        for i in self.recv_thread.keys():
#            self.recv_thread[i].join()

        time.sleep(.1)
        self.clear_pool()
        self.close_clients()

        return results


class TaskRun(object):
    """
    Contains information about a parallel task that has been launched
    by the :func:`start_task` function.

    Methods:

    ``get_info()``
        Returns information about the current run asynchronously.

    ``get_result()``
        Returns the result. Blocks until the task has finished.
    """
    def __init__(self, task_id, type, machines, nodes, args, kwds):
        self.task_id = task_id
        self.nodes = dict([(node.index, node) for node in nodes])
        self.machines = machines  # list of Machine object
        self.type = type
        self._machines = [m.to_tuple() for m in self.machines]
        self.local = None
        # arguments of the task initialization
        self.args, self.kwds = args, kwds

    def set_local(self, v):
        log_debug("setting local to %s" % str(v))
        self.local = v

    def get_machines(self):
        return self._machines

    def get_machine_index(self, machine):
        for i in xrange(len(self.machines)):
            if (self.machines[i] == machine):
                return i

    def concatenate(self, l):
        """
        Concatenates an object returned by RpcClients to a list
        adapted to the list
        of nodes
        """
        newl = {}
        for i, n in self.nodes.iteritems():
            machine_index = self.get_machine_index(n.machine)
            unit_index = n.unitidx
            newl[i] = l[machine_index][unit_index]
        newl = [newl[i] for i in xrange(len(self.nodes))]
        return newl

    def get_info(self):
        GC.set(self.get_machines(), handler_class=SyncHandler,
                                 handler_id=self.task_id)
        disconnect = GC.connect()
        log_info("Retrieving task info")
        info = GC.get_info()
        if disconnect:
            GC.disconnect()
        return self.concatenate(info)

    def get_result(self):
        if not self.local:
            time.sleep(2)
        else:
            time.sleep(.1)
        GC.set(self.get_machines(), handler_class=SyncHandler,
                                 handler_id=self.task_id)
        disconnect = GC.connect()
        if not self.local:
            log_info("Retrieving task results")
        result = GC.get_result()
        if disconnect:
            GC.disconnect()
        result = self.concatenate(result)
        if self.local:
            close_servers(self.get_machines())
        return result

    def __repr__(self):
        units = len(self.nodes)
        if units > 1:
            plural = 's'
        else:
            plural = ''
        return "<Task '%s' on %d machines and %d %s%s>" % (self.task_id,
                                                          len(self.machines),
                                                          len(self.nodes),
                                                          self.type,
                                                          plural)


class ParallelTask(object):
    """
    The base class from which any parallel task must derive.

    Three methods must be implemented:

    ``initialize(self, *args, **kwds)``
        Initialization function, with any arguments and keyword arguments,
        which
        are specified at runtime in the :func:`start_task` function.

    ``start(self)``
        Start the task.

    ``get_result(self)``
        Return the result.

    One method can be implemented.

    ``get_info(self)``
        Return information about the task. Can be called asynchronously at
        any time
        by the client, to obtain for example the current iteration number.

    Two methods from the base class are available:

    ``push(self, name, data)``
        Put some ``data`` into the tube ``name``. Named tubes are
        associated to a single source
        and a single target. Only the source can call this method.
        Note that several tubes in the network can have
        the same name, but two tubes entering or exiting a given
        node cannot have the same name.

    ``pop(self, name)``
        Pop data in the tube ``name``: return the first item in
        the tube (FIFO queue) and remove it.
        If the tube is empty, block until the source put some
        data into it. The call to this method
        is equivalent to a synchronisation barrier.

    Finally, the following read-only attributes are available:

    ``self.index``
        The index of the current node, between 0 and n-1 if there
        are n nodes in the network.

    ``self.unitidx``
        The index of the CPU or GPU on the machine running the
        current node.

    ``self.shared_data``
        The shared data dictionary (see :ref:`shared_data`).

    ``self.unit_type``
        The unit type on this node, ``'CPU'`` or ``'GPU'``.

    ``self.tubes_in``
        The list of the incoming tube names on the current node.

    ``self.tubes_out``
        The list of the outcoming tube names on the current node.
    """
    def __init__(self, index, unitidx, nodes, tubes, shared_data, unit_type):
        self.index = index
        self.unitidx = unitidx
        self.nodeidx = index
        self.nodes = nodes
        self.node = nodes[index]
        self.tubes = tubes
        self.shared_data = shared_data
        self.unit_type = unit_type
        self.tubes_in = self.tubes.get_tubes_in()
        self.tubes_out = self.tubes.get_tubes_out()

    def pop(self, name):
        return self.tubes.pop(name)

    def push(self, name, data):
        self.tubes.push(name, data)

    def initialize(self):
        log_warn("The <initialize> method of a parallel task may \
be implemented")

    def start(self):
        log_warn("The <start method> of a parallel task must be \
implemented")

    def get_result(self):
        """
        Default behavior: returns self.result
        """
        return self.result

    def get_info(self):
        log_warn("The <get_info> method of a parallel task may be \
implemented")
