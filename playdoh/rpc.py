from debugtools import *
from gputools import *
from pool import *
from connection import *
from baserpc import *
from numpy import array
import time
import traceback


__all__ = ['RpcServer', 'RpcClient', 'RpcClients',
           'Machine', 'GlobalConnection', 'GC',
           'Procedure', 'open_server', 'close_servers']


def open_server(port=None, maxcpu=None, maxgpu=None, local=None):
    """
    Start the Playdoh server.

    Arguments:

    ``port=DEFAULT_PORT``
        The port (integer) of the Playdoh server. The default is DEFAULT_PORT,
        which is 2718.

    ``maxcpu=MAXCPU``
        The total number of CPUs the Playdoh server can use. ``MAXCPU`` is the
        total number of CPUs on the computer.

    ``maxgpu=MAXGPU``
        The total number of GPUs the Playdoh server can use. ``MAXGPU`` is the
        total number of GPUs on the computer, if PyCUDA is installed.
    """
    if maxcpu is not None:
        globals()['MAXCPU'] = maxcpu
    else:
        maxcpu = MAXCPU
    if maxgpu is not None:
        globals()['MAXGPU'] = maxgpu
    else:
        maxgpu = get_gpu_count()
    if port is None:
        port = DEFAULT_PORT
    if not local:
        log_info("Opening Playdoh server on port %d with %d CPU(s) and %d \
GPU(s)" % (port, maxcpu, maxgpu))
    RpcServer(port=port).listen()


def close_servers(addresses):
    """
    Close the specified Playdoh server(s) remotely.

    Arguments:

    ``addresses``
        The list of the Playdoh server addresses to shutdown.
    """
    if type(addresses) is str:
        addresses = [(addresses, DEFAULT_PORT)]
    if type(addresses) is tuple:
        addresses = [addresses]
    RpcClients(addresses).close_server()


class GlobalConnection(object):
    def __init__(self, servers=None, handler_class=None, handler_id=None):
        self.clients = None
        self.set(servers, handler_class, handler_id)

    def set(self, servers=None, handler_class=None, handler_id=None):
        if type(servers) is not list:
            servers = [servers]
        self.servers = servers
        self.handler_class = handler_class
        self.handler_id = handler_id
        if self.clients is not None:
            self.clients.handler_class = handler_class
            self.clients.handler_id = handler_id

    def connect(self, raiseiferror=False, trials=None):
        """
        Connects if needed and returns True if the caller must disconnect
        manually
        """
        if array(self.connected).all():
            return False
        log_debug("Connecting to %d server(s), class=%s, id=<%s>" % \
                    (len(self.servers),
                     self.handler_class,
                     self.handler_id))
        self.clients = RpcClients(self.servers,
                                  self.handler_class,
                                  self.handler_id)
        self.clients.connect(trials)
        boo = array(self.clients.is_connected()).all()
        if raiseiferror:
            if not boo:
                raise Exception("Connection error")
        return boo

    def is_connected(self):
        return (self.clients != None) and (self.clients.is_connected())
    connected = property(is_connected)

    def disconnect(self):
        log_debug("Disconnecting from servers")
        if self.clients is not None:
            self.clients.disconnect()
        self.clients = None

    def execute_native(self, method, *args, **kwds):
        return self.clients.execute_native(method, *args, **kwds)

    def add_handler(self, handler_id, handler_class):
        self.handler_id = handler_id
        self.handler_class = handler_class
        self.clients.add_handler(handler_id, handler_class)

    def __getattr__(self, name):
        return getattr(self.clients, name)

GC = GlobalConnection()


class Machine(object):
    def __init__(self, arg, port=DEFAULT_PORT):
        """
        Represents a machine.

        arg can be a string IP or a tuple (IP, port)
        """
        if type(arg) is str:
            self.ip = arg
            self.port = port
        elif type(arg) is tuple:
            self.ip, self.port = arg
        elif type(arg) is Machine:
            self.ip = arg.ip
            self.port = arg.port
        # HACK
        if self.ip == 'localhost':
            self.ip = LOCAL_IP

    def to_tuple(self):
        return (self.ip, self.port)

    def __repr__(self):
        return "<machine '%s' on port %d>" % (self.ip, self.port)

    def __eq__(self, y):
        return (self.ip == y.ip) and (self.port == y.port)


class DistantException(object):
    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return self.msg


class Procedure(object):
    """
    A procedure sent by a client to a server.
    object handler_id, instance of handler_class, call of
    method_name(*args, **kwds)
    """
    def __init__(self, handler_class, handler_id, name, args, kwds):
        self.handler_class = handler_class
        if handler_class is not None:
            self.handler_class_name = handler_class.__name__
        else:
            # handler_class = None means that the method is a method of
            # the RpcServer not a method of handler_class
            self.handler_class_name = 'native'
        if args is None:
            args = ()
        if kwds is None:
            kwds = {}
        self.handler_id = handler_id
        self.name = name
        self.args = args
        self.kwds = kwds

    def is_native(self):
        return self.handler_class_name == 'native'

    def __repr__(self):
        return "%s.%s" % (self.handler_class_name, self.name)

#def get_max_gpu(conn):
#    MAXGPU = initialise_cuda()
#    close_cuda()
#    conn.send(MAXGPU)
#    conn.close()
#    return MAXGPU


class RpcServer(BaseRpcServer):
    def initialize(self):
        global MAXGPU
#        if CANUSEGPU and type(MAXGPU) is not int:
##            MAXGPU = initialise_cuda()
#            # can't initialize CUDA before working, otherwise bugs appear
#            # on Linux
#            # so initializing CUDA in a separated process just to retrieve
             # the number of GPUs!
#            parent_conn, child_conn = Pipe()
#            p=multiprocessing.Process(target=get_max_gpu, args=(child_conn,))
#            p.start()
#            MAXGPU=parent_conn.recv()
##            log_debug("found %d" % MAXGPU)
#            p.join()
        if type(MAXGPU) is not int:
            MAXGPU = get_gpu_count()

        self.pools = {}
        # when creating a CustomPool, set its pool property to this if using
        # CPUs
        self.pools['CPU'] = CustomPool.create_pool(MAXCPU)
        if CANUSEGPU:
            self.pools['GPU'] = CustomPool.create_pool(MAXGPU)
        else:
            self.pools['GPU'] = None

        # resources currently assigned to every client
        self.resources = {}
        self.resources['CPU'] = {}
        if CANUSEGPU:
            self.resources['GPU'] = {}
        else:
            self.resources['GPU'] = None

        # total number of resources available on the server
        self.total_resources = {}
        self.total_resources['CPU'] = MAXCPU
        if CANUSEGPU:
            self.total_resources['GPU'] = MAXGPU
        else:
            self.total_resources['GPU'] = 0

        self.handlers = {}
        # dict handler_id => client associated to the handler
        self.handler_clients = {}

    def get_client_name(self, client):
        return client

    def request_resources(self, client, **resources):
        """
        Allocates units type to client
        type is CPU or GPU
        """
        for type, units in resources.iteritems():
            if units is None:
                continue
            if self.resources[type] is not None:
                res = min(units, self.total_resources[type])
                log_info("Assigning %d %s(s) to client '%s'" % (res, type,
                                                                client))
                self.resources[type][client] = res
#                self.resources[type][client] = units
            else:
                log_warn("no %s(s) are available on this server" % type)

    def get_all_resources(self, client):
        """
        resources['CPU']['client'] = nbr
        """
        return self.resources

    def set_total_resources(self, client, **total_resources):
        total = dict(CPU=get_cpu_count(), GPU=get_gpu_count())
        for type, units in total_resources.iteritems():
            units = min(units, total[type])
            log_info("Changing the total number of resources available on \
the server: %d %s(s)" % (units, type))
            self.total_resources[type] = units

        # changing allocated resources
        for typ in self.resources.keys():
            if self.resources[typ] is not None:
                for client in self.resources[typ].keys():
                    self.resources[typ][client] = \
                        min(self.total_resources[typ],
                            self.resources[typ][client])
        for handler in self.handlers.itervalues():
            handler.resources = self.resources

    def get_total_resources(self, client):
        """
        Returns the total number of resources available on the server
        for every type of resource
        """
        return self.total_resources

    def add_handler(self, client, handler_id, handler_class):
        log_debug("server: creating new handler '%s', '%s' instance, for\
                    client '%s'" % (handler_id, handler_class, client))
        self.handlers[handler_id] = handler_class()  # Object initialization
        self.handlers[handler_id].client = client  # client IP
        self.handlers[handler_id].this_machine = Machine(LOCAL_IP, self.port)
        self.handlers[handler_id].handler_id = handler_id
        self.handlers[handler_id].pools = self.pools  # the handler objects can
                                    # access to the global pools

        for typ in self.resources.keys():
            # assigning MAXtyp units by default
            if self.resources[typ] is not None and \
                    client not in self.resources[typ]:
                log_debug("Default assignment of %d %s(s) to client '%s'" %
                    (self.total_resources[typ], typ, client))
                self.resources[typ][client] = self.total_resources[typ]

        self.handlers[handler_id].resources = self.resources

        # handler_id corresponds to client
        self.handler_clients[handler_id] = client
#        self.update_resources()

    def delete_handler(self, client, handler_id):
        if handler_id is None:
            handler_id = client
        log_debug("server: deleting handler '%s'" % handler_id)
        del self.handlers[handler_id]
        del self.handler_clients[handler_id]

    def process(self, client, procedure):
        if client == '127.0.0.1':
            client = LOCAL_IP
        if procedure.is_native():
            # call self.name(*args, **kwds)
#            try:
            result = getattr(self, procedure.name)(client, *procedure.args,
                                                   **procedure.kwds)
#            except Exception as e:
#                log_warn("The procedure '%s' is not valid" % procedure.name)
#                result = e
        else:
            # Default handler id is the client IP
            if procedure.handler_id is None:
                procedure.handler_id = client
            # creates the handler if it doesn't exist
            if procedure.handler_id not in self.handlers.keys():
                self.add_handler(client, procedure.handler_id,
                                 procedure.handler_class)
            # call self.handlers[id](*args, **kwds)
            try:
                result = getattr(self.handlers[procedure.handler_id],
                                               procedure.name)(*procedure.args,
                                                              **procedure.kwds)
            except:
                msg = traceback.format_exc()
                log_warn(msg)
                result = DistantException(msg)
        return result

    def restart(self, client):
        log_info("Restarting the server")
        self.shutdown()
        time.sleep(.5)
        self.initialize()

    def shutdown(self):
        for id, handler in self.handlers.iteritems():
            if hasattr(handler, 'close'):
                log_debug("closing handler '%s'" % id)
                handler.close()
        for type, pool in self.pools.iteritems():
            if pool is not None:
                log_debug("closing pool of %s" % type)
                pool.close()


class RpcClient(BaseRpcClient):
    def __init__(self, server, handler_class=None, handler_id=None):
        BaseRpcClient.__init__(self, server)
        self.handler_class = handler_class
        self.handler_id = handler_id

    def execute_method(self, handler_class, handler_id, method,
                       *args, **kwds):
        procedure = Procedure(handler_class, handler_id, method,
                              args, kwds)
        result = self.execute(procedure)
        if type(result) is DistantException:
            print result.msg
            raise result
        return result

    def execute_native(self, method, *args, **kwds):
        return self.execute_method(None, None, method,
                                   *args, **kwds)

    def set_handler_id(self, handler_id):
        log_debug("client: setting handler id='%s'" % str(handler_id))
        self.handler_id = handler_id

    def set_handler_class(self, handler_class):
        log_debug("client: setting handler class='%s'" % str(handler_class))
        self.handler_class = handler_class

    def add_handler(self, handler_id, handler_class=None):
        self.set_handler_id(handler_id)
        if handler_class is not None:
            self.set_handler_class(handler_class)
        log_debug("client: adding handler '%s'" % str(handler_id))
        return self.execute_native('add_handler', self.handler_id,
                                   self.handler_class)

    def delete_handler(self, handler_id=None):
        log_debug("client: deleting handler '%s'" % str(handler_id))
        return self.execute_native('delete_handler', handler_id)

    def restart(self):
        self.execute_native('restart')

    def __getattr__(self, method):
        log_debug("getting attribute '%s'" % method)
        return lambda *args, **kwds: self.execute_method(self.handler_class,
                                                         self.handler_id,
                                                         method,
                                                         *args,
                                                         **kwds)


class RpcClients(BaseRpcClients):
    def __init__(self, servers, handler_class=None, handler_id=None):
        if type(servers) is str:
            servers = [servers]
        if type(servers) is tuple:
            servers = [servers]
        self.servers = servers
        self.handler_class = handler_class
        self.handler_id = handler_id

        self.clients = [RpcClient(server, handler_class, handler_id) \
                            for server in servers]
        self.results = {}

        self.indices = None

    def distribute(self, handler_class, handler_id, name, *argss, **kwdss):
        if self.indices is None:
            self.indices = xrange(len(self.servers))

        # True if the connection must be closed on the server side while
        # processing the procedure
        close_connection_temp = kwdss.pop('_close_connection_temp', False)

        procedures = []
        k = len(argss)  # number of non-named arguments
        keys = kwdss.keys()  # keyword arguments

        # Duplicates non-list args
        argss = list(argss)
        for l in xrange(k):
            if type(argss[l]) is not list:
                argss[l] = [argss[l]] * len(self.indices)

        # Duplicates non-list kwds
        for key in keys:
            if type(kwdss[key]) is not list:
                kwdss[key] = [kwdss[key]] * len(self.indices)

        for i in xrange(len(self.indices)):
            args = [argss[l][i] for l in xrange(k)]
            kwds = dict([(key, kwdss[key][i]) for key in keys])
            procedure = Procedure(handler_class, handler_id, name,
                                  args, kwds)

            # close conn on the server while processing the procedure
            procedure.close_connection_temp = close_connection_temp

            procedures.append(procedure)

        return self.execute(procedures, indices=self.indices)

    def set_client_indices(self, indices):
        """
        Set the client indices to connect to. None by default = connect
        to all clients
        """
        self.indices = indices

    def execute_native(self, method, *argss, **kwdss):
        return self.distribute(None, None, method, *argss, **kwdss)

    def add_handler(self, handler_id, handler_class=None):
        self.handler_id = handler_id
        self.handler_class = handler_class
        return self.distribute(None, None, 'add_handler',
                               [handler_id] * len(self.servers),
                               [handler_class] * len(self.servers))

    def delete_handler(self, handler_id=None):
        if handler_id is None:
            handler_id = self.handler_id
        return self.distribute(None, None, 'delete_handler',
                               [handler_id] * len(self.servers))

    def __getattr__(self, name):
        return lambda *argss, **kwdss: self.distribute(self.handler_class,
                                                       self.handler_id,
                                                       name,
                                                       *argss,
                                                       **kwdss)
