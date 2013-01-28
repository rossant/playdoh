"""
Native Python RPC Layer
"""
from debugtools import *
from codehandler import *
from connection import *
from userpref import *
from subprocess import Popen, PIPE
import threading
import os
import time
from Queue import Queue

__all__ = ['DEFAULT_PORT', 'BaseRpcServer', 'BaseRpcClient', 'BaseRpcClients',
           'open_base_server', 'close_base_servers',
           'open_restart_server', 'restart'
#           'DistantException'
           ]

DEFAULT_PORT = USERPREF['port']


def open_base_server(port=None):
    BaseRpcServer(port=port).listen()


def close_base_servers(addresses):
    if type(addresses) is str:
        addresses = [(addresses, DEFAULT_PORT)]
    if type(addresses) is tuple:
        addresses = [addresses]
    BaseRpcClients(addresses).close_server()


#class DistantException(Exception):
#    """
#    Distant Exception class. Allows to re-raise exception on the client,
#    giving the filename and the line number of the original server-side
#    exception.
#    """
#    def __init__(self, exception = None):
#        if type(exception) == str:
#            exception = Exception(exception)
#        self.exception = exception
#        try:
#            self.filename = __file__
#            if os.path.splitext(__file__)[1] == '.pyc':
#                self.filename = __file__[:-1]
#        except:
#            self.filename = 'unknown'
#        try:
#            self.line = sys._getframe(1).f_lineno
#        except:
#            self.line = 0
#        try:
#            self.function = sys._getframe(1).f_code.co_name
#        except:
#            self.function = 'unknown'
#
#    def setinfo(self):
#        (self.filename, self.line, self.function, self.text) =
#               traceback.extract_tb(sys.exc_info()[2])[-1]
#
#    def __str__(self):
#        s = "A distant exception happened in:"
#        s += "\n  File \"%s\", line %d, in %s" % (self.filename,
#           self.line, str(self.function))
#        s += "\n    %s" % str(self.exception)
#        return s


class BaseRpcServer(object):
    def __init__(self, port=None, bindip=''):
        """
        to be implemented by a deriving class:
        initialize()
        process(client, procedure)
        shutdown()
        """
        if port is None:
            port = DEFAULT_PORT
        self.port = port
        self.address = (bindip, self.port)
        self.bool_shutdown = False

        # HACK: to fix windows bug: the server must not accept while
        # restarting subprocesses
        self.temp_result = None
        self.wait_before_accept = False
        self.acceptqueue = Queue()

    def serve(self, conn, client):
        # called in a new thread
        keep_connection = None  # allows to receive several procedures
                                # during the same session
        # None : close connection at the next iteration
        # False : close connection now
        # True : keep connection for now

        while keep_connection is not False:
            log_debug("server: serving client <%s>..." % str(client))
            procedure = conn.recv()
            log_debug("server: procedure '%s' received" % procedure)

            if procedure == 'keep_connection':
                keep_connection = True
                continue  # immediately waits for a procedure
            elif procedure == 'close_connection':
                keep_connection = False
                break  # close connection
            elif procedure == 'shutdown':
                log_debug("server: shutdown signal received")
                keep_connection = False
                self.bool_shutdown = True
                break  # closes the connection immediately
            elif procedure == 'get_temp_result':
                log_debug("sending temp result")
                conn.send(self.temp_result)
                self.temp_result = None
                keep_connection = False
                break

            # Mechanism to close the connection while processing a procedure
            # used to fix a bug: Processes shouldn't be started on Windows
            # while a connection is opened
            if (hasattr(procedure, 'close_connection_temp') and
                    procedure.close_connection_temp):
                log_debug("closing connection while processing procedure %s" %
                    str(procedure))
                self.wait_before_accept = True
#                conn.close()
#                conn = None

            # Dispatches the procedure to the handler unless the procedure
            # asks the server to close the handler or itself
            log_debug("server: processing procedure")
#            if procedure is not None:
            result = self.process(client, procedure)
#            else:
#                log_debug("Connection error happened, exiting")
#                result = None
#                break

            if (hasattr(procedure, 'close_connection_temp') and
                    procedure.close_connection_temp):
                self.temp_result = result
                # make sure that the accepting thread is waiting now
                time.sleep(.1)
                self.wait_before_accept = False
                self.acceptqueue.put(True)
                keep_connection = False
            else:
                log_debug("server: returning the result to the client")
                conn.send(result)

            if keep_connection is None:
                keep_connection = False

        if conn is not None:
            conn.close()
            conn = None
        log_debug("server: connection closed")

    def listen(self):
        """
        Listens to incoming connections and create one handler for each
        new connection.
        """
        conn = None
        threads = []
        index = 0

        # Initializing
        log_debug("Initializing server with IP %s on port %d" % (LOCAL_IP,
                                                                 self.port))
        self.initialize()

        while not self.bool_shutdown:
            try:
                log_debug("server: waiting for incoming connection on port \
                    %d..." % self.address[1])
                if self.wait_before_accept:
                    log_debug("waiting before accepting a connection...")
#                    time.sleep(USERPREF['wait_before_accept'])
                    self.acceptqueue.get()
                    log_debug("I can accept now!")
                conn, client = accept(self.address)  # accepts an incoming
                                                     # connection
                log_debug("Server established a connection with client %s on \
                           port %d" % (client, self.address[1]))
            except:
                log_warn("server: connection NOT established, closing now")
                break

            thread = threading.Thread(target=self.serve, args=(conn, client))
            thread.start()
            threads.append(thread)
            index += 1
            time.sleep(.1)

        # Closes the connection at the end of the server lifetime
        if conn is not None:
            conn.close()
            conn = None

        # closing
        log_debug("Closing server")
        self.shutdown()

        for i in xrange(len(threads)):
            log_debug("joining thread %d/%d" % (i + 1, len(threads)))
            threads[i].join(.1)

    def initialize(self):
        log_warn("server: 'initialize' method not implemented")

    def process(self, client, procedure):
        log_warn("server: 'process' method not implemented")
        return procedure

    def shutdown(self):
        log_warn("server: 'shutdown' method not implemented")


class BaseRpcClient(object):
    """
    RPC Client constructor.
    """
    def __init__(self, server):
        self.conn = None
        if type(server) is tuple:
            server, port = server
        else:
            port = None
        self.server = server
        if port is None:
            port = DEFAULT_PORT
        self.port = port
        self.keep_connection = False

    def open_connection(self, trials=None):
        log_debug("client: connecting to '%s' on port %d" % (self.server,
                                                             self.port))
        try:
            self.conn = connect((self.server, self.port), trials)
        except:
            log_warn("Error when connecting to '%s' on port %d" % (self.server,
                                                                   self.port))
            self.conn = None

    def close_connection(self):
        log_debug("client: closing connection")
        if self.conn is not None:
            self.conn.close()
        self.conn = None

    def is_connected(self):
        return self.conn is not None

    def execute(self, procedure):
        """
        Calls a procedure on the server from the client.
        """
        if not self.is_connected():
            self.keep_connection = False
            self.open_connection()

        log_debug("client: sending procedure '%s'" % procedure)
        try:
            self.conn.send(procedure)
        except:
            log_warn("client: connection lost while sending the procedure,\
                      connecting again...")
            self.open_connection()
            self.conn.send(procedure)

        # handling special mechanism to close the connection while processing
        # the procedure
        if (hasattr(procedure, 'close_connection_temp') and
                procedure.close_connection_temp):
            log_debug("closing the connection while processing the procedure")
            self.close_connection()
            log_debug("waiting...")
            time.sleep(.5)
            log_debug("opening the connection again")
            self.open_connection()
            self.conn.send("get_temp_result")
            log_debug("receiving temp result")
            result = self.conn.recv()
            log_debug("temp result received! closing connection")
            self.close_connection()
            log_debug("connection closed()")
        else:
            log_debug("client: receiving result")
            try:
                result = self.conn.recv()
            except:
                log_warn("client: connection lost while retrieving the result,\
                          connecting again...")
                self.open_connection()
                result = self.conn.recv()

            if not self.keep_connection:
                self.close_connection()

        # re-raise the Exception on the client if one was raised on the server
        if isinstance(result, Exception):
            raise result
        # Handling list of exceptions
#        if type(result) is list:
#            raise result[0]
#            exceptions = []
#            for r in result:
#                if isinstance(r, Exception):
#                    exceptions.append(str(r))
#            if len(exceptions)>0:
#                raise Exception("\n".join(exceptions))

        return result

    def connect(self, trials=None):
        self.keep_connection = True
        self.open_connection(trials)
        if self.conn is not None:
            self.conn.send('keep_connection')

    def disconnect(self):
        self.keep_connection = False
        if self.conn is not None:
            self.conn.send('close_connection')
        self.close_connection()

    def close_server(self):
        """
        Closes the server from the client.
        """
        if not self.is_connected():
            self.open_connection()
        log_debug("client: sending the shutdown signal")
        self.conn.send('shutdown')
        self.close_connection()


class BaseRpcClients(object):
    def __init__(self, servers):
        self.servers = servers
        self.clients = [BaseRpcClient(server) for server in servers]
        self.results = {}

    def open_threads(self, targets, argss=None, indices=None):
        if indices is None:
            indices = xrange(len(self.servers))
        if argss is None:
            argss = [()] * len(indices)
        threads = []
        for i in xrange(len(indices)):
            thread = threading.Thread(target=targets[i],
                                      args=argss[i])
            thread.start()
            threads.append(thread)
        [thread.join() for thread in threads]

    def _execute(self, index, procedure):
        log_debug("_execute %s" % str(procedure))
#        if self.clients[index].is_connected():
        self.results[index] = self.clients[index].execute(procedure)
#        else:
#            log_warn("The connection to client %d has been lost" % index)
#            self.results[index] = None

    def execute(self, procedures, indices=None):
        """
        Makes simultaneously (multithreading) several calls to different
        RPC servers.
        """
        if indices is None:
            indices = xrange(len(self.servers))
        results = []
        self.exceptions = []
        self.open_threads([self._execute] * len(indices),
                          [(indices[i], procedures[i]) for i in
                                xrange(len(indices))],
                          indices)

#        if len(self.exceptions)>0:
#            raise Exception("\n".join(self.exceptions))
        for i in indices:
            if i in self.results.keys():
                result = self.results[i]
            else:
                result = None
            results.append(result)
        return results

    def is_connected(self):
        return [c.is_connected() for c in self.clients]

    def connect(self, trials=None):
        self.open_threads([client.connect for client in self.clients],
                          argss=[(trials,)] * len(self.clients))

    def disconnect(self):
        self.open_threads([client.disconnect for client in self.clients])

    def close_server(self):
        self.open_threads([client.close_server for client in self.clients])


def open():
    import playdoh
    scriptdir = os.path.join(os.path.dirname(playdoh.__file__), 'scripts')
    os.chdir(scriptdir)
    cpu = playdoh.MAXCPU
    gpu = playdoh.get_gpu_count()
    port = playdoh.DEFAULT_PORT
    os.system("python open.py %d %d %d" % (cpu, gpu, port))


def kill_linux():
    """
    Kill all Python processes except the RestartRpcServer
    """
    pid = str(os.getpid())
    cmd = "ps -ef | grep python | awk '{print $2}'"
    r = commands.getoutput(cmd)
    r = r.split('\n')
    if pid in r:
        r.remove(pid)
    n = len(r)
    r = ' '.join(r)
    cmd = 'kill %s' % r
    os.system(cmd)
    return n


def kill_windows():
    pid = str(os.getpid())
    cmd = ['tasklist', "/FO", "LIST", "/FI", "IMAGENAME eq python.exe"]
    r = Popen(cmd, stdout=PIPE).communicate()[0]
    r = r.split("\n")
    ps = []
    for line in r:
        if line[:3] == 'PID':
            ps.append(line[4:].strip())
    if pid in ps:
        ps.remove(pid)
    n = len(ps)
    args = " ".join(["/pid %s" % p for p in ps])
    cmd = "taskkill /F %s" % args
    os.system(cmd)
    return n


def kill():
    if os.name == 'posix':
                n = kill_linux()
    else:
        n = kill_windows()
    return n


class RestartRpcServer(BaseRpcServer):
    def initialize(self):
        log_info("waiting to kill...")
        self.thread = None

    def _open(self):
        open()

    def open(self):
        self.thread = threading.Thread(target=self._open)
        self.thread.start()

    def process(self, client, procedure):
        n = None
        if procedure == 'kill':
            n = kill()
            log_info("%d processes killed" % n)
        elif procedure == 'open':
            n = None
            self.open()
        elif procedure == 'restart':
            n = kill()
            log_info("%d processes killed" % n)
            self.open()
        return n

    def shutdown(self):
        pass


def open_restart_server():
    RestartRpcServer(port=27182).listen()


def restart(server=None, procedure=None):
    if server is None:
        server = 'localhost'
    if procedure is None:
        procedure = 'restart'
    c = BaseRpcClient((server, 27182))
    c.execute(procedure)
