from debugtools import *
from pool import MAXCPU
from gputools import *
from baserpc import DEFAULT_PORT, LOCAL_IP
from rpc import *
from multiprocessing import Process
import numpy


def get_server_resources(servers):
    """
    Get the complete resource allocation on the specified servers.

    Arguments:

    ``servers``
        The list of the servers. Every item is a string with the IP address of
        the server, or a tuple ``(IP, port)``.

    Return an object ``resources``, which is a list of dictionaries that can be
    used like this:
    ``nbr = resources[serverindex][type][client]``, where:

    * ``serverindex`` is the index of the server in the ``servers`` argument,
    * ``type is ``'CPU'`` or ``'GPU'``,
    * ``client`` is the IP address of the client, or ``ME`` if it corresponds
      to this client,
      i.e. the computer which made the call to ``get_server_resources``,
    * ``nbr`` is the number of ``type``(s) allocated to ``client`` on server
      ``serverindex``.
    """
    if type(servers) is not list:
        servers = [servers]
    GC.set(servers)
    disconnect = GC.connect(raiseiferror=True)
    client_name = GC.get_client_name()
    result = GC.execute_native('get_all_resources')
    for i in xrange(len(result)):
        # replaces my IP by 'ME'
        for typ in result[i].keys():
            if result[i][typ] is not None and \
                    client_name[i] in result[i][typ].keys():
                result[i][typ]['ME'] = result[i][typ][client_name[i]]
                del result[i][typ][client_name[i]]
    if disconnect:
        GC.disconnect()
    return result


def get_my_resources(servers):
    """
    Get the resources allocated to this client on the specified servers.

    Arguments:

    ``servers``
        The list of the servers. Every item is a string with the IP address
        of the server,
        or a tuple ``(IP, port)``.

    Return an object ``resources``, which is a dictionary where keys are
    ``'CPU'`` or ``'GPU'``,
    and values are the number of allocated resources for this client.
    """
    result = get_server_resources(servers)
    for r in result:
        for typ, res in r.iteritems():
            if res is not None and 'ME' in res.keys():
                r[typ] = res['ME']
            else:
                r[typ] = 0
    return result


def get_available_resources(servers):
    """
    Get the total number of (potentially) available resources for this client
    on the specified servers, i.e. the number of idle resources (allocated
    to no one)
    plus the resources already allocated to this client.

    Arguments:

    ``servers``
        The list of the servers. Every item is a string with the IP address of
        the server,
        or a tuple ``(IP, port)``.

    Return an object ``resources``, which is a dictionary where keys are
    ``'CPU'`` or ``'GPU'``,
    and values are the number of idle resources for this client.
    """
    if type(servers) is not list:
        servers = [servers]
    GC.set(servers)
    disconnect = GC.connect(raiseiferror=True)
    resources = get_server_resources(servers)
    total_resources = get_total_resources(servers)
    idle_resources = [None] * len(servers)
    for i in xrange(len(total_resources)):
        idle_resources[i] = {}
        for r in ['CPU', 'GPU']:
            if resources[i][r] is None:
                idle_resources[i][r] = 0
            else:
                idle_resources[i][r] = total_resources[i][r] - \
                    sum(resources[i][r].values())
                if 'ME' in resources[i][r].keys():
                    idle_resources[i][r] += resources[i][r]['ME']
    if disconnect:
        GC.disconnect()
    return idle_resources


def request_resources(servers, **resources):
    """
    Allocate resources for this client on the specified servers.

    Arguments:

    ``servers``
        The list of the servers. Every item is a string with the IP address
        of the server,
        or a tuple ``(IP, port)``.

    ``**resources``
        A dictionary where keys are ``'CPU'` or ``'GPU'`` and values are lists
        with the number of
        CPUs or GPUs to allocate on each server.

    Example: ``request_resources('bobs-machine.university.com', CPU=2)``
    """
    if type(servers) is not list:
        servers = [servers]
    for k, v in resources.iteritems():
        if type(resources[k]) is not list:
            resources[k] = [v]
    GC.set(servers)
    disconnect = GC.connect(raiseiferror=True)
    result = GC.execute_native('request_resources', **resources)
    if disconnect:
        GC.disconnect()
    return result


def request_all_resources(servers, type, skip=[], units=None):
    """
    Allocate resources optimally for this client on the specified servers, i.e.
    as many resources as possible.

    Arguments:

    ``servers``
        The list of the servers. Every item is a string with the IP address of
        the server,
        or a tuple ``(IP, port)``.

    ``type``
        The unit type: ``'CPU'` or ``'GPU'``.

    Return a list with the number of resources that just have been allocated
    for every server.

    Example: ``n = request_all_resources('bobs-machine.university.com',
    type='CPU')[0]``
    """
    # skip is a list of server indices to skip
    if __builtins__['type'](servers) is not list:
        servers = [servers]
    GC.set(servers)
    disconnect = GC.connect(raiseiferror=True)
    maxunits = GC.execute_native('get_total_resources')
    resources = GC.execute_native('get_all_resources')
    params = {type: []}
    clients = GC.get_client_name()
    for i in xrange(len(servers)):
        if resources[i][type] is not None:
            if clients[i] in resources[i][type].keys():
                del resources[i][type][clients[i]]
            busy = sum([r for r in resources[i][type].itervalues()])
        else:
            busy = 0
        n = maxunits[i][type] - busy
        # if units is set, the number of units is at most units
        if units is not None:
            n = min(n, units)
        if n <= 0:
            log_warn("there are no resources left on the server: %d %s out\
                        of %d are used" % (busy, type, maxunits[i][type]))
            n = 0
        params[type].append(n)

    # skip some servers
    for typ in params.keys():
        for i in skip:
            params[type][i] = None

    request_resources(servers, **params)
    if disconnect:
        GC.disconnect()
    return params[type]


def get_total_resources(servers):
    """
    Return the total number of resources available to the clients on the given
    server.
    It is a dict resources[type]=nbr
    """
    if type(servers) is not list:
        servers = [servers]
    GC.set(servers)
    disconnect = GC.connect(raiseiferror=True)
    result = GC.execute_native('get_total_resources')
    if disconnect:
        GC.disconnect()
    return result


def set_total_resources(server, **resources):
    """
    Specify the total number of resources available on the given server
    """
    GC.set(server)
    disconnect = GC.connect(raiseiferror=True)
    result = GC.execute_native('set_total_resources', **resources)
    if disconnect:
        GC.disconnect()
    return result


class Allocation(object):
    """
    Contain information about resource allocation on remote machines.
    Is returned by the :func:`allocate` function.

    Attributes:

    ``self.total_units``
        Total number of units.

    ``self.allocation``
        Allocation dictionary, were keys are machine tuples (IP, port) and
        values are the number
        of resources allocated to the client.
    """
    def __init__(self, servers=[],
                       total_units=None,
                       unit_type='CPU',
                       cpu=None,
                       gpu=None,
                       allocation=None,
                       local=None):
        if type(servers) is not list:
            servers = [servers]
        self.servers = servers
        self.total_units = total_units
        self.unit_type = unit_type
        self.cpu = cpu
        self.gpu = gpu
        # Determines whether the run is local or not
        if local is not None:
            self.local = local
        else:
            self.local = (type(servers) is list and len(servers) == 0)
        if cpu is not None:
            self.unit_type = 'CPU'
            self.total_units = cpu
        elif gpu is not None:
            self.unit_type = 'GPU'
            self.total_units = gpu
        elif total_units is not None:
            self.unit_type = unit_type
            self.total_units = total_units
        elif servers == []:
            # Default: use all resources possible
            if unit_type == 'CPU':
                self.total_units = numpy.inf  # MAXCPU-1
            if unit_type == 'GPU':
#                MAXGPU = initialise_cuda()
                self.total_units = numpy.inf  # MAXGPU
        self.allocation = {}
        if allocation is None:
            self.allocate()
        else:
            self.allocation = allocation
            self.machines = self.get_machines()
            self.total_units = numpy.sum(allocation.values())
            self.local = False

    def allocate(self):
        # Creates the allocation dict
        if self.local:
            # creates the local server
            if self.unit_type == 'CPU':
                self.total_units = min(self.total_units, MAXCPU)
                args = (DEFAULT_PORT, self.total_units, 0, True)
            if self.unit_type == 'GPU':
                MAXGPU = get_gpu_count()
                #MAXGPU = #initialise_cuda() # THIS VERSION NOT SAFE ON LINUX
                self.total_units = min(self.total_units, MAXGPU)
                args = (DEFAULT_PORT, 0, self.total_units, True)
            self.allocation[(LOCAL_IP, DEFAULT_PORT)] = self.total_units
            p = Process(target=open_server, args=args)
            p.start()
        else:

            GC.set(self.servers)
            disconnect = GC.connect(raiseiferror=True)

            for i in xrange(len(self.servers)):
                server = self.servers[i]
                if type(server) is str:
                    server = (server, DEFAULT_PORT)
                if server[0] == 'localhost' or server[0] == '127.0.0.1':
                    server = (LOCAL_IP, server[1])
                self.servers[i] = server

            units = get_my_resources(self.servers)
            units = numpy.array([u[self.unit_type] for u in units],
                                   dtype=numpy.int)

            # allocate optimally servers with no resource allocated yet
            notalreadyallocated = numpy.nonzero(numpy.array(units,
                                                dtype=numpy.int) == 0)[0]
            alreadyallocated = numpy.nonzero(numpy.array(units,
                                                dtype=numpy.int) > 0)[0]
            if len(notalreadyallocated) > 0:
                newunits = request_all_resources(self.servers,
                                                 skip=list(alreadyallocated),
                                                           type=self.unit_type)
                newunits = numpy.array([u for u in newunits if u is not None],
                                        dtype=numpy.int)
                units[notalreadyallocated] = newunits
            # keep servers with units available
#            ind2 = numpy.nonzero(numpy.array(newunits,dtype=int)>0)[0]
            if (self.total_units is not None) and\
                    (numpy.sum(units) > self.total_units):
                # indices of servers with too much units
                ind = numpy.nonzero(numpy.cumsum(units) > self.total_units)[0]
                units[ind] = 0
                units[ind[0]] = self.total_units - numpy.sum(units)
            self.total_units = numpy.sum(units)
            for i in xrange(len(self.servers)):
                self.allocation[self.servers[i]] = units[i]
            if disconnect:
                GC.disconnect()
        self.machines = self.get_machines()
        if not self.local:
            log_info("Using %d %s(s) on %d machine(s)" % (self.total_units,
                                                          self.unit_type,
                                                          len(self.machines)))
        else:
            log_info("Using %d %s(s) on the local machine" % (self.total_units,
                                                              self.unit_type))
        return self.allocation

    def get_machines(self):
        """
        Gets the machines list from an allocation, sorted alphabetically.
        Allocation is a dict {('IP', port): nbr}
        """
        machines = [m for m in self.allocation.keys()
                                if self.allocation[m] > 0]
        machines = [Machine(m) for m in machines]
        machines.sort(cmp=lambda m1, m2: cmp(m1.to_tuple(), m2.to_tuple()))
        return machines

    def get_machine_tuples(self):
        return [m.to_tuple() for m in self.machines]

    machine_tuples = property(get_machine_tuples)

    def iteritems(self):
        return self.allocation.iteritems()

    def keys(self):
        return self.allocation.keys()

    def __getitem__(self, key):
        if type(key) is Machine:
            key = key.to_tuple()
        return self.allocation[key]

    def __repr__(self):
        if self.servers is None or self.servers == []:
            strmachines = "the local machine"
        else:
            strmachines = "%d machine(s)" % len(self.servers)
        return "<Allocation of %d %s(s) on %s>" % (self.total_units,
                                                   self.unit_type,
                                                   strmachines)

    def __len__(self):
        return self.total_units


def allocate(machines=[],
             total_units=None,
             unit_type='CPU',
             allocation=None,
             local=None,
             cpu=None,
             gpu=None):
    """
    Automatically allocate resources on different machines using available
    resources.
    Return an :class:`Allocation` object which can be passed to Playdoh
    functions like :func:`map`,
    :func:`minimize`, etc.

    Arguments:

    ``machines=[]``
        The list of machines to use, as a list of strings (IP addresses)
        or tuples
        (IP address and port number).

    ``cpu=None``
        The total number of CPUs to use.

    ``gpu=None``
        The total number of GPUs to use.

    ``allocation=None``
        This argument is specified when using manual resource allocation.
        In this case,
        ``allocation`` must be a dictionary with machine IP addresses as
        keys and
        resource number as values. The unit type must also be specified.

    ``unit_type='CPU'``
        With manual resource allocation, specify the unit type: ``CPU``
        or ``GPU``.
    """
    al = Allocation(machines, total_units, unit_type, cpu, gpu, allocation,
                    local)
    return al
