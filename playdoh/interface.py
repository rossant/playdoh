from asyncjobhandler import *
from synchandler import *
from optimization import *
from rpc import *
from debugtools import *
from gputools import *
from resources import *
from codehandler import *
from numpy import ndarray, float, floor, log10, inf, ones
from numpy.random import rand
import datetime
import inspect


__all__ = ['remote', 'map_async', 'minimize_async', 'maximize_async',
           'map', 'minimize', 'maximize',
           'start_task', 'start_optimization',
           'print_table']


def remote(f, machines, codedependencies=[]):
    if type(machines) is not list:
        machines = [machines]

    def f0(x):
        return f()

    r = map(f0, [None], machines=machines, codedependencies=codedependencies)
    if len(machines) == 1:
        return r[0]
    else:
        return r


def map_async(fun, *argss, **kwdss):
    """
    Asynchronous version of ``map``. Return a :class:`JobRun` object
    which allows to poll the jobs'
    status asynchronously and retrieve the results later.

    The :func:`map` function is equivalent to ``map_async(...).get_results()``.
    """
    allocation = kwdss.pop('allocation', None)
    shared_data = kwdss.pop('shared_data', {})
    do_redirect = kwdss.pop('do_redirect', None)
    cpu = kwdss.pop('cpu', None)
    gpu = kwdss.pop('gpu', None)
    machines = kwdss.pop('machines', [])
    unit_type = kwdss.pop('unit_type', 'CPU')
    total_units = kwdss.pop('total_units', None)
    codedependencies = kwdss.pop('codedependencies', None)
    disconnect = kwdss.pop('disconnect', True)

    if not inspect.isfunction(fun):
        raise Exception("The first argument of 'map' must be a valid \
            Python function (or a lambda function)")

    # Default allocation of resources
    if allocation is None:
        allocation = allocate(machines=machines,
                              total_units=total_units,
                              unit_type=unit_type,
                              cpu=cpu,
                              gpu=gpu)
    else:
        # if allocation is explicitely specified, then not local
        local = False
    local = allocation.local

    GC.set(allocation.machine_tuples, handler_class=AsyncJobHandler)
    GC.connect()

    # Send dependencies
    if not local:
        fun_pkl = send_dependencies(allocation.machine_tuples, fun,
                                    codedependencies)
    else:
        fun_pkl = pickle_class(fun)

    if not local:
        log_info("Submitting jobs to the servers")
    myjobs = submit_jobs(fun_pkl, argss=argss, kwdss=kwdss,
                         allocation=allocation,
                         unit_type=allocation.unit_type,
                         do_redirect=do_redirect,
                         local=local,
                         shared_data=shared_data)

    if disconnect:
        GC.disconnect()

    return myjobs


def map(*args, **kwds):
    """
    Parallel version of the built-in ``map`` function.
    Executes the function ``fun`` with the arguments ``*argss`` and
    keyword arguments ``**kwdss`` across CPUs on one or several computers.
    Each argument and keyword argument is a list with the arguments
    of every job.
    This function returns the result as a list, one item per job.
    If an exception occurs within the function, :func:`map` returns
    the Exception object as a result. This object has an extra attribute,
    ``traceback``, which contains the traceback of the exception.

    Special keyword arguments:

    ``cpu=MAXCPU``
        Total number of CPUs to distribute the function over.
        If ``machines`` is not specified or is an empty list,
        only CPUs on the local computer will be used.
        The total number of CPUs is obtained with the global
        variable ``MAXCPU``. By default, all CPUs on the machine
        are used.

    ``gpu=0``
        If the function loads CUDA code using the PyCUDA package,
        Playdoh is able to distribute it across GPUs on one or
        several machines. In this case, ``gpu`` is the total
        number of GPUs to use and works the same way as ``cpu``.

        .. seealso:: User guide :ref:`gpu`.

    ``machines=[]``
        The list of computers to distribute the function over.
        Items can be either IP addresses as strings, or tuples
        ``('IP', port)`` where ``port`` is an integer giving
        the port over which the Playdoh server is listening
        on the remote computer. The default port is obtained
        with the global variable ``DEFAULT_PORT``.

    ``allocation=None``
        Resource allocation is normally done automatically assuming
        that all CPUs are equivalent. However, it can also be done
        manually by specifying the number of CPUs to use on every
        computer. It is done with the :func:`allocate` function.

        .. seealso:: User guide for :ref:`allocation`.

    ``shared_data={}``
        Large data objects (NumPy arrays) can be shared across CPUs
        running on the same
        computer, but they must be read-only. The ``shared_data``
        argument is a
        dictionary: keys are variable names and values are large
        NumPy arrays that should be stored in shared memory
        on every computer if possible.

        .. seealso:: User guide for :ref:`shared_data`.

    ``codedependencies=[]``
        If the function to distribute uses external Python modules,
        these modules must be transported to every machine along
        with the function code. The ``codedependencies`` argument
        contains the list of these modules' pathnames relatively
        to the directory where the function's module is defined.

        .. seealso:: User guide for :ref:`code_transport`.
    """
    kwds['disconnect'] = False
    myjobs = map_async(*args, **kwds)
    results = myjobs.get_results()
    GC.disconnect()
    return results


def minimize_async(fun,
             popsize=100,
             maxiter=5,
             algorithm=CMAES,
             allocation=None,
             unit_type='CPU',
             shared_data={},
             maximize=False,
             groups=1,
             cpu=None,
             gpu=None,
             total_units=None,
             codedependencies=None,
             optparams={},
             machines=[],
             returninfo=False,
             bounds=None,
             initrange=None,
             scaling=None,
             do_redirect=None,
             args=(),
             kwds={},
             **params):
    """
    Asynchronous version of :func:`minimize`. Returns an
    :class:`OptimizationRun` object.
    """

    # Make sure the parameters are float numbers
    for k, v in params.iteritems():
        if type(v) is tuple or type(v) is list:
            for i in xrange(len(v)):
                if type(v[i]) is int:
                    v[i] = float(v[i])

    argtype = 'matrix'  # keywords or matrix argument for the fitness function
    if bounds is not None and initrange is not None:
        pass
#        ndimensions = bounds.shape[0]
    elif bounds is not None:
#        ndimensions = bounds.shape[0]
        initrange = bounds
    elif initrange is not None:
        bounds = ones(initrange.shape)
        bounds[:, 0] = -inf
        bounds[:, 1] = inf
#        ndimensions = bounds.shape[0]
    else:
        argtype = 'keywords'
#        ndimensions = len(params)
        params4 = {}
        for k in params.keys():
            ksplit = k.split('_')
            k0 = '_'.join(ksplit[:-1])
            k1 = ksplit[-1]
            if len(ksplit) > 1 and k1 in ['bounds', 'initrange'] and \
                    k0 not in params4.keys():
                params4[k0] = [-inf, None, None, inf]
            if len(ksplit) == 1 or k1 not in ['bounds', 'initrange']:
                params4[k] = params[k]
        for name in params4.keys():
            name_bound = '%s_bounds' % name
            name_init = '%s_initrange' % name
            if name_bound in params.keys():
                params4[name][0] = params[name_bound][0]
                params4[name][1] = params[name_bound][0]
                params4[name][2] = params[name_bound][1]
                params4[name][3] = params[name_bound][1]
            if name_init in params.keys():
                params4[name][1] = params[name_init][0]
                params4[name][2] = params[name_init][1]
        params = params4
    if bounds is not None:
        bounds = bounds.astype(float)
    if initrange is not None:
        initrange = initrange.astype(float)

    # Default allocation of resources
    if allocation is None:
        allocation = allocate(machines=machines,
                              total_units=total_units,
                              unit_type=unit_type,
                              cpu=cpu,
                              gpu=gpu)
    else:
        # if allocation is explicitely specified, then not local
        local = False

    local = allocation.local

    GC.set(allocation.machine_tuples)
    disconnect = GC.connect()

    # Send dependencies
    if not local:
        fun_pkl = send_dependencies(allocation.machine_tuples, fun,
                                    codedependencies)
    else:
        fun_pkl = pickle_class(fun)

    task = start_optimization(fun_pkl,
                              maxiter,
                              popsize=popsize,
                              local=local,
                              groups=groups,
                              maximize=maximize,
                              algorithm=algorithm,
                              allocation=allocation,
                              optparams=optparams,
                              shared_data=shared_data,
                              argtype=argtype,
                              unit_type=unit_type,
                              initrange=initrange,
                              bounds=bounds,
                              scaling=scaling,
                              return_info=returninfo,
                              do_redirect=do_redirect,
                              initargs=args,
                              initkwds=kwds,
                              **params)

    if disconnect:
        GC.disconnect()

    return task


def minimize(*args, **kwds):
    """
    Minimize a fitness function in parallel across CPUs on one or several
    computers.

    Arguments:

    ``fitness``
        The first argument is the fitness function. There are four
        possibilities:
        it can be a Python function or a Python class (deriving from
        :class:`Fitness`).
        It can also accept either keyword named arguments (like
        ``fitness(**kwds)``)
        or a ``DxN`` matrix (like ``fitness(X)``) where there are ``D``
        dimensions in the
        parameter space and ``N`` particles.

        Using a class rather than a function allows to implement an
        initialization step
        at the beginning of the optimization. See the reference for
        :class:`Fitness`.

        If the fitness is a simple keyword-like Python function, it must have
        the right keyword arguments.
        For example, if there are two parameters ``x`` and ``y`` to optimize,
        the fitness function
        must be like ``def fitness(x,y):``. If it's a matrix-like function,
        it must accept a single argument
        which is a matrix: ``def fitness(X):``.

        Fitness functions can also accept static arguments, given in the
        :func:`minimize` functions
        and alike with the ``args`` and ``kwds`` parameters (see below).

        In addition, the fitness function can accept several special keyword
        arguments:

        ``dimension``
            The dimension of the state space, or the number of optimizing
            parameters

        ``popsize``
            The total population size for each group across all nodes.

        ``subpopsize``
            The population size for each group on this node.

        ``groups``
            The number of groups.

        ``nodesize``
            The population size for all groups on this node.

        ``nodecount``
            The number of nodes used for this optimization.

        ``shared_data``
            The dictionary with shared data.

        ``unit_type``
            The unit type, ``CPU`` or ``GPU``.

        For example, use the following syntax to retrieve within the function
        the shared data dictionary and the size of the population on the
        current node:
        ``def fitness(X, shared_data, nodesize):``.


    ``popsize=100``
        Size of the population. If there are several groups, it is the size
        of the population for every group.

    ``maxiter=5``
        Maximum number of iterations.

    ``algorithm=PSO``
        Optimization algorithm. For now, it can be :class:`PSO`, :class:`GA`
        or :class:`CMAES`.

    ``allocation=None``
        :class:`Allocation` object.

        .. seealso:: User guide for :ref:`allocation`.

    ``shared_data={}``
        Dictionary containing shared data between CPUs on a same computer.

        .. seealso:: User guide for :ref:`shared_data`.

    ``groups=1``
        Number of groups. Allows to optimize independently several populations
        by using a single vectorized call to the fitness function at every
        iteration.

        .. seealso:: User guide for :ref:`groups`.

    ``cpu=MAXCPU``
        Total number of CPUs to use.

    ``gpu=0``
        If the fitness function loads CUDA code using the PyCUDA package,
        several
        GPUs can be used. In this case, ``gpu`` is the total number of GPUs.

    ``codedependencies=[]``
        List of dependent modules.

        .. seealso:: User guide for :ref:`code_transport`.

    ``optparams={}``
        Optimization algorithm parameters. It is a dictionary: keys are
        parameter names,
        values are parameter values or lists of parameters (one value per
        group).
        This argument is specific to the optimization
        algorithm used. See :class:`PSO`, :class:`GA`, :class:`CMAES`.

    ``machines=[]``
        List of machines to distribute the optimization over.

    ``scaling=None``
        Specify the scaling used for the parameters during the optimization.
        It can be ``None`` or ``'mapminmax'``. It is ``None``
        by default (no scaling), and ``mapminmax`` by default for the
        CMAES algorithm.

    ``returninfo=False``
        Boolean specifying whether information about the optimization
        should be returned with the results.

    ``args=()``
        With fitness functions, arguments of the fitness function in addition
        of the
        optimizing parameters.
        With fitness classes, arguments of the ``initialize`` method of the
        :class:`Fitness` class.
        When using a fitness keyword-like function, the arguments must be
        before the optimizing
        parameters, i.e. like ``def fitness(arg1, arg2, x1, x2):``.

    ``kwds={}``
        With fitness functions, keyword arguments of the fitness function in
        addition of the
        optimizing parameters.
        With fitness classes, keyword arguments of the ``initialize`` method
        of the :class:`Fitness` class.

    ``bounds=None``
        Used with array-like fitness functions only.
        This argument is a Dx2 NumPy array with the boundaries of the parameter
        space.
        The first column contains the minimum values acceptable for
        the parameters
        (or -inf), the second column contains the maximum values
        (or +inf).

    ``initrange=None``
        Used with array-like fitness functions only.
        This argument is a Dx2 NumPy array with the initial range in which
        the parameters should be sampled at the algorithm initialization.

    ``**params``
        Used with keyword-like fitness functions only.
        For every parameter <paramname>, the initial sampling interval
        can be specified with the keyword ``<paramname>_initrange`` which is
        a tuple with two values ``(min,max)``.
        The boundaries can be specified with the keyword ``<paramname>_bound``
        which is a tuple with two values ``(min,max)``.
        For example, if there is a single parameter in the fitness function,
        ``def fitness(x):``,
        use the following syntax:
        ``minimize(..., x_initrange=(-1,1), x_bounds=(-10,10))``.

    Return an :class:`OptimizationResult` object with the following attributes:

    ``best_pos``
        Minimizing position found by the algorithm. For array-like fitness
        functions,
        it is a single vector if there is one group, or a list of vectors.
        For keyword-like fitness functions, it is a dictionary
        where keys are parameter names and values are numeric values. If there
        are several groups,
        it is a list of dictionaries.

    ``best_fit``
        The value of the fitness function for the best positions. It is a
        single value if
        there is one group, or it is a list if there are several groups.

    ``info``
        A dictionary containing various information about the optimization.

        .. seealso:: User guide for :ref:`optinfo`.
    """
#    returninfo = kwds.pop('returninfo', None)
    task = minimize_async(*args, **kwds)
    return task.get_result()


def maximize_async(*args, **kwds):
    """
    Asynchronous version of :func:`maximize`. Returns an
    :class:`OptimizationRun` object.
    """
    kwds['maximize'] = True
    return minimize_async(*args, **kwds)


def maximize(*args, **kwds):
    """
    Maximize a fitness function in parallel across CPUs on one or several
    computers. Completely analogous to :func:`minimize`.
    """
#    returninfo = kwds.pop('returninfo', None)
    task = maximize_async(*args, **kwds)
    return task.get_result()


def print_quantity(x, precision=3):
    if x == 0.0:
        u = 0
    elif abs(x) == inf:
        return 'NaN'
    else:
        u = int(3 * floor((log10(abs(x)) + 1) / 3))
    y = float(x / (10 ** u))
    s = ('%2.' + str(precision) + 'f') % y
    if (y > 0) & (y < 10.0):
        s = '  ' + s
    elif (y > 0) & (y < 100.0):
        s = ' ' + s
    if (y < 0) & (y > -10.0):
        s = ' ' + s
    elif (y < 0) & (y > -100.0):
        s = '' + s
    if u is not 0:
        su = 'e'
        if u > 0:
            su += '+'
        su += str(u)
    else:
        su = ''
    return s + su


def print_row(name, values, colwidth):
    spaces = ' ' * (colwidth - len(name))
    print name + spaces,
    if type(values) is not list and type(values) is not ndarray:
        values = [values]
    for value in values:
        s = print_quantity(value)
        spaces = ' ' * (colwidth - len(s))
        print s + spaces,
    print


def print_table(results, precision=4, colwidth=16):
    """
    Displays the results of an optimization in a table.

    Arguments:

    ``results``
        The results returned by the ``minimize`` of ``maximize`` function.

    ``precision = 4``
        The number of decimals to print for the parameter values.

    ``colwidth = 16``
        The width of the columns in the table.
    """
#    if type(results['best_fit']) is not list:
#        group_count = 1
#    else:
#        group_count = len(results['best_fit'])
    group_count = results.groups

    print "RESULTS"
    print '-' * colwidth * (group_count + 1)

    if group_count > 1:
        print ' ' * colwidth,
        for i in xrange(group_count):
            s = 'Group %d' % i
            spaces = ' ' * (colwidth - len(s))
            print s + spaces,
        print

    best_pos = results.best_pos

    if best_pos is None:
        log_warn("The optimization results are not valid")
        return

    if group_count == 1:
        best_pos = [best_pos]
    if results.parameters.argtype != 'keywords':
#    keys = results.keys()
#    keys.sort()
#    if 'best_pos' in keys:
#        best_pos = results['best_pos']
#        if best_pos.ndim == 1:
#            best_pos = best_pos.reshape((-1,1))
#        for i in xrange(best_pos.shape[0]):
        for i in xrange(len(best_pos[0])):
#            val = best_pos[i,:]
            val = [best_pos[k][i] for k in xrange(len(best_pos))]
            print_row("variable #%d" % (i + 1), val, colwidth)
    else:
        keys = best_pos[0].keys()
        keys.sort()
        for key in keys:
            val = [results[i].best_pos[key] for i in xrange(group_count)]
#            if key[0:8] != 'best_fit':
            print_row(key, val, colwidth)

    val = [results[i].best_fit for i in xrange(group_count)]
    print_row('best fit', val, colwidth)
    print


def start_task(task_class,
               task_id=None,
               topology=[],
               nodes=None,  # nodes can be specified manually
               allocation=None,  # in general, allocation is specified
               machines=[],
               total_units=None,
               unit_type='CPU',
               cpu=None,
               gpu=None,
               local=None,
               codedependencies=None,
               pickle_task=True,
               shared_data={},
               do_redirect=None,
               args=(),
               kwds={}):
    """
    Launches a parallel task across CPUs on one or several computers.

    Arguments:

    ``task_class``
        The class implementing the task, must derive from the base class
        ``ParallelTask``.

    ``task_id=None``
        The name of this particular task run. It should be unique, by default
        it is randomly
        generated based on the date and time of the launch. It is used to
        retrieve the results.

    ``topology=[]``
        The network topology. It defines the list of tubes used by the task.
        It is a list of tuples ``(tubename, source, target)`` where
        ``tubename`` is the name of the tube, ``source`` is an integer
        giving the
        node index of the source, ``target`` is the node index of the target.
        Node indices start at 0.

    ``cpu=None``
        The total number of CPUs to use.

    ``gpu=None``
        When using GPUs, the total number of GPUs to use.

    ``machines=[]``
        The list of machine IP addresses to launch the task over.

    ``allocation=None``
        The allocation object returned by the ``allocate`` function.

    ``codedependencies``
        The list of module dependencies.

    ``shared_data={}``
        Shared data.

    ``args=()``
        The arguments to the ``initialize`` method of the task. Every argument
        item
        is a list with one element per node.

    ``kwds={}``
        The keyword arguments to the ``initialize`` method of the task. Every
        value is a list with one element per node.
    """
    # Default task id
    if task_id is None:
        now = datetime.datetime.now()
        task_id = "task-%d%d%d-%d%d%d-%.4d" % (now.year, now.month, now.day,
                                              now.hour, now.minute, now.second,
                                              int(rand() * 1000))

    # Default allocation of resources
    if allocation is None:
        allocation = allocate(machines=machines,
                              total_units=total_units,
                              unit_type=unit_type,
                              cpu=cpu,
                              gpu=gpu)

    machines = allocation.machines
    unit_type = allocation.unit_type
    total_units = allocation.total_units

    # list of nodes on every machine
    nodes_on_machines = dict([(m, []) for m in machines])
    # Creates a Nodes list from an allocation
    if nodes is None:
        nodes = []
        index = 0
        for m in machines:
            units = allocation[m]
            for unitidx in xrange(units):
                nodes.append(Node(index, m, unit_type, unitidx))
                nodes_on_machines[m].append(index)
                index += 1

    mytask = TaskRun(task_id, unit_type, machines, nodes, args, kwds)

    if local is None:
        local = allocation.local
    mytask.set_local(local)

    # Gets the args and kwds of each machine
    argss = args
    kwdss = kwds
    k = len(argss)  # number of non-named arguments
    keys = kwdss.keys()  # keyword arguments

    # Duplicates non-list args
    argss = list(argss)
    for l in xrange(k):
        if type(argss[l]) is not list:
            argss[l] = [argss[l]] * len(nodes)

    # Duplicates non-list kwds
    for key in keys:
        if type(kwdss[key]) is not list:
            kwdss[key] = [kwdss[key]] * len(nodes)

    # Now, argss is a list of list and kwdss a dict of lists
    # argss[l][i] is the list of arg #l for node #i
    # we must convert it so that
    argss2 = [[] for _ in xrange(k)]
    kwdss2 = dict([(key, []) for key in keys])
    for i in xrange(len(machines)):
        m = machines[i]
        local_nodes = nodes_on_machines[m]
        for l in xrange(k):
            argss2[l].append([argss[l][ln] for ln in local_nodes])
        for key in keys:
            kwdss2[key].append([kwdss[key][ln] for ln in local_nodes])

    GC.set(allocation.machine_tuples)
    disconnect = GC.connect()

    if pickle_task:
        if not local:
            task_class_pkl = send_dependencies(allocation.machine_tuples,
                                               task_class, codedependencies)
        else:
            task_class_pkl = pickle_class(task_class)
    else:
        task_class_pkl = task_class

    n = len(allocation.machines)
    GC.set(allocation.machine_tuples, handler_class=SyncHandler,
           handler_id=task_id)

    # BUG: when there is shared_data, the processes are relaunched => bug
    # on windows when a socket connection is open: the connection is "forked"
    # to the subprocesses and the server is not reachable anymore
    if len(shared_data) > 0:
        close_connection_temp = True
    else:
        close_connection_temp = False

    if not local:
        log_info("Submitting the task to %d server(s)" % n)
    GC.submit([task_class_pkl] * n,
              [task_id] * n,
              [topology] * n,
              [nodes] * n,
              # local nodes
              [[nodes[i] for i in nodes_on_machines[m]] for m in machines],
              [unit_type] * n,
              do_redirect=[do_redirect] * n,
              shared_data=[shared_data] * n,
              _close_connection_temp=close_connection_temp)

    if close_connection_temp:
        GC.connect()

    if not local:
        log_info("Initializing task")
    GC.initialize(*argss2, **kwdss2)

    if not local:
        log_info("Starting task")
    GC.start()

    if disconnect:
        GC.disconnect()
    return mytask


def start_optimization(fitness_class,
                       maxiter=5,
                       popsize=100,
                       groups=1,
                       algorithm=PSO,
                       maximize=False,
                       task_id=None,
                       unit_type='CPU',
                       nodes=None,
                       allocation=None,
                       local=None,
                       shared_data={},
                       optparams={},
                       scaling=None,
                       initrange=None,
                       bounds=None,
                       argtype='keywords',
                       return_info=False,
                       do_redirect=None,
                       initargs=(),
                       initkwds={},
                       **parameters):
    """
    Starts an optimization. Use ``minimize_async`` and ``maximize_async``
    instead.
    """
    nodecount = allocation.total_units
    topology = algorithm.get_topology(nodecount)

    parameters = OptimizationParameters(scaling=scaling,
                                        argtype=argtype,
                                        initrange=initrange,
                                        bounds=bounds,
                                        **parameters)

    args = (algorithm,
            fitness_class,
            maximize,
            maxiter,
            scaling,
            popsize,
            nodecount,
            groups,
            return_info,
            parameters,
            optparams,
            initargs,
            initkwds)

    task = start_task(Optimization,
                      task_id,
                      topology,
                      unit_type=unit_type,
                      nodes=nodes,
                      local=local,
                      allocation=allocation,
                      shared_data=shared_data,
                      codedependencies=[],
                      pickle_task=False,
                      do_redirect=do_redirect,
                      args=args)

    return OptimizationRun(task)
