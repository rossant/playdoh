from numpy import array, sort,\
zeros, inf, squeeze, zeros_like
import inspect
from algorithm import *
from pso import *
from ..codehandler import *
from ..debugtools import *
from ..rpc import *
from ..synchandler import *
from ..gputools import *

__all__ = ['Optimization', 'OptimizationParameters',
           'Fitness', 'OptimizationRun', 'OptimizationResult']


class OptimizationParameters(object):
    """Internal class used to manipulate optimization parameters.
    It basically handles conversion between parameter dictionaries and arrays.

    Initialized with arguments:

    ``**params``
        Parameters list ``param_name=[bound_min, min_ max, bound_max]``

    **Methods**

    .. method:: get_initial_param_values(N)

        Returns initial parameter values sampled uniformly within the parameter
        interval given in the constructor of the class. ``N`` is the number
        of particles. The result is a dictionary ``{param_name=values}`` where
        values is a vector of values.

    .. method:: set_constraints()

        Returns the constraints for each parameter. The result is
        (min_values, max_values) where each variable is a vector containing
        the minimum and maximum values for each parameter.

    .. method:: get_param_values(X)

        Converts an array containing parameter values into a dictionary.

    .. method:: get_param_matrix(param_values)

        Converts a dictionary containing parameter values into an array.
    """
    def __init__(self,
                 scaling=None,
                 initrange=None,
                 bounds=None,
                 argtype='keywords',
                 **params):
        self.scaling = scaling
        self.initrange = initrange
        self.bounds = bounds
        self.argtype = argtype
        if argtype == 'keywords':
            self.params = params
            self.param_names = sort(params.keys())
            self.param_count = len(params)
        else:
            self.param_count = initrange.shape[0]

    def set_constraints(self):
        """
        Returns constraints of a given model
        returns min_values, max_values
        min_values is an array of length p where p is the number of parameters
        min_values[i] is the minimum value for parameter i
        """

        if self.argtype == 'matrix':
            # TODO BERTRAND: scaling
            boundaries = zeros((self.initrange.shape[0], 2))
            self.scaling_factor_a = []
            self.scaling_factor_b = []
            for idim in xrange(self.initrange.shape[0]):
                if self.bounds is None:
                    boundaries[idim, :] = [-inf, inf]
                    #used for the scaling
                    self.scaling_factor_a.append(self.initrange[idim, 0])
                    self.scaling_factor_b.append(self.initrange[idim, 1])
                else:
                    boundaries[idim, :] = \
                    [self.bounds[idim, 0], self.bounds[idim, 1]]
                    #used for the scaling
                    self.scaling_factor_a.append(self.bounds[idim, 0])
                    self.scaling_factor_b.append(self.bounds[idim, 1])

        else:
            param_names = self.param_names
            boundaries = zeros((len(param_names), 2))
            self.scaling_factor_a = []
            self.scaling_factor_b = []
            icount = 0
            for key in param_names:
                value = self.params[key]
                # No boundary conditions if only two values are given
                if len(value) == 2:
                    # One default interval,
                    #no boundary counditions on parameters
                    boundaries[icount, :] = [-inf, inf]
                    #used for the scaling
                    self.scaling_factor_a.append(value[0])
                    self.scaling_factor_b.append(value[-1])
                elif len(value) == 4:
                    # One default interval,
                    #value = [min, init_min, init_max, max]
                    boundaries[icount, :] = [value[0], value[3]]
                    #used for the scaling
                    self.scaling_factor_a.append(value[1])
                    self.scaling_factor_b.append(value[2])
                icount += 1
        self.scaling_factor_a = array(self.scaling_factor_a)
        self.scaling_factor_b = array(self.scaling_factor_b)
        self.boundaries = boundaries
        return boundaries

    def scaling_func(self, x):
        x = squeeze(x)
        x_new = zeros_like(x)
        if x.ndim == 1:
            for idim in xrange(len(x)):
                x_new[idim] = 2 * x[idim] / (self.scaling_factor_b[idim] - \
                self.scaling_factor_a[idim]) + (self.scaling_factor_a[idim] + \
                self.scaling_factor_b[idim]) / (self.scaling_factor_a[idim] - \
                self.scaling_factor_b[idim])
        else:
            for idim in xrange(x.shape[0]):
                x_new[idim, :] = 2 * x[idim, :] / \
                (self.scaling_factor_b[idim] - \
                self.scaling_factor_a[idim]) + (self.scaling_factor_a[idim] + \
                self.scaling_factor_b[idim]) / (self.scaling_factor_a[idim] - \
                self.scaling_factor_b[idim])
        return x_new

    def unscaling_func(self, x):
        x = squeeze(x)
        x_new = zeros_like(x)
        if x.ndim == 1:
            for idim in xrange(len(x)):
                x_new[idim] = (self.scaling_factor_b[idim] - \
                self.scaling_factor_a[idim]) / \
                2 * (x_new[idim] - (self.scaling_factor_a[idim] \
                + self.scaling_factor_b[idim]) / \
                (self.scaling_factor_a[idim] - \
                 self.scaling_factor_b[idim]))
        else:
            for idim in xrange(x.shape[0]):
                x_new[idim] = (self.scaling_factor_b[idim] - \
                self.scaling_factor_a[idim]) / 2 * (x_new[idim] - \
                (self.scaling_factor_a[idim] + self.scaling_factor_b[idim]) / \
                (self.scaling_factor_a[idim] - self.scaling_factor_b[idim]))
        return x_new

    def get_param_values(self, X):
        """
        Converts a matrix containing param values into a dictionary
        (from the algorithm to the fitness (unscale))
        """
        param_values = {}
        if X.ndim <= 1:
            X = X.reshape((-1, 1))
        for i in range(len(self.param_names)):
            if self.scaling == None:
                param_values[self.param_names[i]] = X[i, :]
            else:
                param_values[self.param_names[i]] =\
                (self.scaling_factor_b[i] -\
                self.scaling_factor_a[i]) / 2 * (X[i, :]\
                 - (self.scaling_factor_a[i] +\
                self.scaling_factor_b[i]) / (self.scaling_factor_a[i] - \
                self.scaling_factor_b[i]))
        return param_values

    def get_param_matrix(self, param_values):
        """
        Converts a dictionary containing param values
        into a matrix (from the fitness to the algorithm (scale))
        """
        p = self.param_count
        # Number of parameter values (number of particles)
        n = len(param_values[self.param_names[0]])
        X = zeros((p, n))
        for i in range(p):
            if self.scaling == None:
                X[i, :] = param_values[self.param_names[i]]
            else:
                X[i, :] = 2 * param_values[self.param_names[i]] / \
                (self.scaling_factor_b[i] - self.scaling_factor_a[i]) + \
                (self.scaling_factor_a[i] + self.scaling_factor_b[i]) / \
                (self.scaling_factor_a[i] - self.scaling_factor_b[i])
        return X


class Optimization(ParallelTask):
    def initialize(self, algorithm,
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
                         init_args,
                         init_kwds,
                         ):
        """
        Initializes the optimization.
        algorithm: optimization algorithm class
        fitness_class: the class implementing the fitness function
        maxiter: number of iterations
        groups: number of groups
        parameters: OptimizationParameters object
        optparams: dict containing the parameters specific to the algorithm
        """
        log_debug("Optimization initialization")
        self.algorithm = algorithm
        self.fitness_class = fitness_class
        self.maximize = maximize
        self.scaling = scaling
        self.return_info = return_info
        self.dimension = parameters.param_count
        self.maxiter = maxiter
        self.parameters = parameters
        self.iteration = 0
        self.nodeidx = self.index
        self.init_args = init_args
        self.init_kwds = init_kwds

        if type(self.fitness_class) == PicklableClass:
            self.isfunction = self.fitness_class.isfunction
            if self.isfunction:
                self.arglist = self.fitness_class.arglist
        else:
            self.isfunction = inspect.isfunction(self.fitness_class)
            if self.isfunction:
                self.arglist = inspect.getargspec(self.fitness_class)[0]

        # dict {group: particles} for each group on this node
#        self.this_groups = groups.groups_by_node[self.node.index]
        # total number of particles on this node
#        self.particles = self.groups.particles_in_nodes[self.node.index]

        # number of groups
        self.groups = groups
        # makes popsize a multiple of node_count
        # size of the subpopulation of each group on each node
        self.subpopsize = popsize / nodecount
        # size of the population of each group (split across nodes)
        self.popsize = self.subpopsize * nodecount
        # number of nodes
        self.nodecount = nodecount
        # total size on each node
        self.nodesize = groups * self.subpopsize

        # optparams is a dict {key: val} or dict{key: [val1, val2...]}
        # if one wants to use different values with different groups
        # the following converts optparams to the second case and
        # fills it with default values if the key is not specified in optparams
        default_optparams = self.algorithm.default_optparams()
        optparamskeys = default_optparams.keys()
#        self.optparams = dict([(k, []) for k in optparamskeys])
        for key in optparamskeys:
            # fills with default value if needed
            if key not in optparams.keys():
                optparams[key] = default_optparams[key]
            # converts to a list if not already a list (one element per group)
            if type(optparams[key]) is not list:
                optparams[key] = [optparams[key]] * groups
        self.optparams = optparams

        self.initialize_algorithm()
        self.initialize_fitness_class()

    def initialize_algorithm(self):
        log_debug("Algorithm initialization")
        self.engine = self.algorithm(self.index,
                                    self.nodes,
                                    self.tubes,
                                    self.popsize,
                                    self.subpopsize,
                                    self.nodesize,
                                    self.groups,
                                    self.return_info,
                                    self.maxiter,
                                    self.scaling,
                                    self.parameters,
                                    self.optparams)

        self.engine.boundaries = self.parameters.set_constraints()
        self.engine.initialize()
        self.engine.initialize_particles()
        self.X = self.engine.X

    def initialize_fitness_class(self):
        """
        Initializes the fitness object, or function
        if fitness_class is not a class but a function
        """
        if self.isfunction:
            self.fitness_object = self.fitness_class
        else:
            self.fitness_object = self.fitness_class(
                                                 self.parameters.param_count,
                                                 self.popsize,
                                                 self.subpopsize,
                                                 self.groups,
                                                 self.nodesize,
                                                 self.nodecount,
                                                 self.shared_data,
                                                 self.unit_type,
                                                 self.init_args,
                                                 self.init_kwds)

    def get_fitness(self):
        if self.parameters.argtype == 'matrix':
            param_values = {}
        else:
            param_values = self.parameters.get_param_values(self.X)
        kwds = param_values
        if self.isfunction:
            # Pass special keyword arguments to the function
            for k in self.arglist:
                if hasattr(self, k):
                    kwds[k] = getattr(self, k)

            # add init_args/kwds to the function if using a fitness function
            args = self.init_args
            for k, v in self.init_kwds.iteritems():
                kwds[k] = v
        else:
            args = ()

        if self.parameters.argtype == 'keywords':
            self.fitness = self.fitness_object(*args, **kwds)
        else:
            self.fitness = self.fitness_object(self.X, *args, **kwds)

        return self.fitness

    def start(self):
        log_debug("Starting optimization")

        # main loop
        for self.iteration in xrange(self.maxiter):
            # Print Iteration i/n only once on each machine
            # TODO: not use unitidx to be more general
            if self.unitidx == 0:
                log_info("Iteration %d/%d" \
                         % (self.iteration + 1, self.maxiter))
            else:
                log_debug("Iteration %d/%d" %\
                           (self.iteration + 1, self.maxiter))
            # pre-fitness
#            for group, engine in self.engines.iteritems():
#                engine.pre_fitness()
            self.engine.pre_fitness()

            # evaluates the fitness
#            log_debug("Get fitness")
            fitness = self.get_fitness()
            # MAXIMIZE
            if self.maximize:
                fitness *= -1

#            fitness_split = self.groups.split_matrix(fitness, self.node.index)

            # post-fitness
#            for group, engine in self.engines.iteritems():
#                fitness=engine.post_fitness(fitness_split[group])
            self.engine.post_fitness(fitness)

            # iterate the algorithm on each group
#            new_X_split = {}
#            for group, engine in self.engines.iteritems():
#                engine.iterate(self.iteration, fitness_split[group])
#                new_X_split[group] = engine.X
#            self.X = self.groups.concatenate_matrix\
            #(new_X_split, self.node.index)
            self.engine.iterate(self.iteration, fitness)
            self.X = self.engine.X

#        if self.unit_type == 'GPU':
#            close_cuda()

#        self.X_best_split = {}
#        self.fitness_best_split = {}
#        for group, engine in self.engines.iteritems():
#            self.X_best_split[group], self.fitness_best_split[group]\
#             = engine.get_result()
#            # MAXIMIZE
#            if self.maximize:
#                self.fitness_best_split[group] =\
#           -self.fitness_best_split[group]

        # WARNING: only node 0 returns the result
        self.best_X, self.best_fit = self.engine.get_result()

        if self.maximize:
            self.best_fit = [-bf for bf in self.best_fit]

    def get_info(self):
#        info = {}
#        for group, engine in self.engines.iteritems():
#            info[group] = engine.get_info()
        return self.engine.get_info()

    def get_result(self):
        """
        Returns a tuple (best_pos, best_fit, info).
        Each one is a dict {group: value}
        """
        # ONLY node 0 returns the result
        if self.index != 0:
            return [], [], []
        info = self.get_info()
        if self.parameters.argtype == 'keywords':
            best_pos = []            for group in xrange(self.groups):
                group_best_X = self.best_X[group]
                if self.groups == 1:
                    group_best_X = [group_best_X]
                group_best_X = array(group_best_X)
                group_best_pos = self.parameters.\
                get_param_values(self.best_X[group])
                best_pos.append(group_best_pos)
#                for k in self.parameters.param_names:
#                    best_pos[k].append(group_best_pos[k][0])
        else:
            best_pos = self.best_X
        return best_pos, self.best_fit, info


class OptimizationRun(object):
    """
    Contains information about a parallel optimization that has been launched
    with the ``minimize_async`` or ``maximize_async`` function.

    Methods:

    ``get_info()``
        Return information about the current optimization asynchronously.

    ``get_result()``
        Return the result as an :class:`OptimizationResult` instance.
        Block until the optimization has finished.
    """
    def __init__(self, taskrun):
        self.taskrun = taskrun

    def get_info(self, node=0):
        # returns only info about node 0
        info = array(self.taskrun.get_info())
        return list(info[node])

    def get_result(self):
        """
        Returns a tuple (best_pos, best_fit, info) if returninfo is True,
            or (best_pos, best_fit) otherwise.
        If there is a single group:
            best_pos is a dict {param_name: value}
            best_fit is a number
            info is any object
        Otherwise:
            best_pos is a dict {param_name: values (list of values for groups)}
            best_fit is a list [group: value]
            info is a dict{group: info}
        """

#        result = self.taskrun.get_result()
#
#        groups = self.taskrun.args[7]
#        parameters = self.taskrun.args[8]
#
#        # all nodes return the same result
#        best_pos, best_fit, info = result[0]
##        if groups.group_count == 1:
#        if groups == 1:
##            best_pos = best_pos[0]
#            best_fit = best_fit[0]
#            info = info[0]
##            best_pos = parameters.get_param_values(best_X)
#            if parameters.argtype == 'keywords':
#                for k in parameters.param_names:
#                    best_pos[k] = best_pos[k][0]
#            else:
#                best_pos = best_pos.flatten()
#
#        if groups > 1:
#            best_fit = list(best_fit)
#
#        if parameters.argtype == 'keywords':
#            result = best_pos
#            result['best_fit'] = best_fit
#        else:
#            result = {}
#            result['best_pos'] = best_pos
#            result['best_fit'] = best_fit
#
#        if returninfo:
#            return result, info
#        else:
#            return result

        result = self.taskrun.get_result()

        return OptimizationResult(result, self.taskrun.args)


class GroupOptimizationResult(object):
    def __init__(self, group, best_pos, best_fit, info={}):
        self.group = group
        self.best_pos = best_pos
        self.best_fit = best_fit
        self.info = info

    def __getitem__(self, key):
        if type(key) is str:
            return self.best_pos[key]

    def __repr__(self):
        return ("Best position for group %d: " \
                % self.group) + str(self.best_pos)


class OptimizationResult(object):
    """
    Type of objects returned by optimization functions.

    Attributes:

    ``best_pos``
        Minimizing position found by the algorithm.
        For array-like fitness functions,
        it is a single vector if there is one group,
        or a list of vectors.
        For keyword-like fitness functions, it is a dictionary
        where keys are parameter names and values are numeric values.
        If there are several groups, it is a list of dictionaries.

    ``best_fit``
        The value of the fitness function for the best positions.
        It is a single value i there is one group, or it is a list
        if there are several groups.

    ``info``
        A dictionary containing various information
        about the optimization.

    Also, the following syntax is possible with an
    ``OptimizationResult`` instance ``or``. The ``key`` is either
    an optimizing parameter name for keyword-like fitness functions,
    or a dimension index for array-like fitness functions.

    ``or[key]``
        it is the best ``key`` parameter found (single value),
        or the list of the best parameters ``key`` found for all groups.

    ``or[i]``
        where ``i`` is a group index. This object has attributes
        ``best_pos``, ``best_fit``, ``info`` but only for group ``i``.

    ``or[i][key]``
        where ``i`` is a group index, is the same as ``or[i].best_pos[key]``.
    """
    def __init__(self, result, args):
        self.result = result
        self.args = args
        self.groups = args[7]
        self.returninfo = args[8]
        self.parameters = args[9]
        self.best_pos = self.best_fit = self.info = None

        # all nodes return the same result
#        if self.returninfo:
        try:
            self.best_pos, self.best_fit, self.info = result[0]
        except:
            log_warn("An exception occurred on the servers")
            return

        if self.parameters.argtype == 'keywords':
            self.best_pos = [dict([(key, self.best_pos[i][key][0])\
                             for key in self.best_pos[i].keys()])
                             for i in xrange(len(self.best_pos))]
            self.best_params = dict([(key, [self.best_pos[i][key]\
                                     for i in xrange(len(self.best_pos))])
                                     for key in self.best_pos[0].keys()])

        self.results = []
        for i in xrange(self.groups):
            groupresult = GroupOptimizationResult(i,
                                                  self.best_pos[i],
                                                  self.best_fit[i])
            if self.returninfo:
                groupresult.info = self.info[i]
            self.results.append(groupresult)
        # flatten lists if only 1 group
        if self.groups == 1:
            self.best_pos = self.best_pos[0]
            self.best_fit = self.best_fit[0]
            if self.returninfo:
                self.info = self.info[0]

    def __getitem__(self, key):
        if type(key) is str:
            if self.groups == 1:
                return self.best_pos[key]
            else:
                return [self.best_pos[g][key] for g in xrange(self.groups)]
        if type(key) is int:
            return self.results[key]

    def __repr__(self):
        result = "Best position: " + str(self.best_pos)
        result += "\n"
        result += "Best fitness: " + str(self.best_fit)
        return result


class Fitness(object):
    """
    The base class from which any fitness class must derive.
    When using several CPUs or several machines, every node contains
    its own instance of this class.
    The derived class must implement two methods:

    ``initialize(self, *args, **kwds)``
        This method initializes the fitness function at the beginning
        of the optimization. The arguments are provided from an optimization
        function like :func:`minimize` or :func:`maximize`, with the
        parameters ``args`` and ``kwds``.

    ``evaluate(self, **kwds)``.
        This method evaluates the fitness against particle positions.
        For keyword-like fitness functions, ``kwds`` is a dictionary where
        keys are parameter names, and values are vectors of parameter values.
        This method must return a vector with fitness values for all particles.

    In addition, several properties are available in this class:

    ``self.dimension``
        The dimension of the state space, or the number of optimizing
         parameters

    ``self.popsize``
        The total population size for each group across all nodes.

    ``self.subpopsize``
        The population size for each group on this node.

    ``self.groups``
        The number of groups.

    ``self.nodesize``
        The population size for all groups on this node.

    ``self.nodecount``
        The number of nodes used for this optimization.

    ``self.shared_data``
        The dictionary with shared data.

    ``self.unit_type``
        The unit type, ``CPU`` or ``GPU``.
    """
    def __init__(self,   dimension,
                         popsize,
                         subpopsize,
                         groups,
                         nodesize,
                         nodecount,
                         shared_data,
                         unit_type,
                         init_args,
                         init_kwds):
        self.dimension = dimension
        self.popsize = popsize
        self.subpopsize = subpopsize
        self.groups = groups
        self.nodesize = nodesize
        self.nodecount = nodecount
        self.shared_data = shared_data
        self.unit_type = unit_type
        self.init_args = init_args
        self.init_kwds = init_kwds
        self.initialize(*init_args, **init_kwds)

    def initialize(self, *args, **kwds):
        log_debug("<initialize> method not implemented")

    def evaluate(self, *args, **kwds):
        raise Exception("<evaluate> method not implemented")

    def __call__(self, *args, **kwds):
        return self.evaluate(*args, **kwds)
