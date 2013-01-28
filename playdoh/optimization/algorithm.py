from ..synchandler import *
from ..debugtools import *
from optimization import *
from numpy import zeros, tile, mod
from numpy.random import rand, seed
import os
from time import time
__all__ = ['OptimizationAlgorithm']


class OptimizationAlgorithm(object):
    def __init__(self,
                 index,
                 nodes,
                 tubes,
                 popsize,
                 subpopsize,
                 nodesize,
                 groups,
                 return_info,
                 maxiter,
                 scaling,
                 parameters,
                 optparams):
        self.index = index
        self.nodes = nodes
        self.nodecount = len(nodes)
        self.scaling = scaling
        self.node = nodes[index]
        self.tubes = tubes
        self.return_info = return_info
        self.popsize = popsize
        self.subpopsize = subpopsize
        self.nodesize = nodesize
        self.groups = groups

#        self.nparticles = nparticles # number of columns of X
#        self.ntotal_particles = ntotal_particles

        self.ndimensions = parameters.param_count  # number of rows of X
        self.parameters = parameters
        self.maxiter = maxiter
        self.optparams = optparams

    @staticmethod
    def default_optparams():
        """
        MUST BE OVERRIDEN
        Returns the default values for optparams
        """
        return {}

    @staticmethod
    def get_topology(node_count):
        """
        MUST BE OVERRIDEN
        Returns the list of tubes (name, i1, i2)
        as a function of the number of nodes
        """
        return []

    def initialize(self):
        """
        MAY BE OVERRIDEN
        Initializes the optimization algorithm.
        X is the matrix of initial particle positions.
        X.shape == (ndimensions, particles)
        """
        pass

    def initialize_particles(self):
        """
        MAY BE OVERRIDEN
        Sets the initial positions of the particles
        By default, samples uniformly
        params is a list of tuples (bound_min, init_min, init_max, bound_max)
        """
        # initializes the particles
        if os.name == 'posix':
            t = time()
            t = mod(int(t * 10000), 1000000)
            seed(int(t + self.index * 1000))

        if self.parameters.argtype == 'keywords':
            params = [self.parameters.params[name]\
                       for name in self.parameters.param_names]
            X = zeros((self.ndimensions, self.nodesize))
            for i in xrange(len(params)):
                value = params[i]
                if len(value) == 2:
                    # One default interval,
                    # no boundary counditions on parameters
                    X[i, :] = value[0] + (value[1] - value[0]) \
                    * rand(self.nodesize)
                elif len(value) == 4:
                    # One default interval,
                    # value = [min, init_min, init_max, max]
                    X[i, :] = value[1] + (value[2] - value[1])\
                     * rand(self.nodesize)
            self.X = X
        else:
            initmin = tile(self.parameters.initrange[:, 0].reshape((-1, 1)),\
                            (1, self.nodesize))
            initmax = tile(self.parameters.initrange[:, 1].reshape((-1, 1)),\
                            (1, self.nodesize))
            self.X = initmin + (initmax - initmin) * \
            rand(self.ndimensions, self.nodesize)

#    def set_boundaries(self, params):
#        """
#        MAY BE OVERRIDEN
#        Sets the boundaries as a
#        ndimensionsx2 array, boundaries[i,:]=[min,max] for
#        parameter #i
#        """
#        boundaries = zeros((self.ndimensions, 2))
#        for i in xrange(len(params)):
#            value = params[i]
#            if len(value) == 2:
#                # One default interval, no boundary
#                counditions on parameters
#                boundaries[i,:] = [-inf, inf]
#            elif len(value) == 4:
#                # One default interval,
#                  value = [min, init_min, init_max, max]
#                boundaries[i,:] = [value[0], value[3]]
#        self.boundaries = boundaries

    def pre_fitness(self):
        """
        MAY BE OVERRIDEN
        """
        pass

    def post_fitness(self, fitness):
        """
        MAY BE OVERRIDEN
        """
        return fitness

    def iterate(self, iteration, fitness):
        """
        MUST BE OVERRIDEN
        """
        pass

    def get_info(self):
        """
        MAY BE OVERRIDEN
        """
        pass

    def get_result(self):
        """
        MUST BE OVERRIDEN
        Returns (X_best, fitness_best)
        X_best.shape = (ndimensions, groups)
        fitness_best.shape = (groups)
        """
        pass
