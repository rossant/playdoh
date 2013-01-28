from ..synchandler import *
from ..debugtools import *
from algorithm import *
from numpy import zeros, ones, inf, tile,\
nonzero, isscalar, maximum, minimum, kron, squeeze
from numpy.random import rand


__all__ = ['PSO']


class PSO(OptimizationAlgorithm):
    """
    Particle Swarm Optimization algorithm.
    See the
    `wikipedia entry on PSO
    <http://en.wikipedia.org/wiki/Particle_swarm_optimization>`__.

    Optimization parameters:

    ``omega``
        The parameter ``omega`` is the "inertial constant"

    ``cl``
        ``cl`` is the "local best" constant affecting how much
         the particle's personal best position influences its movement.

    ``cg``
        ``cg`` is the "global best" constant affecting how much the global best
        position influences each particle's movement.

    See the
    `wikipedia entry on PSO
    <http://en.wikipedia.org/wiki/Particle_swarm_optimization>`__
    for more details (note that they use ``c_1`` and ``c_2`` instead of ``cl``
    and ``cg``). Reasonable values are (.9, .5, 1.5), but experimentation
    with other values is a good idea.
    """
    @staticmethod
    def default_optparams():
        """
        Returns the default values for optparams
        """
        optparams = dict(omega=.8,
                         cl=.1,
                         cg=.1)
        return optparams

    @staticmethod
    def get_topology(node_count):
        topology = []
        # 0 is the master, 1..n are the workers
        if node_count > 1:
            for i in xrange(1, node_count):
                topology.extend([('to_master_%d' % i, i, 0),
                                 ('to_worker_%d' % i, 0, i)])
        return topology

    def initialize(self):
        """
        Initializes the optimization algorithm. X is the matrix of initial
        particle positions. X.shape == (ndimensions, nparticles)
        """
        # self.optparams[k] is a list (one element per group)
        self.omega = tile(kron(self.optparams['omega'],\
                                                       ones(self.subpopsize)),\
                                                         (self.ndimensions, 1))
        self.cl = tile(kron(self.optparams['cl'], ones(self.subpopsize)),\
                                                         (self.ndimensions, 1))
        self.cg = tile(kron(self.optparams['cg'], ones(self.subpopsize)),\
                                                         (self.ndimensions, 1))

        self.V = zeros((self.ndimensions, self.nodesize))

        if self.scaling == None:
            self.Xmin = tile(self.boundaries[:, 0].\
                            reshape(self.ndimensions, 1), (1, self.nodesize))
            self.Xmax = tile(self.boundaries[:, 1].\
                            reshape(self.ndimensions, 1), (1, self.nodesize))
        else:
            self.Xmin = tile(self.parameters.scaling_func(self.\
          boundaries[:, 0]).reshape(self.ndimensions, 1), (1, self.nodesize))
            self.Xmax = tile(self.parameters.\
             scaling_func(self.boundaries[:, 1]).reshape(self.ndimensions, 1),\
                                                           (1, self.nodesize))

        # Preparation of the optimization algorithm
        self.fitness_lbest = inf * ones(self.nodesize)
        self.fitness_gbest = inf * ones(self.groups)
        self.X_lbest = zeros((self.ndimensions, self.nodesize))
        self.X_gbest = zeros((self.ndimensions, self.groups))
        self.best_fitness = zeros((self.maxiter, self.groups))
        self.best_particule = zeros((self.ndimensions, self.maxiter,\
                                                                 self.groups))

    def get_global_best(self):
        """
        Returns the global best pos/fit on the current machine
        """
        for group in xrange(self.groups):
            fitness = self.fitness[group * self.subpopsize:(group + 1) *\
                                                             self.subpopsize]
            X = self.X[:, group * self.subpopsize:(group + 1) *\
                                                             self.subpopsize]
            min_fitness = fitness.min()
            if min_fitness < self.fitness_gbest[group]:
                index_gbest = nonzero(fitness == min_fitness)[0]
                if not(isscalar(index_gbest)):
                    index_gbest = index_gbest[0]
                self.X_gbest[:, group] = X[:, index_gbest]
                self.fitness_gbest[group] = min_fitness
        return self.fitness_gbest

    def get_local_best(self):
        indices_lbest = nonzero(self.fitness < self.fitness_lbest)[0]
        if (len(indices_lbest) > 0):
            self.X_lbest[:, indices_lbest] = self.X[:, indices_lbest]
            self.fitness_lbest[indices_lbest] = self.fitness[indices_lbest]

    def communicate(self):
        # communicate with master to have the absolute global best
        if self.index > 0:
            # WORKERS
            log_debug("I'm worker #%d" % self.index)

            # sends the temp global best to the master
            to_master = 'to_master_%d' % self.index
            self.tubes.push(to_master, (self.X_gbest, self.fitness_gbest))

            # receives the absolute global best from the master
            to_worker = 'to_worker_%d' % self.index
            (self.X_gbest, self.fitness_gbest) = self.tubes.pop(to_worker)
        else:
            # MASTER
            log_debug("I'm the master (#%d)" % self.index)

            # best values for each node, including the master (current node)
            X_gbest = self.X_gbest
            fitness_gbest = self.fitness_gbest

            # receives the temp global best from the workers
            for node in self.nodes:  # list of incoming tubes, ie from workers
                if node.index == 0:
                    continue
                tube = 'to_master_%d' % node.index
                log_debug("Receiving best values from <%s>..." % tube)
                X_gbest_tmp, fitness_gbest_tmp = self.tubes.pop(tube)

                for group in xrange(self.groups):
                    # this one is better
                    if fitness_gbest_tmp[group] < fitness_gbest[group]:
                        X_gbest[:, group] = X_gbest_tmp[:, group]
                        fitness_gbest[group] = fitness_gbest_tmp[group]

            # sends the absolute global best to the workers
            for node in self.nodes:  # list of outcoming tubes, ie to workers
                if node.index == 0:
                    continue
                tube = 'to_worker_%d' % node.index
                self.tubes.push(tube, (X_gbest, fitness_gbest))

            self.X_gbest = X_gbest
            self.fitness_gbest = fitness_gbest

    def update(self):
        # update matrix
        rl = rand(self.ndimensions, self.nodesize)
        rg = rand(self.ndimensions, self.nodesize)
        X_gbest_expanded = kron(self.X_gbest, ones((1, self.subpopsize)))
        self.V = self.omega * self.V + self.cl * rl *\
                                                   (self.X_lbest - self.X) +\
                                     self.cg * rg * (X_gbest_expanded - self.X)
        self.X = self.X + self.V

        # constrain boundaries
        self.X = maximum(self.X, self.Xmin)
        self.X = minimum(self.X, self.Xmax)

    def iterate(self, iteration, fitness):
        """
        Must return the new population
        """
#        log_debug("iteration %d/%d" % (iteration+1, self.iterations))
        self.iteration = iteration
        self.fitness = fitness

        # get local/global best on this machine
        self.get_local_best()
        self.get_global_best()

        # communicate with other nodes to have
        #the absolute global best position
        self.communicate()

        # updates the particle positions
        self.update()

        if self.return_info is True:
            self.collect_info()

    def collect_info(self):
#        print self.iteration
        self.best_fitness[self.iteration, :] = self.fitness_gbest
        self.best_particule[:, self.iteration, :] = self.X_gbest

    def get_info(self):
        info = list()
        if self.return_info is True:
            for igroup in xrange(self.groups):
                temp = dict()
                temp['best_fitness'] = squeeze(self.best_fitness[:, igroup])
                temp['best_position'] = squeeze(self.best_particule[:, :,\
                                                                     igroup])
                info.append(temp)

        return info

    def get_result(self):
        """
        Returns (X_best, fitness_best)
        """

        best_position = list()
        best_fitness = list()
        for igroup in xrange(self.groups):
            if self.index == 0:
                if self.scaling == None:
                    best_position.append(squeeze(self.X_gbest[:, igroup]))
                    best_fitness.append(self.fitness_gbest[igroup])
                else:
                    best_position.append(squeeze(self.parameters.\
                                    unscaling_func(self.X_gbest[:, igroup])))
                    best_fitness.append(self.fitness_best_of_all[igroup])

            else:
                best_position, best_fitness = [], []
#        print best_position, best_fitness
        return (best_position, best_fitness)
