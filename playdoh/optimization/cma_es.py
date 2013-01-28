from ..synchandler import *
from ..debugtools import *
from optimization import *
from algorithm import *
from cma_utils import *
from numpy import zeros, ones, array, \
inf, tile, maximum, minimum, argsort, \
argmin, floor, where, unique, squeeze, mod
from numpy.random import seed
from scipy import linalg
import os
from time import time

__all__ = ['CMAES']


class CMAES(OptimizationAlgorithm):
    """
    Covariance Matrix Adaptation Evolution Strategy algorithm
    See the
    `wikipedia entry on CMAES <http://en.wikipedia.org/wiki/CMA - ES>`__
    and also the author's website <http://www.lri.fr/~hansen/cmaesintro.html>`

    Optimization parameters:

    ``proportion_selective = 0.5``
        This parameter (refered to as mu in the CMAES algorithm) is the
        proportion (out of 1) of the entire population that is selected and
        used to update the generative distribution. (note for different groups
        case: this parameter can only have one value, i.e. every group
        will have the same value (the first of the list))

    ``bound_strategy = 1``:
        In the case of a bounded problem, there are two ways to handle the new
        generated points which fall outside the boundaries.
        (note for different groups case: this parameter can only have one
        value, i.e. every group will have the same
        value (the first of the list))

        ``bound_strategy = 1``. With this strategy, every point outside the
        domain is repaired, i.e. it is projected to its nearset possible
        value :math:`x_{repaired}`. In other words, components that are
        infeasible in :math:`x` are set to the (closest) boundary value
        in :math:`x_{repaired}` The fitness function on the repaired search
        points is evaluated and a penalty which depends on the distance to
        the repaired solution is added
        :math:`f_{fitness}(x) = f(x_{repaired})+\gamma \|x-x_{repaired}\|^{2}`
        The repaired solution is disregarded afterwards.

        ``bound_strategy = 2``. With this strategy any infeasible solution x is
        resampled until it become feasible. It should be used only if the
        optimal solution is not close to the infeasible domain.

    See p.28 of <http://www.lri.fr/~hansen/cmatutorial.pdf> for more details
    ``gamma``:

        ``gamma`` is the weight :math:`\gamma` in the previously introduced
        penalty function. (note for different groups case: this parameter can
        only have one value, i.e. every group will have the same
        value (the first of the list))
    """
    @staticmethod
    def default_optparams():
        """
        Returns the default values for optparams
        """
        optparams = dict()
        optparams['proportion_selective'] = 0.5
        optparams['alpha'] = 500   # Boundaries and Constraints:
        optparams['bound_strategy'] = 1
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
        particle positions.
        """
        if os.name == 'posix':
            t = time()
            t = mod(int(t * 10000), 1000000)
            seed(int(t + self.index * 1000))

        initial_solution = zeros(self.ndimensions)

        self.bound_strategy = self.optparams['bound_strategy'][0]
        self.mu = floor(self.popsize *\
                                    self.optparams['proportion_selective'][0])
        opt = dict()
        opt['CMAmu'] = self.mu
        opt['popsize'] = self.popsize
        opt['maxiter'] = self.maxiter + 1
        opt['verb_disp'] = False
        self.es = [0] * self.groups
        for igroup in xrange(self.groups):
            self.es[igroup] = CMAEvolutionStrategy(initial_solution, .1, opt)
            self.es[igroup].sigma_iter = []
            self.es[igroup].iteration = 0

        self.alpha = array(self.optparams['alpha'])
        if self.scaling == None:
            self.Xmin = tile(self.boundaries[:, 0].\
                            reshape(self.ndimensions, 1), (1, self.nodesize))
            self.Xmax = tile(self.boundaries[:, 1].\
                             reshape(self.ndimensions, 1), (1, self.nodesize))
        else:
            self.Xmin = tile(self.parameters.scaling_func(self.\
            boundaries[:, 0]).reshape(self.ndimensions, 1), (1, self.nodesize))
            self.Xmax = tile(self.parameters.scaling_func(self.\
            boundaries[:, 1]).reshape(self.ndimensions, 1), (1, self.nodesize))

        # Preparation of the optimization algorithm

        self.fitness_gbest = inf * ones((self.groups, self.mu))
        self.X_gbest = zeros((self.groups, self.ndimensions, self.mu))
        if self.mu <= self.subpopsize:
            self.fitness_lbest = inf * ones((self.groups, self.mu))
            self.X_lbest = zeros((self.groups, self.ndimensions, self.mu))
        else:
            self.fitness_lbest = inf * ones((self.groups, self.subpopsize))
            self.X_lbest = zeros((self.groups, self.ndimensions,\
                                                         self.subpopsize))

        if self.return_info:
            self.best_fitness = zeros((self.groups, self.maxiter))
            self.best_particule = zeros((self.groups, self.ndimensions,\
                                                                self.maxiter))
            self.dist_mean = zeros((self.groups, self.ndimensions,\
                                                             self.maxiter))
            self.dist_std = zeros((self.groups, self.ndimensions,\
                                                                 self.maxiter))

        self.mean_distr = zeros((self.groups, self.ndimensions))
        self.std_distr = zeros((self.groups, self.ndimensions))
        self.C = zeros((self.groups, self.ndimensions, self.ndimensions))
        self.D = zeros((self.groups, self.ndimensions))
        self.B = zeros((self.groups, self.ndimensions, self.ndimensions))
        self.iteration = 0
        self.X_best_of_all = zeros((self.groups, self.ndimensions))
        self.fitness_best_of_all = [inf] * ones(self.groups)

    def initialize_particles(self):
        self.X = zeros((self.ndimensions, self.nodesize))
        if self.parameters.argtype == 'keywords':
            params = self.parameters.params
            params = [self.parameters.params[name] for name in\
                                                self.parameters.param_names]
            init_mean = zeros(len(params))
            init_std = zeros(len(params))
            for i in xrange(len(params)):
                value = params[i]
                value = array(value).astype('float')
                if len(value) == 2:
                    init_mean[i] = array((value[1] + value[0]) / 2)
                    init_std[i] = array((value[1] - value[0]) / 6)
                elif len(value) == 4:
                    init_mean[i] = array((value[2] + value[1]) / 2)
                    init_std[i] = array((value[2] - value[1]) / 6)
#                print name, init_mean[i], init_std[i], value[2] , value[1]

        else:
            init_range = self.parameters.initrange
            init_mean = (init_range[:, 1] + init_range[:, 0]) / 2
            init_std = (init_range[:, 1] - init_range[:, 0]) / 6

        if self.scaling is not None:
            init_mean = self.parameters.scaling_func(init_mean)
            init_std = 2 * init_std / (self.parameters.scaling_factor_b\
                                           - self.parameters.scaling_factor_a)

        for igroup in xrange(self.groups):
            self.es[igroup].sigma = init_std
            self.es[igroup].mean = init_mean
        self.X = array(self.es[0].ask(number=self.nodesize)).T

    def communicate_before_update(self):
        ## the workers send their mu best positions and fitness and
        ## the master receives them
        if self.index > 0:
            # WORKERS
#            log.info("I'm worker #%d" % self.index)

            # sends the mu best local position and their fitness
            to_master = 'to_master_%d' % self.index
            self.tubes.push(to_master, (self.X_lbest, self.fitness_lbest))

        else:
            # MASTER
            if len(self.nodes) == 1:   # if only one worker sort the particles
                self.X_gbest = self.X_lbest
                self.fitness_gbest = self.fitness_lbest
                for igroup in xrange(self.groups):
                    indices_population_sorted = argsort(self.\
                                        fitness_gbest[igroup, :])[0:self.mu]
                    self.fitness_gbest[igroup, :] = self.\
                               fitness_gbest[igroup, indices_population_sorted]
                    self.X_gbest[igroup, :, :] = self.X_gbest[igroup, :,\
                                                   indices_population_sorted].T
            else:
                # receives the best from workers
                #if len(self.nodes)>1:
                fitness_temp = zeros((self.groups, self.popsize))
                X_temp = zeros((self.groups, self.ndimensions, self.popsize))
                ind_data = 0
                # list of incoming tubes, ie from workers
                for node in self.nodes:

                    if node.index != 0:
                        tube = 'to_master_%d' % node.index
                        self.X_lbest, self.fitness_lbest = self.tubes.pop(tube)

                    for igroup in xrange(self.groups):
                        worker_len = len(self.fitness_lbest[igroup, :])
                        fitness_temp[igroup, ind_data:ind_data + worker_len] =\
                                                  self.fitness_lbest[igroup, :]
                        X_temp[igroup, :, ind_data:ind_data + worker_len] =\
                                                     self.X_lbest[igroup, :, :]
                    ind_data = ind_data + worker_len

                for igroup in xrange(self.groups):
                    indices_population_sorted = argsort(fitness_temp[igroup,\
                                                                :])[0:self.mu]
                    self.fitness_gbest[igroup, :] = fitness_temp[igroup,\
                                                    indices_population_sorted]
                    self.X_gbest[igroup, :, :] = X_temp[igroup, :,\
                                                  indices_population_sorted].T

    def communicate_after_update(self):
     ##the workers send their mu best positions and
     ## fitness and the mater receives them
        if self.index > 0:
            # WORKERS
#            log.info("I'm worker #%d" % self.index)
            # receives the updated generation distribution parameters
            to_worker = 'to_worker_%d' % self.index
            # set the new distribution in the CMA class
            (mean, sigma, C, D, B) = self.tubes.pop(to_worker)
            for igroup in xrange(self.groups):
                self.es[igroup].sigma = sigma[igroup, :]
                self.es[igroup].mean = mean[igroup, :]
                self.es[igroup].C = C[igroup, :]
                self.es[igroup].D = D[igroup, :]
                self.es[igroup].B = B[igroup, :]

        else:
            # MASTER
            # sends the updated distribution parameters
            for node in self.nodes:  # list of outcoming tubes, ie to workers
                if node.index == 0:
                    continue
                tube = 'to_worker_%d' % node.index
                self.tubes.push(tube, (self.mean_distr, self.std_distr,\
                                                      self.C, self.D, self.B))

    def find_best_local(self):
        for igroup in xrange(self.groups):
            fitness = self.fitness[igroup * self.subpopsize:(igroup + 1) *\
                                                              self.subpopsize]
            if self.mu <= self.subpopsize:
                indices_population_sorted = argsort(fitness)[0:self.mu]
            else:
                indices_population_sorted = argsort(fitness)
            self.fitness_lbest[igroup, :] = self.fitness[igroup *\
                                   self.subpopsize + indices_population_sorted]
            self.X_lbest[igroup, :, :] = self.X[:, igroup * self.subpopsize +\
                                                     indices_population_sorted]

    def iterate(self, iteration, fitness):
        self.iteration = iteration
        self.fitness = fitness
        # find mu best local
        self.find_best_local()
        # communicate with other nodes
        self.communicate_before_update()
        # updates the particle positions (only on MASTER)
        if self.index == 0:
            for igroup in xrange(self.groups):
                self.es[igroup].tell(self.X_gbest[igroup, :, :].T,\
                                                 self.fitness_gbest[igroup, :])
                self.mean_distr[igroup, :] = self.es[igroup].mean
                self.std_distr[igroup, :] = self.es[igroup].sigma
                self.es[igroup].sigma_iter.append(self.es[igroup].sigma)
                self.es[igroup].iteration = iteration
                self.C[igroup, :] = self.es[igroup].C
                self.D[igroup, :] = self.es[igroup].D
                self.B[igroup, :] = self.es[igroup].B
                ind_best = argmin(self.fitness_gbest[igroup, :])
                best_fitness = min(self.fitness_gbest[igroup, :])
                if self.fitness_best_of_all[igroup] > best_fitness:
                    self.X_best_of_all[igroup, :] = self.X_gbest[igroup, :,\
                                                                     ind_best]
                    self.fitness_best_of_all[igroup] = best_fitness
#                print self.fitness_best_of_all
        # the master send the new distribution parameters,
        # the workers receive them
        self.communicate_after_update()

        if self.return_info:
            self.collect_info()
        #new particules are generated

        for igroup in xrange(self.groups):
            if self.bound_strategy is not 2:
                # get list of new solutions
                temp = self.es[igroup].ask(number=self.subpopsize)
                for k in xrange(self.subpopsize):
                    self.X[:, igroup * self.subpopsize + k] = temp[k]

            else:  # resample until everything is in the boundaries
                temp = self.es[igroup].ask(number=self.subpopsize)
                self.Xtemp = zeros((self.ndimensions, self.subpopsize))
                for k in range(self.subpopsize):
                    self.Xtemp[:, k] = temp[k]

                self.Xtemp_new = maximum(self.Xtemp, self.Xmin[:,\
                                                             :self.subpopsize])
                self.Xtemp_new = minimum(self.Xtemp, self.Xmax[:,\
                                                             :self.subpopsize])
                self.ind_changed_sample = unique(where(self.Xtemp_new !=\
                                                                self.Xtemp)[1])
                self.ind_changed = where(self.Xtemp_new != self.Xtemp)
                #print len(self.ind_changed_sample)
                while len(self.ind_changed_sample) != 0:
                    #print len(self.ind_changed_sample)
                    temp = self.es[igroup].ask(number=len(self.ind_changed[0]))
                    for k in xrange(len(self.ind_changed[0])):
                        self.Xtemp[self.ind_changed[0][k],\
                      self.ind_changed[1][k]] = temp[k][self.ind_changed[0][k]]
                    self.Xtemp_new = maximum(self.Xtemp,\
                                                self.Xmin[:, :self.subpopsize])
                    self.Xtemp_new = minimum(self.Xtemp,\
                                                self.Xmax[:, :self.subpopsize])
                    self.ind_changed_sample = unique(where(self.Xtemp_new !=\
                                                                self.Xtemp)[1])
                    self.ind_changed = where(self.Xtemp_new != self.Xtemp)
                self.X[:, igroup * self.subpopsize:(igroup + 1) *\
                                            self.subpopsize] = self.Xtemp_new

    def pre_fitness(self):
        if self.bound_strategy == 1:
            if any(self.boundaries[:, 0] != -inf) or\
                                            any(self.boundaries[:, 1] != inf):
                self.Xold = self.X
                self.X = maximum(self.X, self.Xmin)
                self.X = minimum(self.X, self.Xmax)
                self.ind_changed = unique(where(self.X != self.Xold)[1])
                #print len(self.ind_changed)

    def post_fitness(self, fitness):
        #print fitness
        if self.bound_strategy == 1:
            if any(self.boundaries[:, 0] != -inf) or\
                                             any(self.boundaries[:, 1] != inf):
                if len(self.ind_changed) is not 0:
                    fitness[self.ind_changed] = fitness[self.ind_changed] +\
                                      self.alpha[0] * linalg.norm(self.Xold[:,\
                        self.ind_changed] - self.X[:, self.ind_changed]) ** 2
                self.X = self.Xold

        return fitness

    def collect_info(self):
        for igroup in xrange(self.groups):
            if self.index == 0:  # only mater info
                self.best_fitness[igroup, self.iteration] = \
                                            self.fitness_best_of_all[igroup]
                self.dist_std[igroup, :, self.iteration] =\
                                                   self.std_distr[igroup, :]
                if self.scaling == None:
                    self.dist_mean[igroup, :, self.iteration] =\
                                                   self.mean_distr[igroup, :]
                    self.best_particule[igroup, :, self.iteration] =\
                                                self.X_best_of_all[igroup, :]
                else:
                    self.dist_mean[igroup, :, self.iteration] =\
                                         self.parameters.unscaling_func(self.\
                                                        mean_distr[igroup, :])
                    self.best_particule[igroup, :, self.iteration] =\
                                         self.parameters.unscaling_func(self.\
                                                     X_best_of_all[igroup, :])

    def get_info(self):
        info = list()
        if self.return_info:
            for igroup in xrange(self.groups):
                temp = dict()
                temp['dist_std'] = squeeze(self.dist_std[igroup, 0, :])
                temp['dist_mean'] = squeeze(self.dist_mean[igroup, :, :])
                temp['best_fitness'] = squeeze(self.best_fitness[igroup, :])
                temp['best_position'] = squeeze(self.\
                                               best_particule[igroup, :, :])
                info.append(temp)
        return info

    def get_result(self):
        best_position = list()
        best_fitness = list()
        self.scaling = None
        for igroup in xrange(self.groups):
            if self.index == 0:
                best_position.append(self.X_best_of_all[igroup, :])
                best_fitness.append(self.fitness_best_of_all[igroup])
            else:
                best_position, best_fitness = [], []
        return (best_position, best_fitness)
