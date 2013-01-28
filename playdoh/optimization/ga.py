from ..synchandler import *
from ..debugtools import *
from optimization import *
from algorithm import *
from numpy import inf, zeros, tile, nonzero, \
maximum, minimum, ceil, floor, argsort, array, \
sort, mod, cumsum, sum, arange, ones, argmin, squeeze
from numpy.random import rand, randint, randn

__all__ = ['GA']


class GA(OptimizationAlgorithm):
    """
    Standard genetic algorithm.
    See the
    `wikipedia entry on GA <http://en.wikipedia.org/wiki/Genetic_algorithm>`__

    If more than one worker is used, it works in an island topology, i.e. as a
    coarse - grained parallel genetic algorithms which assumes
    a population on each of the computer nodes and migration of individuals
    among the nodes.

    Optimization parameters:

    ``proportion_parents = 1``
        proportion (out of 1) of the entire population taken
        as potential parents.

    ``migration_time_interval = 20``
           whenever more than one worker is used, it is the number of
           iteration at which a migration happens.
           (note for different groups case: this parameter can only have
           one value, i.e. every group will have the same value
           (the first of the list))

    ``proportion_migration = 0.2``
          proportion (out of 1) of the island population that will migrate to
          the next island (the best one) and also the worst that will be
          replaced by the best of the previous island. (note for different
          groups case: this parameter can only have one value, i.e. every group
          will have the same value (the first of the list))


    ``proportion_xover = 0.65``
        proportion (out of 1) of the entire population which will
        undergo a cross over.

    ``proportion_elite = 0.05``
        proportion (out of 1) of the entire population which will be kept
        for the next generation based on their best fitness.

        The proportion of mutation is automatically set to
         ``1 - proportion_xover - proportion_elite``.

    ``func_selection = 'stoch_uniform'``
        This function define the way the parents are chosen
        (it is the only one available). It lays out a line in
        which each parent corresponds to a section of the line of length
        proportional to its scaled value. The algorithm moves along the
        line in steps of equal size. At each step, the algorithm allocates
        a parent from the section it lands on. The first step is
        a uniform random number less than the step size.


    ``func_xover = 'intermediate'``

        ``func_xover`` specifies the function that performs the crossover.
         The following ones are available:

        * `intermediate`: creates children by taking a random weighted average
           of the parents. You can specify the weights by a single parameter,
           ``ratio_xover`` (which is 0.5 by default). The function creates the
           child from parent1 and parent2 using the  following formula::

              child = parent1 + rand * Ratio * ( parent2 - parent1)

        * `discrete_random`: creates a random binary vector and selects the
           genes where the vector is a 1 from the first parent, and the gene
           where the vector is a 0 from the second parent, and combines the
           genes to form the child.

        * `one_point`: chooses a random integer n between 1 and ndimensions
          and then selects vector entries numbered less than or equal to n
          from the first parent. It then Selects vector entries numbered
          greater than n from the second parent. Finally, it concatenates
          these entries to form a child vector.

        * `two_points`: it selects two random integers m and n between 1 and
           ndimensions. The function selects vector entries numbered less than
           or equal to m from the first parent. Then it selects vector entries
           numbered from m + 1 to n, inclusive, from the second parent. Then
           it selects vector entries numbered greater than n from the first
           parent. The algorithm then concatenates these genes to form
           a single gene.

        * `heuristic`: returns a child that lies on the line containing the two
          parents, a small distance away from the parent with the better
          fitness value in the direction away from the parent with the worse
          fitness value. You can specify how far the child is from the
          better parent by the parameter ``ratio_xover``
          (which is 0.5 by default)

        * `linear_combination`: creates children that are linear combinations
          of the two parents with  the parameter ``ratio_xover``
          (which is 0.5 by default and should be between 0 and 1)::

              child = parent1 + Ratio * ( parent2 - parent1)

          For  ``ratio_xover = 0.5`` every child is an arithmetic mean of
          two parents.

    ``func_mutation = 'gaussian'``

        This function define how the genetic algorithm makes small random
        changes in the individuals in the population to create mutation
        children. Mutation provides genetic diversity and enable the genetic
        algorithm to search a broader space. Different options are available:

        * `gaussian`: adds a random number taken from a Gaussian distribution
          with mean 0 to each entry of the parent vector.

          The 'scale_mutation' parameter (0.8 by default) determines the
          standard deviation at the first generation by
          ``scale_mutation * (Xmax - Xmin)`` where
          Xmax and Xmin are the boundaries.

          The 'shrink_mutation' parameter (0.2 by default) controls how the
          standard deviation shrinks as generations go by::

              :math:`sigma_{i} = \sigma_{i-1}(1-shrink_{mutation} * i/maxiter)`
              at iteration i.

        * `uniform`: The algorithm selects a fraction of the vector entries of
          an individual for mutation, where each entry has a probability
          ``mutation_rate`` (default is 0.1) of being mutated. In the second
          step, the algorithm replaces each selected entry by a random number
          selected uniformly from the range for that entry.

    """
    @staticmethod
    def default_optparams():
        """
        Returns the default values for optparams
        """
        optparams = dict()
        optparams['proportion_elite'] = 0.05
        # proportion_mutation + proportion_xover + proportion_elite = 1
        optparams['proportion_xover'] = 0.65
        optparams['func_scale'] = 'ranking'
        optparams['func_selection'] = 'stoch_uniform'
        optparams['func_xover'] = 'intermediate'
        optparams['func_mutation'] = 'uniform'
        optparams['scale_mutation'] = 0.8
        optparams['shrink_mutation'] = 0.02
        optparams['mutation_rate'] = 0.1
        optparams['migration_time_interval'] = 20
        optparams['proportion_migration'] = 0.2
        optparams['proportion_parents'] = 1
        optparams['ratio_xover'] = 0.5
        return optparams

    @staticmethod
    def get_topology(node_count):
        topology = []
        # 0 is the master, 1..n are the workers
        if node_count > 1:
            for i in xrange(0, node_count):
                topology.extend([('to_next_island_%d' %\
                i, i, mod(i + 1, node_count))])
        return topology

    def initialize(self):
        """
        Initializes the optimization algorithm. X is the
        matrix of initial particle positions.
        X.shape == (ndimensions, nparticles)
        """
        self.time_from_last_migration = 1
        self.fitness_best_of_all = inf

        self.D = self.ndimensions
        self.N = self.subpopsize
        self.nbr_elite = ceil(array(self.optparams['proportion_elite'])\
                                                  * self.N).astype('int')
        self.nbr_xover = floor(array(self.optparams['proportion_xover'])\
                                                   * self.N).astype('int')
        self.mutation_rate = self.optparams['mutation_rate']
        self.nbr_mutation = self.N - self.nbr_elite - self.nbr_xover
        self.nbr_offspring = self.nbr_xover + self.nbr_mutation
        self.nbr_migrants = int(ceil(self.N *\
                               self.optparams['proportion_migration'][0]))
        self.nbr_parents = self.N
        self.migration_time_interval = \
                         int(self.optparams['migration_time_interval'][0])

        if self.scaling == None:
            self.Xmin = tile(self.boundaries[:, 0].\
                              reshape(self.ndimensions, 1), (1, self.nodesize))
            self.Xmax = tile(self.boundaries[:, 1].\
                              reshape(self.ndimensions, 1), (1, self.nodesize))
        else:
            self.Xmin = tile(self.parameters.\
                                scaling_func(self.boundaries[:, 0]).\
                              reshape(self.ndimensions, 1), (1, self.nodesize))
            self.Xmax = tile(self.parameters.\
                               scaling_func(self.boundaries[:, 1]).\
                              reshape(self.ndimensions, 1), (1, self.nodesize))

        # Preparation of the optimization algorithm
        self.fitness_lbest = inf * ones(self.subpopsize)
        self.fitness_gbest = [inf] * ones(self.groups)
        self.X_lbest = zeros((self.groups, self.ndimensions, self.subpopsize))
        self.X_gbest = zeros((self.groups, self.ndimensions))
        self.best_fitness = zeros((self.groups, self.maxiter))
        self.best_particule = zeros((self.groups,\
                                              self.ndimensions, self.maxiter))
        self.X_best_of_all = zeros((self.groups, self.ndimensions))
        self.fitness_best_of_all = [inf] * ones(self.groups)
        self.X_migrants = zeros((self.ndimensions, self.nbr_migrants *\
                                                                 self.groups))
        self.fitness_migrants = zeros(self.nbr_migrants * self.groups)
        self.sigmaMutation = zeros(self.groups)
        for igroup in xrange(self.groups):
            if  self.optparams['func_mutation'][igroup] == 'gaussian':

                Xmin = self.boundaries[:, 0]
                Xmax = self.boundaries[:, 1]
                if (Xmin != -inf * ones(self.D)).any() and (Xmax != inf *\
                                                           ones(self.D)).any():
                    self.sigmaMutation[igroup] = self.\
                            optparams['scale_mutation'][igroup] * (Xmax - Xmin)
                else:      # if boudneries are infinite
                    self.sigmaMutation[igroup] = self.\
                                 optparams['scale_mutation'][igroup] * 1000
#                                 (self.parameters.scaling_factor_b -\
#                                  self.parameters.scaling_factor_a))

    def communicate(self):
        #there is communication only if there are several island
        if len(self.nodes) > 1:

            # sends the  best local position
            # and their fitness to the next neighbouring island
            to_next_island = 'to_next_island_%d' % self.index
            self.tubes.push(to_next_island, (self.X_migrants,\
                                                        self.fitness_migrants))

            # receive the best from the previous neighbouring island
            # (take the next island tude of the previous island)
            from_previous_island = 'to_next_island_%d' %\
                                           mod(self.index - 1, len(self.nodes))
            a, b = self.tubes.pop(from_previous_island)
            for igroup in xrange(self.groups):
                self.X[:, (igroup + 1) * self.subpopsize -\
                           self.nbr_migrants:(igroup + 1) * self.subpopsize] =\
                           a[:, igroup * self.nbr_migrants:(1 + igroup) *\
                           self.nbr_migrants]
                self.fitness[(igroup + 1) * self.subpopsize -\
                           self.nbr_migrants:(igroup + 1) * self.subpopsize] =\
                            b[igroup * self.nbr_migrants:(1 + igroup) *\
                            self.nbr_migrants]
        else:
            pass

    def iterate(self, iteration, fitness):

        self.iteration = iteration
        self.fitness = fitness
        # resort the population from best to worst
        for igroup in xrange(self.groups):
            fitness = self.fitness[igroup * self.subpopsize:(igroup + 1) *\
                                                               self.subpopsize]
            indices_Population_sorted = argsort(fitness)
            self.X[:, igroup * self.subpopsize:(igroup + 1) *\
                                         self.subpopsize] = self.X[:, igroup *\
                                   self.subpopsize + indices_Population_sorted]
            self.fitness[igroup * self.subpopsize:(igroup + 1) *\
                                      self.subpopsize] = self.fitness[igroup *\
                                  self.subpopsize + indices_Population_sorted]

        #migrate if if it is time to do so
        #(communicate does smth if  there is more than one worker)
        if len(self.nodes) > 1:
            if  self.time_from_last_migration == self.migration_time_interval:
                for igroup in xrange(self.groups):
                    self.X_migrants[:, igroup *\
                        self.nbr_migrants:(igroup + 1) * self.nbr_migrants] =\
                                  self.X[:, igroup * self.subpopsize:igroup *\
                                  self.subpopsize + self.nbr_migrants]
                    self.fitness_migrants[igroup *\
                                   self.nbr_migrants:(igroup + 1) *\
                                   self.nbr_migrants] = self.fitness[igroup *\
                                   self.subpopsize:igroup * self.subpopsize +\
                                   self.nbr_migrants]

                self.communicate()
                self.time_from_last_migration = 0
                #print 'migration happened!'
            else:
                self.time_from_last_migration += 1

        for igroup in xrange(self.groups):
            self.igroup = igroup
            self.X_group = self.X[:, igroup * self.subpopsize:(igroup + 1) *\
                                                             self.subpopsize]
            self.fitness_group = self.fitness[igroup *\
                               self.subpopsize:(igroup + 1) * self.subpopsize]
            ##rescale fitness
            self.fitness_scaled = self.scale_fitness()
             ### select parents ######
            self.parents_indices = self.select_parents()
            #compute the island elite from the fitness
            # that has been just computed for X
            self.fitness_lElite, self.X_lElite, self.X_bestIsland,\
                                  self.fitness_bestIsland = self.find_elite()
            ind_best = argmin(self.fitness_group)
            if self.fitness_best_of_all[igroup] > self.fitness_group[ind_best]:
                self.X_best_of_all[igroup, :] = self.X_group[:, ind_best]
                self.fitness_best_of_all[igroup] =\
                                                   self.fitness_group[ind_best]

            ## reproduction: recombine parents with xover and mutation
            self.X[:, igroup * self.subpopsize:(igroup + 1) *\
                                           self.subpopsize] = self.recombine()

#        print self.fitness_best_of_all

        self.collect_info()

        # Boundary checking
        self.X = maximum(self.X, self.Xmin)
        self.X = minimum(self.X, self.Xmax)

    def collect_info(self):
        for igroup in xrange(self.groups):
            self.best_fitness[igroup, self.iteration] = \
                                               self.fitness_best_of_all[igroup]
            self.best_particule[igroup, :, self.iteration] =\
                                                  self.X_best_of_all[igroup, :]

    def get_info(self):
        info = list()
        if self.return_info is True:
            for igroup in xrange(self.groups):
                temp = dict()
                temp['best_fitness'] = squeeze(self.best_fitness[igroup, :])
                temp['best_position'] =\
                                    squeeze(self.best_particule[igroup, :, :])

                info.append(temp)
        return  info

#    def get_result(self):
#        return self.X_best_of_all, self.fitness_best_of_all

    def scale_fitness(self):
        ### rescale the fitness values of the entire population###
        if self.optparams['func_scale'][self.igroup] == 'ranking':
            fitness_scaled = arange(self.N, 0, -1.)
        return fitness_scaled

    def select_parents(self):
        #In a strict generational replacement scheme
        #the size of the mating pool is always equal
        #to the size of the population.

        ### Stochastic univerasl uniform
        if self.optparams['func_selection'][self.igroup] == 'stoch_uniform':
            # wheel has the length of the entire population
            wheel = cumsum(self.fitness_scaled / sum(self.fitness_scaled))
            parents_indices = zeros((self.nbr_parents), 'int')
            stepSize = 1. / self.nbr_parents
            position = rand(1) * stepSize
            lowest = 1

            for iparent in range((self.nbr_parents)):
                # find the wheel position
                for ipos in arange(lowest, wheel.shape[0]):
                    # if the step fall in this chunk ipos of the wheel
                    if(position < wheel[ipos]):
                        parents_indices[iparent] = ipos
                        lowest = ipos
                        break
                position = position + stepSize

        return parents_indices

    def find_elite(self):
        fitness_lElite = self.fitness_group[:self.nbr_elite[self.igroup]]
        X_lElite = self.X_group[:, :self.nbr_elite[self.igroup]]
        fitness_bestIsland = self.fitness_group[0]
        X_bestIsland = self.X_group[:, 0]
        return fitness_lElite, X_lElite, X_bestIsland, fitness_bestIsland

    def recombine(self):
        Xnew = zeros((self.D, self.N))
        Xnew = self.do_crossover(Xnew)
        Xnew = self.do_mutation(Xnew)
        Xnew = self.include_elite(Xnew)
        return Xnew

    def do_crossover(self, Xnew):
        ### CROSSOVER
        if self.optparams['func_xover'][self.igroup] == 'discrete_random':
            for ixover in range(self.nbr_xover[self.igroup]):
                parent1_ind = self.parents_indices[randint(self.nbr_parents,\
                                                                    size=1)[0]]
                parent2_ind = parent1_ind
                # make sure the two parents are not the same
                while parent1_ind == parent2_ind:
                    parent2_ind = self.parents_indices[randint(\
                                                  self.nbr_parents, size=1)[0]]
                vec = randint(0, 2, self.D)
                Xnew[nonzero(vec), ixover] = self.X_group[nonzero(vec),\
                                                                   parent1_ind]
                vec = abs(vec - 1)
                Xnew[nonzero(vec), ixover] = self.X_group[nonzero(vec),\
                                                                   parent2_ind]

        if self.optparams['func_xover'][self.igroup] == 'one_point':
            for ixover in range(self.nbr_xover[self.igroup]):
                parent1_ind = self.parents_indices[randint(self.nbr_parents,\
                                                                   size=1)[0]]
                parent2_ind = parent1_ind
                while parent1_ind == parent2_ind:
                    parent2_ind = self.parents_indices[randint(\
                                                 self.nbr_parents, size=1)[0]]

                split_point = randint(self.D, size=1)[0]
                if split_point != 0:
                    Xnew[:split_point, ixover] = self.X_group[:split_point,\
                                                                   parent1_ind]
                else:
                    Xnew[split_point, ixover] = self.X_group[split_point,\
                                                                   parent1_ind]
                if split_point != self.D:
                    Xnew[split_point:, ixover] = self.X_group[split_point:,\
                                                                   parent2_ind]
                else:
                    Xnew[split_point, ixover] = self.X_group[split_point,\
                                                                   parent2_ind]

        if self.optparams['func_xover'][self.igroup] == 'two_points':
            for ixover in range(self.nbr_xover[self.igroup]):
                parent1_ind = self.parents_indices[randint(self.nbr_parents,\
                                                                    size=1)[0]]
                parent2_ind = parent1_ind
                while parent1_ind == parent2_ind:
                    parent2_ind = self.parents_indices[randint(\
                                                  self.nbr_parents, size=1)[0]]

                split_points1 = randint(self.D, size=1)
                split_points2 = split_points1
                while split_points1 == split_points2:
                    split_points2 = randint(self.D, size=1)
                split_points = sort(array([split_points1, split_points2]))

                if split_points[0] != 0:
                    Xnew[:split_points[0], ixover] = \
                                    self.X_group[:split_points[0], parent1_ind]
                else:
                    Xnew[split_points[0], ixover] = \
                                     self.X_group[split_points[0], parent1_ind]

                Xnew[split_points[0]:split_points[1], ixover] = \
                     self.X_group[split_points[0]:split_points[1], parent2_ind]

                if split_points[1] != self.D:
                    Xnew[split_points[1]:, ixover] = \
                                    self.X_group[split_points[1]:, parent1_ind]
                else:
                    Xnew[split_points[1], ixover] = \
                                     self.X_group[split_points[1], parent1_ind]

        if self.optparams['func_xover'][self.igroup] == 'heuristic':
            for ixover in range(self.nbr_xover[self.igroup]):
                parent1_ind = self.parents_indices[randint(self.nbr_parents,\
                                                                   size=1)[0]]
                parent2_ind = parent1_ind
                while parent1_ind == parent2_ind:
                    parent2_ind = self.parents_indices[randint(\
                                                 self.nbr_parents, size=1)[0]]

                if self.fitness_group[parent1_ind] >= \
                                              self.fitness_group[parent2_ind]:
                    Xnew[:, ixover] = self.X_group[:, parent2_ind] + \
                                 self.optparams['ratio_xover'][self.igroup] *\
                                   (self.X_group[:, parent1_ind] -\
                                   self.X_group[:, parent2_ind])
                else:
                    Xnew[:, ixover] = self.X_group[:, parent1_ind] + \
                                 self.optparams['ratio_xover'][self.igroup] *\
                                              (self.X_group[:, parent2_ind] -\
                                                 self.X_group[:, parent1_ind])

        if self.optparams['func_xover'][self.igroup] == 'intermediate':
            for ixover in range(self.nbr_xover[self.igroup]):
                parent1_ind = self.parents_indices[randint(self.nbr_parents,\
                                                                   size=1)[0]]
                parent2_ind = parent1_ind
                while parent1_ind == parent2_ind:
                    parent2_ind = self.parents_indices[randint(\
                                                 self.nbr_parents, size=1)[0]]
                Xnew[:, ixover] = self.X_group[:, parent1_ind] + (-0.25 +\
                                                       1.25 * rand(self.D)) *\
                                             (self.X_group[:, parent2_ind] -\
                                               self.X_group[:, parent1_ind])

        if self.optparams['func_xover'][self.igroup] == 'linear_combination':
            for ixover in range(self.nbr_xover[self.igroup]):
                parent1_ind = self.parents_indices[randint(self.nbr_parents,\
                                                                    size=1)[0]]
                parent2_ind = parent1_ind
                while parent1_ind == parent2_ind:
                    parent2_ind = self.parents_indices[randint(\
                                                  self.nbr_parents, size=1)[0]]

                Xnew[:, ixover] = self.X_group[:, parent1_ind] + \
                                 self.optparams['ratio_xover'][self.igroup] *\
                                              (self.X_group[:, parent2_ind] -\
                                                self.X_group[:, parent1_ind])
        return Xnew

    def do_mutation(self, Xnew):
                #### MUTATION
        if self.optparams['func_mutation'][self.igroup] == 'gaussian':
            for imut in range(self.nbr_mutation[self.igroup]):
                Xnew[:, self.nbr_xover[self.igroup] + imut] = \
self.X_group[:, self.parents_indices[randint(self.nbr_parents, size=1)[0]]] +\
self.sigmaMutation[self.igroup] * randn(self.D)
            self.sigmaMutation[self.igroup] =\
                         self.sigmaMutation[self.igroup] *\
                         (1 - self.optparams['shrink_mutation'][self.igroup] *\
                                                 self.iteration / self.maxiter)

        if self.optparams['func_mutation'][self.igroup] == 'uniform':

            for imut in range(self.nbr_mutation[self.igroup]):

                Xnew[:, self.nbr_xover[self.igroup] + imut] = \
   self.X_group[:, self.parents_indices[randint(self.nbr_parents, size=1)[0]]]
                for idim in xrange(self.ndimensions):
                    if rand() < self.mutation_rate[self.igroup]:
                        Xnew[idim, self.nbr_xover[self.igroup] + imut] = \
                            Xnew[idim, self.nbr_xover[self.igroup] + imut] +\
                            (self.Xmax[idim, 0] - self.Xmin[idim, 0]) * rand()
        return Xnew

    def include_elite(self, Xnew):
        ### add the current elite to the next  generation

        Xnew[:, self.nbr_xover[self.igroup] + self.nbr_mutation[self.igroup]:]\
                                                                = self.X_lElite
        return Xnew

    def get_result(self):

        best_position = list()
        best_fitness = list()
        for igroup in xrange(self.groups):
            if self.index == 0:
                if self.scaling == None:
                    best_position.append(self.X_best_of_all[igroup, :])
                    best_fitness.append(self.fitness_best_of_all[igroup])
                else:
                    best_position.append(self.parameters.\
                           unscaling_func(self.X_best_of_all[igroup, :]))
                    best_fitness.append(self.fitness_best_of_all[igroup])

            else:
                best_position, best_fitness = [], []
#        print best_position
        return (best_position, best_fitness)
