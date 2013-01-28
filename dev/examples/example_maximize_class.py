from playdoh import *
from numpy import exp, tile

class FitnessTest(Fitness):
    def initialize(self):
        self.a = self.shared_data['a']
    
    def evaluate(self, x):
        if self.dimension == 1:
            x = x.reshape((1,-1))
        result = self.a*exp(-(x**2).sum(axis=0))
        return result

if __name__ == '__main__':
    dimension = 2
    initrange = tile([-10,10], (dimension,1))
    results = maximize(FitnessTest,
                       popsize = 1000,
                       maxiter = 10,
                       cpu = 2,
                       shared_data = {'a': 3},
                       codedependencies = [],
                       initrange = initrange)
    print_table(results)
    