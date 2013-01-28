from playdoh import *
from numpy import *
from expfun import fun

if __name__ == '__main__':
    dimension = 4
    initrange = tile([-10,10], (dimension,1))
    results = maximize(fun,
                       popsize = 10000,
                       maxiter = 10,
                       cpu = 1,
                       machines = ['localhost'],
                       codedependencies = ['expfun.py', 'expfun2.py'],
                       initrange = initrange)
    print_table(results)