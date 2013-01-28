from playdoh import *
from test import *
import numpy, sys, time
from numpy import int32, ceil

def fitness_fun(x):
    if x.ndim == 1:
        x = x.reshape((1,-1))
    result = numpy.exp(-(x**2).sum(axis=0))
    return result

if __name__ == '__main__':
    nlocal = 2
    
    # List of machines external IP addresses
    machines = []
    local_machines = [('localhost', 2718+i) for i in xrange(nlocal)]
    machines.extend(local_machines)
    
    for m in local_machines:
        Process(target=open_server, args=(m[1],1,0)).start()
        time.sleep(.2)
        
    # State space dimension (D)
    dimension = 10
    
    # ``initrange`` is a Dx2 array with the initial intervals for every dimension 
    initrange = numpy.tile([-10.,10.], (dimension,1))
    
    result = maximize(fitness_fun,
                      algorithm = CMAES,
                      maxiter = 100,
                      popsize = 1000,
                      machines = machines,
                      initrange = initrange)
    
    time.sleep(.2)
    close_servers(machines)
    
    print result.best_pos

