"""
Monte Carlo simulation example of pi estimation.
This example shows how to use the Playdoh interface
to execute loosely coupled parallel tasks.
"""
from playdoh import *
import numpy as np

# Any task class must derive from the ParallelTask
class PiMonteCarlo(ParallelTask): 
    def initialize(self, n):
        # Specify the number of samples on this node
        self.n = n
    
    def start(self):
        # Draw n points uniformly in [0,1]^2
        samples = np.random.rand(2,self.n)
        # Count the number of points inside the quarter unit circle
        self.count = np.sum(samples[0,:]**2+samples[1,:]**2<1)
    
    def get_result(self):
        # Return the result
        return self.count
    
def pi_montecarlo(samples, machines):
    allocation = allocate(machines=machines, unit_type='CPU')
    nodes = len(allocation)
    # Calculate the number of samples for each node
    split_samples = [samples/nodes]*nodes
    # Launch the task on the local CPUs
    task = start_task(PiMonteCarlo, # name of the task class
                      allocation = allocation,
                      args=(split_samples,)) # arguments of MonteCarlo.initialize as a list, 
                                             # node #i receives split_samples[i] as argument
    # Retrieve the result, as a list with one element returned by MonteCarlo.get_result per node
    result = task.get_result()
    # Return the estimation of Pi
    return sum(result)*4.0/samples

if __name__ == '__main__':
    machines = ['localhost'
                ]

    nlocal = 2
    
    # List of machines external IP addresses
    machines = []
    local_machines = [('localhost', 2718+i) for i in xrange(nlocal)]
    machines.extend(local_machines)
    
    for m in local_machines:
        Process(target=open_server, args=(m[1],1,0)).start()
    
    result = pi_montecarlo(1000000, machines)
    print result
    
    close_servers(local_machines)
    