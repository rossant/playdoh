from playdoh import *
from numpy import *
from numpy.random import rand
from pylab import *
import time

class ConnectionError(ParallelTask):
    def initialize(self, iterations):
        self.iterations = iterations
        self.iteration = 0

    def send_boundaries(self):
        if 'left' in self.tubes_out:
            self.push('left', None)
        if 'right' in self.tubes_out:
            self.push('right', None)
    
    def recv_boundaries(self):
        if 'right' in self.tubes_in:
            self.pop('right')
        if 'left' in self.tubes_in:
            self.pop('left')

    def start(self):
        for self.iteration in xrange(self.iterations):
            log_info("Iteration %d/%d" % (self.iteration+1, self.iterations))
            self.send_boundaries()
            self.recv_boundaries()
    
    def get_result(self):
        return None

if __name__ == '__main__':
    n = 6

    for i in xrange(n):
        p = Process(target=open_server, args=(2718+i,1,0))
        p.start()
    
#    machines = []
    machines = [('localhost',2718+i) for i in xrange(n)]
#    machines.append(('192.168.1.59', 2718))
    
    allocation = allocate(machines=machines)
    nodes = len(allocation)
    
    topology = []
    for i in xrange(nodes-1):
        topology.append(('right', i, i+1))
        topology.append(('left', i+1, i))
    
    task = start_task(ConnectionError,
                      topology=topology,
                      allocation=allocation,
                      args=(20,))

    result = task.get_result()
    
    
    close_servers(machines)
    