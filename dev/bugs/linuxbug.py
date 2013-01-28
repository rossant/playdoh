from playdoh import *
from test import *
from multiprocessing import Process
from numpy.random import rand
from numpy import max, mean
import time, sys

iterations = 500

class TaskTest(ParallelTask):
    def initialize(self, cpu):
        self.cpu = cpu
    
    def start(self):
        for i in xrange(iterations):
            print i
            sys.stdout.flush()
            if self.node.index==0:
                [self.tubes.push('tube%d' % j, rand(100,100)) for j in xrange(1, self.cpu)]
                [self.tubes.pop('tube%dbis' % j) for j in xrange(1, self.cpu)]
            for index in xrange(1,self.cpu):
                if self.node.index==index:
                    self.tubes.pop('tube%d' % index)
                    self.tubes.push('tube%dbis' % index, rand(100,100))
    
    def get_result(self):
        return None
    


if __name__ == '__main__':

    cpu = MAXCPU-1
    topology = []
    for i in xrange(1,cpu):
        topology.extend([('tube%d' % i, 0, i),
                         ('tube%dbis' % i, i, 0)])
        
    for i in xrange(1):
        print i
        task = start_task(TaskTest, topology = topology, 
                          cpu = cpu,
                          args=(cpu,))
        #time.sleep(2)
        result = task.get_result()





