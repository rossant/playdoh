from playdoh import *
import numpy as np
import pylab as pl
from scipy.sparse import lil_matrix

class HeatSolver(ParallelTask):
    """
    2D heat equation solver
    """
    def initialize(self, X, dx, dt, iterations):
        self.X = X # matrix with the function values and the boundary values
        # X must contain the borders of the neighbors ("overlapping Xs")
        self.n = X.shape[0]
        self.dx = dx
        self.dt = dt
        self.iterations = iterations
        self.iteration = 0

    def send_boundaries(self):
        if 'left' in self.tubes_out:
            self.push('left', self.X[:,1])
        if 'right' in self.tubes_out:
            self.push('right', self.X[:,-2])
    
    def recv_boundaries(self):
        if 'right' in self.tubes_in:
            self.X[:,0] = self.pop('right')
        if 'left' in self.tubes_in:
            self.X[:,-1] = self.pop('left')
    
    def update_matrix(self):
        """
        Implements the numerical scheme for the PDE
        """
        Xleft, Xright = self.X[1:-1,:-2], self.X[1:-1,2:]
        Xtop, Xbottom = self.X[:-2,1:-1], self.X[2:,1:-1]
        self.X[1:-1,1:-1] += self.dt*(Xleft+Xright+Xtop+Xbottom-4*self.X[1:-1,1:-1])/self.dx**2

    def start(self):
        for self.iteration in xrange(self.iterations):
            self.send_boundaries()
            self.recv_boundaries()
            self.update_matrix()
    
    def get_result(self):
        return self.X[1:-1,1:-1]
    
    def get_info(self):
        return self.iteration

def heat2d(n, nodes):
    # split is the grid size on each node, without the boundaries
    split = [(n-2)*1.0/nodes for _ in xrange(nodes)]
    split = np.array(split, dtype=int)
    split[-1] = n-2-np.sum(split[:-1])
    
    dx=2./n
    dt = dx**2*.2
    iterations = 500
    
    # dirac function at t=0
    y = np.zeros((n,n))
    y[n/2,n/2] = 1./dx**2
    
    # split y horizontally
    split_y = []
    j = 0
    for i in xrange(nodes):
        size = split[i]
        split_y.append(y[:,j:j+size+2])
        j += size
    
    # double linear topology 
    topology = []
    for i in xrange(nodes-1):
        topology.append(('right', i, i+1))
        topology.append(('left', i+1, i))
    
    # starts the task
    task = start_task(HeatSolver, # name of the task class
                      cpu = nodes, # use <nodes> CPUs on the local machine
                      topology = topology,
                      args=(split_y, dx, dt, iterations))
                                              
    # Retrieves the result, as a list with one element returned by MonteCarlo.get_result per node
    result = task.get_result()
    result = np.hstack(result)
    
    return result

if __name__ == '__main__':
    result = heat2d(100,2)
    pl.hot()
    pl.imshow(result)

#    x = np.linspace(-1.,1.,98)
#    X,Y= np.meshgrid(x,x)
#    import matplotlib.pyplot as plt
#    from matplotlib import cm
#    from mpl_toolkits.mplot3d import Axes3D
#    fig = pl.figure()
#    ax = fig.add_subplot(111, projection='3d')
#    ax.plot_wireframe(X,Y,result)

    pl.show()