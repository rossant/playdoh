
from playdoh import *
from multiprocessing import Process
from threading import Thread
import sys, time, unittest, numpy
from numpy import inf, ones, zeros, exp,sin
from numpy.random import rand
from pylab import *

test_fun=2
if test_fun==1:
#sphere  
    def fun(x):
        return sum(x**2, axis=0)
    min_dom=-5.12
    max_dom=5.12

if test_fun==2:
##schwefel  solution  (-420.9687....)
    def fun(x):
        from numpy import sqrt,sin,abs
        return 418.9829*x.shape[0]+sum(x*sin(sqrt(abs(x))), axis=0)
    min_dom=-512.03
    max_dom=511.97
if test_fun==3:
#Rastrigin solution (0,0,0...)
    def fun(x):
        from numpy import cos,pi
        return 10+sum(x**2-10*cos(2*pi*x), axis=0)
    min_dom=-5.12
    max_dom=5.12





if __name__ == '__main__':
    
    nbr_iterations=200   
    nbr_particles=200
    nbr_cpu=2  
    scale_dom=2./3
    nbr_dim=5
    initrange=hstack((scale_dom*min_dom*ones((nbr_dim,1)),scale_dom*max_dom*ones((nbr_dim,1))))
    bounds=hstack((min_dom*ones((nbr_dim,1)),max_dom*ones((nbr_dim,1))))
    print initrange
    optCMA=dict()
    optCMA['proportion_selective']=0.5
    result_CMA = minimize(fun,
                      algorithm = CMAES,
                      maxiter = nbr_iterations,
                      popsize = nbr_particles,  
                      #scaling='mapminmax',
                      groups=2,
                      cpu = nbr_cpu,
                      initrange=initrange,
                      bounds=bounds,
                      returninfo=True,
                      #optparams=optCMA
                      )

    print result_CMA[0].best_pos
    



