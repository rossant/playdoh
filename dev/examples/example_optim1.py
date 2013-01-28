
from playdoh import *
from multiprocessing import Process
from threading import Thread
import sys, time, unittest, numpy
from numpy import inf, ones, zeros, exp,sin
from numpy.random import rand
from pylab import *
from time import sleep


test_fun=4
if test_fun==1:
#sphere  
    def fun(x1,x2,x3,x4,x5):
        return x1**2+x2**2+x3**2+x4**2+x5**2
    min_dom=-5.12
    max_dom=5.12

if test_fun==2:
##schwefel  solution  (-420.9687....)
    def fun(x1,x2,x3,x4,x5):
        from numpy import sqrt,sin,abs
        return 418.9829*5+x1*sin(sqrt(abs(x1)))+x2*sin(sqrt(abs(x2)))+x3*sin(sqrt(abs(x3)))+x4*sin(sqrt(abs(x4)))+x5*sin(sqrt(abs(x5)))
    min_dom=-512.03
    max_dom=511.97
if test_fun==3:
#Rastrigin solution (0,0,0...)
    def fun(x1,x2,x3,x4,x5):
        from numpy import cos,pi
        return 10.*+x1**2-10*cos(2*pi*x1)+x2**2-10*cos(2*pi*x2)+x3**2-10*cos(2*pi*x3)+x4**2-10*cos(2*pi*x4)+x5**2-10*cos(2*pi*x5)
    min_dom=-5.12
    max_dom=5.12
if test_fun==4:
#Rosenbrock solution (1,1,1...)
    def fun(x1,x2,x3,x4,x5):
        sleep(.1)
        return 100*(-x2+x1**2)**2+(x1-1)**2+100*(-x3+x2**2)**2+(x2-1)**2+100*(-x4+x3**2)**2+(x3-1)**2+100*(-x5+x4**2)**2+(x4-1)**2
    min_dom=-2.048
    max_dom=2.048
    
if test_fun==5:
#Ackley solution (0.0.0.0...)
    def fun(x1,x2,x3,x4,x5):
        from numpy import exp,sqrt
        return 20+exp(1)-20*exp(-0.2*sqrt(0.2*(x1**2+x2**2+x3**2+x4**2+x5**2)))
    min_dom=-2.048
    max_dom=2.048
    
if test_fun==6:   
    def fun(x1,x2,x3,x4,x5):
        from numpy import exp
    #    print x1
        return 1-exp(-(x1*x1+x2*x2+x3*x3+x4*x4+x5*x5))
    min_dom=-2.
    max_dom=2.


if __name__ == '__main__':
    
    nbr_iterations=50
    
    nbr_particles=1000
    nbr_cpu=2
    
    scale_dom=2./3
    optCMA=dict()
    optCMA['proportion_selective']=0.5
    optCMA['returninfo']=True
    result_CMA = minimize(fun,
                      algorithm = CMAES,
                      maxiter = nbr_iterations,
                      popsize = nbr_particles,  
                      groups=2,
                     scaling='mapminmax',
                      cpu = nbr_cpu,
                    ##  x1 = [min_dom,3./4*min_dom,3./4*max_dom,max_dom], x2 =[min_dom,3./4*min_dom,3./4*max_dom,max_dom],x3 = [min_dom,3./4*min_dom,3./4*max_dom,max_dom],x4 =[min_dom,3./4*min_dom,3./4*max_dom,max_dom],x5 =[min_dom,3./4*min_dom,3./4*max_dom,max_dom])
                      x1 = [min_dom,scale_dom*min_dom,scale_dom*max_dom,max_dom], x2 =[min_dom,scale_dom*min_dom,scale_dom*max_dom,max_dom],x3 = [min_dom,scale_dom*min_dom,scale_dom*max_dom,max_dom],x4 =[min_dom,scale_dom*min_dom,scale_dom*max_dom,max_dom],x5 =[min_dom,scale_dom*min_dom,scale_dom*max_dom,max_dom],
                      returninfo=True,
                      #optparams=optCMA
                      )
        
    print result_CMA[0].best_pos,result_CMA[1].best_pos
    plot( result_CMA[0].info['best_fitness'])
    plot( result_CMA[1].info['best_fitness'])
    show()
