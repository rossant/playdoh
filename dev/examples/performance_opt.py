from playdoh import *
from multiprocessing import Process
from threading import Thread
import sys, time, unittest, numpy
from numpy import inf, ones, zeros, exp,sin
from numpy.random import rand
from pylab import *


def alloc(n):
    allocation = {}
#    machines = [(LOCAL_IP, 2718),
#            ('LOCAL_IP', 2718),
#            ('LOCAL_IP', 2718)]
    machines = [(LOCAL_IP, 2718),
                ('129.199.82.3', 2718),
                ('129.199.82.36', 2718)]
    maxcpu = 2
    if n<=maxcpu:
        allocation[machines[0]] = n
    elif n<=2*maxcpu:
        allocation[machines[0]] = maxcpu
        allocation[machines[1]] = n-maxcpu
    elif n<=3*maxcpu:
        allocation[machines[0]] = maxcpu
        allocation[machines[1]] = maxcpu
        allocation[machines[2]] = n-2*maxcpu
    return allocation


test_fun=1

if test_fun==1:
##schwefel  solution  (-420.9687....)


    def fun(a,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10):
        from numpy import sqrt,sin,abs,zeros
        import time
        output=zeros(len(x1))

        for isample in xrange(len(x1)):
            output[isample]=418.9829*10+x1[isample]*sin(sqrt(abs(x1[isample])))+x2[isample]*sin(sqrt(abs(x2[isample])))+x3[isample]*sin(sqrt(abs(x3[isample])))+x4[isample]*sin(sqrt(abs(x4[isample])))+x5[isample]*sin(sqrt(abs(x5[isample])))+x6[isample]*sin(sqrt(abs(x6[isample])))+x7[isample]*sin(sqrt(abs(x7[isample])))+x8[isample]*sin(sqrt(abs(x8[isample])))+x9[isample]*sin(sqrt(abs(x9[isample])))+x10[isample]*sin(sqrt(abs(x10[isample])))
            #print a 
            time.sleep(a)

        return output
    min_dom=-512.03
    max_dom=511.97


if test_fun==2:
#Rosenbrock solution (1,1,1...)
    def fun(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10):

        return 100*(-x2+x1**2)**2+(x1-1)**2+100*(-x3+x2**2)**2+(x2-1)**2+100*(-x4+x3**2)**2+(x3-1)**2+100*(-x5+x4**2)**2+(x4-1)**2+100*(-x6+x5**2)**2+(x5-1)**2+100*(-x7+x6**2)**2+(x6-1)**2+100*(-x8+x7**2)**2+(x7-1)**2+100*(-x9+x8**2)**2+(x8-1)**2
    min_dom=-2.048
    max_dom=2.048
            
if __name__ == '__main__':
    
    nbr_iterations=100
    
    nbr_particles=50
    nbr_cpu=array([1,2])
    
    scale_dom=0.75
    
    pause_values=array([0.000001,0.00001,0.0001])

    time_PSO=zeros((len(nbr_cpu),len(pause_values)))
    time_GA=zeros((len(nbr_cpu),len(pause_values)))
                  
    test_fun=1
    for itopology in xrange(len(nbr_cpu)):
        for ipause in xrange(len(pause_values)):
    
    
            t0 = time.time()
            result_PSO = minimize(fun,
                          algorithm = PSO,
                          ndimensions = 10,
                          other_param=pause_values[ipause],
                              niterations = nbr_iterations,
                              nparticles = nbr_particles,  
                        #  scaling='mapminmax',
                          cpu = nbr_cpu[itopology],
                        ##  x1 = [min_dom,3./4*min_dom,3./4*max_dom,max_dom], x2 =[min_dom,3./4*min_dom,3./4*max_dom,max_dom],x3 = [min_dom,3./4*min_dom,3./4*max_dom,max_dom],x4 =[min_dom,3./4*min_dom,3./4*max_dom,max_dom],x5 =[min_dom,3./4*min_dom,3./4*max_dom,max_dom])
                          x1 = [min_dom,min_dom,max_dom,max_dom], x2 =[min_dom,min_dom,max_dom,max_dom],x3 = [min_dom,min_dom,max_dom,max_dom],x4 =[min_dom,min_dom,max_dom,max_dom],x5 =[min_dom,min_dom,max_dom,max_dom],
                                            x6 = [min_dom,min_dom,max_dom,max_dom], x7 =[min_dom,min_dom,max_dom,max_dom],x8 = [min_dom,min_dom,max_dom,max_dom],x9 =[min_dom,min_dom,max_dom,max_dom],x10 =[min_dom,min_dom,max_dom,max_dom],
        
                              returninfo=True
                              #allocation=alloc(nbr_cpu)
                              )
            time_PSO[itopology,ipause]= time.time()-t0
      
        
            t0 = time.time()
            result_GA = minimize(fun,
                          algorithm = GA,
                          other_param=pause_values[ipause],
                          ndimensions = 10,
                              niterations = nbr_iterations,
                              nparticles = nbr_particles,  
                          cpu = nbr_cpu[itopology],
                          x1 = [min_dom,min_dom,max_dom,max_dom], x2 =[min_dom,min_dom,max_dom,max_dom],x3 = [min_dom,min_dom,max_dom,max_dom],x4 =[min_dom,min_dom,max_dom,max_dom],x5 =[min_dom,min_dom,max_dom,max_dom],
                          x6 = [min_dom,min_dom,max_dom,max_dom], x7 =[min_dom,min_dom,max_dom,max_dom],x8 = [min_dom,min_dom,max_dom,max_dom],x9 =[min_dom,min_dom,max_dom,max_dom],x10 =[min_dom,min_dom,max_dom,max_dom],                          
                          returninfo=True)
        
            time_GA[itopology,ipause]= time.time()-t0
        
    figure(2)
    plot(pause_values,time_PSO[0,:],'-o')
    plot(pause_values,time_PSO[1,:],'-o')
    plot(pause_values,time_GA[0,:],'--o')
    plot(pause_values,time_GA[1,:],'--o')
   # print 'Solution:  CMA: ',result_CMA[0]#,'PSO: ',result_PSO[0]#,'GA: ',result_GA[0]
   # plot(fitness_CMA)
#    plot(fitness_PSO)
#    print result_PSO[0]
#        plot(fitness_GA)


    show()

