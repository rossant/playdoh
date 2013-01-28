'''
Model fitting example using several machines.
Before running this example, you must start the Playdoh server on the remote machines.
'''
from brian import loadtxt, ms, Equations
from brian.library.modelfitting import *
from multiprocessing import Process
from playdoh import *

if __name__ == '__main__':
    nlocal = 2
    
    # List of machines external IP addresses
    machines = []
    local_machines = [('localhost', 2718+i) for i in xrange(nlocal)]
    machines.extend(local_machines)
    
    for m in local_machines:
        Process(target=open_server, args=(m[1],1,0)).start()
    
    equations = Equations('''
        dV/dt=(R*I-V)/tau : 1
        I : 1
        R : 1
        tau : second
    ''')
    input = loadtxt('current.txt')
    spikes = loadtxt('spikes.txt')
    results = modelfitting( model = equations,
                            reset = 0,
                            threshold = 1,
                            data = spikes,
                            input = input,
                            dt = .1*ms,
                            popsize = 1000,
                            maxiter = 5,
                            delta = 4*ms,
                            unit_type = 'CPU',
                            machines = machines,
                            R = [1.0e9, 9.0e9],
                            tau = [10*ms, 40*ms],
                            refractory = [0*ms, 10*ms])
    print_table(results)


    close_servers(local_machines)