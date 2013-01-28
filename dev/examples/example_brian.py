"""
Brian example
"""
import playdoh
from numpy import *
from brian import *

def fun(taum):
    taum *= ms
    
    taue = 5 * ms
    taui = 10 * ms
    Vt = -50 * mV
    Vr = -60 * mV
    El = -49 * mV
    
    eqs = Equations('''
    dv/dt  = (ge+gi-(v-El))/taum : volt
    dge/dt = -ge/taue : volt
    dgi/dt = -gi/taui : volt
    ''')
    
    P = NeuronGroup(4000, model=eqs, threshold=Vt, reset=Vr, refractory=5 * ms)
    P.v = Vr
    P.ge = 0 * mV
    P.gi = 0 * mV
    
    Pe = P.subgroup(3200)
    Pi = P.subgroup(800)
    we = (60 * 0.27 / 10) * mV # excitatory synaptic weight (voltage)
    wi = (-20 * 4.5 / 10) * mV # inhibitory synaptic weight
    Ce = Connection(Pe, P, 'ge', weight=we, sparseness=0.02)
    Ci = Connection(Pi, P, 'gi', weight=wi, sparseness=0.02)
    P.v = Vr + rand(len(P)) * (Vt - Vr)
    
    # Record the number of spikes
    Me = PopulationSpikeCounter(Pe)
    Mi = PopulationSpikeCounter(Pi)
    # A population rate monitor
    M = PopulationRateMonitor(P)
    
    run(1 * second)
    
    return Me.nspikes, Mi.nspikes

if __name__ == '__main__':
    taums = linspace(5, 25, 2)
    
    import time
    t1 = time.clock()
    result = playdoh.map(fun, [i for i in taums], cpu=2)
    d = time.clock()-t1
    
    print result
    print "simulation last %.2f seconds" % d