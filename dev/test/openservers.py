from playdoh import *
import os, threading

def open(n, port):
    os.system("python openserver.py -c %d -g 0 -p %d" % (n, 2718+port))

n = 6
cpus = [1,1,1,1,1,2]
for port in xrange(n):
    cpu = cpus[port]
    t = threading.Thread(target=open, args=(cpu, port))
    t.start()

