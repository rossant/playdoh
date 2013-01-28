"""
Start a Playdoh server. Can be executed in command line like this:

    Usage: python openserver.py [options]
    
    Options:
      -h, --help            show this help message and exit
      -c CPU, --cpu=CPU     number of CPUs (default: MAXCPU)
      -g GPU, --gpu=GPU     number of GPUs (default: MAXGPU)
      -p PORT, --port=PORT  port (default: 2718)
"""
from playdoh import *
import sys, optparse

def main(port=None, maxcpu=None, maxgpu=None):
    MAXGPU = get_gpu_count()
        
    parser = optparse.OptionParser(usage = "usage: python server.py [options]")
    parser.add_option("-c", "--cpu", dest="cpu", default=MAXCPU,
                      help="number of CPUs (default: MAXCPU)", metavar="CPU")
    parser.add_option("-g", "--gpu", dest="gpu", default=MAXGPU,
                      help="number of GPUs (default: MAXGPU)", metavar="GPU")
    parser.add_option("-p", "--port", dest="port", default=DEFAULT_PORT,
                      help="port (default: %s)" % DEFAULT_PORT, metavar="PORT")
    
    (options, args) = parser.parse_args()
    
    if port is None: port = int(options.port)
    if maxcpu is None: maxcpu = int(options.cpu)
    if maxgpu is None: maxgpu = int(options.gpu)
    
    open_server(port=port,
                maxcpu=maxcpu,
                maxgpu=maxgpu)
    
if __name__ == '__main__':
    main(maxcpu=4,port=2718)
    