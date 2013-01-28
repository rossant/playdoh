"""
Command-line tool for resource allocation. Can be executed in command line like this:

    Usage: python allocate.py server nbr type [options]
    
    Arguments:
      server                the IP address of the Playdoh server
      nbr                   the number of resources to allocate to this client
      type                  the resource type: ``'CPU'`` or ``'GPU'``
    
    Options:
      -h, --help            show this help message and exit
      -p PORT, --port=PORT  port (default: 2718)
"""
from playdoh import *
import sys, optparse

def main():
    parser = optparse.OptionParser(usage = "usage: python allocate.py server nbr type")
#    parser.add_option("-c", "--cpu", dest="cpu", default=MAXCPU,
#                      help="number of CPUs (default: MAXCPU)", metavar="CPU")
#    parser.add_option("-g", "--gpu", dest="gpu", default=MAXGPU,
#                      help="number of GPUs (default: MAXGPU)", metavar="GPU")
    parser.add_option("-p", "--port", dest="port", default=DEFAULT_PORT,
                      help="port (default: %s)" % DEFAULT_PORT, metavar="PORT")
    
    (options, args) = parser.parse_args()
    if len(args)==0:
        print "You must specify the server IP address"
        return
    server = args[0]
    port = int(options.port)
    # only the server = get idle resources
    if len(args) == 1:
        resources = get_available_resources((server, port))
        my_resources = get_my_resources((server, port))
        for type in ['CPU', 'GPU']:
            print "%d %s(s) available, %d allocated, on %s" % (resources[0][type], type, my_resources[0][type], server)
    else:
        nbr = int(args[1])
        type = args[2][:3].upper()
        request_resources((server, port), **{type: nbr})
        resources = get_my_resources((server, port))
        print "%d %s(s) allocated to you on %s" % (resources[0][type], type, server)
    
if __name__ == '__main__':
    main()
    