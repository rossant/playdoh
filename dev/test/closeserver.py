"""
Close Playdoh servers. Can be executed in command line like this:

    Usage: python closeserver.py server1:port1 server2:port2 ... [options]
    
    Options:
      -h, --help            show this help message and exit
      -p, --port            specify a port for all servers
  
A port can be specified for every server with the syntax ``IP:port``. Also,
the option --port allows to specify the same port for all servers.
"""
from playdoh import *
import sys, optparse

def main():
    MAXGPU = get_gpu_count()
        
    parser = optparse.OptionParser(usage = "usage: python closeserver.py server1 server2 ... [options]")
    parser.add_option("-p", "--port", dest="port", default=DEFAULT_PORT,
                      help="port (default: %s)" % DEFAULT_PORT, metavar="PORT")
    
    servers = []
    (options, args) = parser.parse_args()
    for arg in args:
        if ':' in arg:
            server, port = arg.split(':')
            port = int(port)
        else:
            server = arg
            port = int(options.port)
        servers.append((server, port))
    if len(servers) == 0: servers = ['localhost']
    log_info("Closing %d server(s)" % len(servers))
    close_servers(servers)
    
if __name__ == '__main__':
    main()
    