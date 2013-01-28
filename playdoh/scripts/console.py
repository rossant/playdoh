#!/usr/bin/env python

from playdoh import *
import os
import optparse
import playdoh
import time


def _open_server(cpu, gpu, port, background=True):
    scriptdir = os.path.join(os.path.dirname(playdoh.__file__),
                                             'scripts')
    cmd = "python \"%s\" %d %d %d" % (os.path.join(scriptdir, 'open.py'),
                                                   cpu, gpu, port)

    # run in background:
    if background:
        if os.name == 'nt':
            cmd = 'start ' + cmd
        else:
            cmd = cmd + ' &'

    os.system(cmd)
    time.sleep(1)


def _open_restart_server():
    scriptdir = os.path.join(os.path.dirname(playdoh.__file__), 'scripts')
    os.system("python %s" % (os.path.join(scriptdir, 'openrestart.py')))


def get_units(args):
    try:
        nbr = int(args[0])
    except:
        nbr = -1  # means ALL
    type = args[1][:3].upper()
    if type not in ['CPU', 'GPU']:
        log_warn("the unit type must be either CPU or GPU instead of %s,\
            using CPU now" % type)
        type = 'CPU'
    return nbr, type


def get_server(string):
    if ':' in string:
        server, port = string.split(':')
        port = int(port)
    else:
        server = string
        port = DEFAULT_PORT
    return (server, port)


def parse_open(args):
    # get the total number of GPUs on this machine
    MAXGPU = get_gpu_count()

    cpu = MAXCPU
    gpu = MAXGPU
    port = DEFAULT_PORT

    if len(args) <= 1:
        if len(args) == 1:
            port = int(args[0])
    elif len(args) <= 3:
        nbr, type = get_units(args)
        if type == 'CPU':
            cpu = nbr
        if type == 'GPU':
            gpu = nbr
        if len(args) == 3:
            port = int(args[2])
    elif len(args) <= 5:
        nbr, type = get_units(args[:2])
        if type == 'CPU':
            cpu = nbr
        if type == 'GPU':
            gpu = nbr
        nbr, type = get_units(args[2:4])
        if type == 'CPU':
            cpu = nbr
        if type == 'GPU':
            gpu = nbr
        if len(args) == 5:
            port = int(args[4])
    return cpu, gpu, port


def parse_close(args):
    servers = []
    for arg in args:
        servers.append(get_server(arg))
    if len(servers) == 0:
        servers = [('localhost', DEFAULT_PORT)]
    return servers


def parse_get(args):
    if len(args) == 0:
        return None
    server, port = get_server(args[0])

    # only the server = get idle resources
    if len(args) == 1:
        resources = get_available_resources((server, port))
        my_resources = get_my_resources((server, port))
        for type in ['CPU', 'GPU']:
            print "%d %s(s) available, %d allocated, on %s" %\
                (resources[0][type], type, my_resources[0][type], server)
    elif len(args) == 2:
        if args[1] == 'all':
            # get all resources
            resources = get_server_resources((server, port))[0]
            for type in ['CPU', 'GPU']:
                if resources[type] is None:
                    print "No %ss available" % type
                    continue
                if len(resources[type]) == 0:
                    print "No %ss allocated" % type
                for client in resources[type]:
                    print "%d %s(s) allocated to %s" %\
                        (resources[type][client], type, client.lower())
    return resources


def parse_request(args):
    if len(args) == 0:
        return None
    server, port = get_server(args[0])

    params = {}
    paramsall = []  # types for which allocate all resources
    nbr, type = get_units(args[1:3])
    if nbr >= 0:
        params[type] = nbr
    else:
        paramsall.append(type)
    if len(args) > 3:
        nbr2, type2 = get_units(args[3:5])
        if nbr2 >= 0:
            params[type2] = nbr2
        else:
            paramsall.append(type2)

    # request some resources
    if len(params) > 0:
        request_resources((server, port), **params)

    # request all resources
    for type in paramsall:
        request_all_resources((server, port), type)

    resources = get_my_resources((server, port))[0]
    keys = resources.keys()
    keys.sort()
    for type in keys:
        print "%d %s(s) allocated to you on %s" % (resources[type],
                                                             type, server)

    return resources


def parse_set(args):
    if len(args) == 0:
        return None
    server, port = ('localhost', DEFAULT_PORT)
    params = {}
    if len(args) >= 2:
        nbr, type = get_units(args[:2])
        params[type] = nbr
    if len(args) >= 4:
        nbr, type = get_units(args[2:4])
        params[type] = nbr
    if 'CPU' in params.keys():
        if params['CPU'] == -1:
            params['CPU'] = get_cpu_count()
    if 'GPU' in params.keys():
        if params['GPU'] == -1:
            params['GPU'] = get_gpu_count()
    set_total_resources((server, port), **params)
    res = get_total_resources((server, port))[0]
    for type in ['CPU', 'GPU']:
        print "%d total %s(s) available on this machine" % (res[type], type)
    return True


def run_console():
    usage = """
    This tool allows you to open/close a server, obtain the available
    resources on distant servers and allocate resources. Here are a
    few usage examples:

        # open the server with all possible resources
        playdoh open

        # open the server with 4 CPUs and 1 GPU
        playdoh open 4 CPU 1 GPU

        # change the total number of resources available on this machine
        playdoh set 2 CPUs 0 GPU

        # show the available resources/all resources on the given server
        playdoh get bobs-machine.university.com [all]

        # request 2 CPUs and 1 GPU on the server
        playdoh request bobs-machine.university.com 2 CPUs 1 GPU

        # request all resources on the server
        playdoh request bobs-machine.university.com all CPUs all GPUs

        # close the server on this machine
        playdoh close

        # close a server remotely
        playdoh close bobs-machine.university.com
    """

    parser = optparse.OptionParser(usage=usage)

    parser.add_option("-b", "--background",
                          dest="background",
                          default="True",
                          help="open the server in background")

    (options, args) = parser.parse_args()

    if len(args) == 0:
        parser.print_help()
    elif args[0] == 'open':
        cpu, gpu, port = parse_open(args[1:])
        _open_server(cpu, gpu, port, options.background == "True")
        # open_server(maxcpu=cpu, maxgpu=gpu, port=port)
    elif args[0] == 'close':
        servers = parse_close(args[1:])
        log_info("Closing %d server(s)" % len(servers))
        close_servers(servers)
    elif args[0] == 'getall':
        print "%d,%d" % (get_cpu_count(), get_gpu_count())
    elif args[0] == 'get':
        if parse_get(args[1:]) is None:
            parser.print_help()
    elif args[0] == 'request':
        if parse_request(args[1:]) is None:
            parser.print_help()
    elif args[0] == 'set':
        if parse_set(args[1:]) is None:
            parser.print_help()
    elif args[0] == 'openrestart':
        # start the restart server, always running on port 27182,
        # allowing to kill/restart/start
        # the Playdoh server remotely
        _open_restart_server()
    elif args[0] == 'remote':
        """
        Open remotely a Playdoh server
            playdoh remote open localhost

        Kill remotely a Playdoh server, even if it is blocked
            playdoh remote kill localhost

        Kill and open a new Playdoh server
            playdoh remote restart localhost
        """
        server = procedure = None
        if len(args) >= 2:
            procedure = args[1]
        if len(args) >= 3:
            server = args[2]
        restart(server, procedure)
    else:
        parser.print_help()


if __name__ == '__main__':
    run_console()
