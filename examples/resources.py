"""
Resource allocation example showing how to allocate resources on the servers.
The Playdoh server must run on the local machine and on the default port
(2718 by default)
for this script to work.
"""
from playdoh import *


# It can also be a list of server IP addresses
servers = 'localhost'

# Get all the allocated resources on the servers
# total_resources[0]['CPU'] is a dictionary where keys are client IP
# addresses and values are the number of CPUs allocated to the corresponding
# clients
total_resources = get_server_resources(servers)
print "Total allocated resources:", total_resources[0]['CPU']

# Get the idle resources on the specified servers
# idle_resources[0]['CPU'] is the number of CPUs available on the first server
# This number includes the already allocated resources for this client
idle_resources = get_available_resources(servers)
print "%d idle CPUs" % idle_resources[0]['CPU']

# Get the resources allocated to this client on the specified servers
# my_resources['CPU'] is the number of CPUs allocated on the servers for this
# client
my_resources = get_my_resources(servers)
print "%d CPUs allocated to me" % my_resources[0]['CPU']

# Allocate as many CPUs as possible on the specified servers for this client
n = request_all_resources(servers, 'CPU')
print "Just allocated %d CPUs on the server" % n[0]

my_resources = get_my_resources(servers)
print "%d CPUs allocated to me now" % my_resources[0]['CPU']

total_resources = get_server_resources(servers)
print "Total allocated resources now:", total_resources[0]['CPU']
