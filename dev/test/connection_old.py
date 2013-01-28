from debugtools import *
from userpref import *
import multiprocessing, multiprocessing.connection, threading, logging
import os, sys, zlib, cPickle, time, traceback, gc, socket, base64, math, binascii, hashlib

BUFSIZE = 2048
try:
    LOCAL_IP = socket.gethostbyname(socket.gethostname())
except:
    LOCAL_IP = '127.0.0.1'

__all__ = ['accept', 'connect', 'LOCAL_IP']

class Connection(object):
    """
    Handles chunking and compression of data.
    
    To minimise data transfers between machines, we can use data compression,
    which this Connection handles automatically.
    """
    def __init__(self, conn, chunked=True, compressed=False):
        self.conn = conn
        self.chunked = chunked
        self.compressed = compressed
        self.BUFSIZE = BUFSIZE
        
    def send(self, obj):
        s = cPickle.dumps(obj, -1)
        if self.compressed:
            s = zlib.compress(s)
        if self.chunked:
            l = int(math.ceil(float(len(s))/self.BUFSIZE))
            # len(s) is a multiple of BUFSIZE, padding right with spaces
            s = s.ljust(l*self.BUFSIZE)
            l = "%08d" % l
            try:
                self.conn.sendall(l)
                self.conn.sendall(s)
            except:
                log_warn("Connection error")
        else:
            self.conn.sendall(s)
            
    def recv(self):
        if self.chunked:
            # Gets the first 8 bytes to retrieve the number of packets.
            l = ""
            n = 8
            while n > 0:
                l += self.conn.recv(n)
                n -= len(l)
            # BUG: sometimes l is filled with spaces??? setting l=1 in this case
            # (not a terrible solution)
            try:
                l = int(l)
            except:
                log_warn("transfer error, the paquet size was empty")
                l = 1
            
            length = l*self.BUFSIZE
            s = ""
            # Ensures that all data have been received
            while len(s) < length:
                data = self.conn.recv(self.BUFSIZE)
                s += data
        else:
            s = self.conn.recv()
        # unpads spaces on the right
        s = s.rstrip()
        if self.compressed:
            s = zlib.decompress(s)
        return cPickle.loads(s)
    
#    def recv(self):
#        for i in xrange(5):
#            try:
#                data = self._recv()
#                break
#            except Exception as e:
#                if i==4: raise Exception("Connection error")
#                log_warn("Connection error: %s" % str(e))
#                time.sleep(.1*(i+1))
#        return data
    
    def close(self):
        if self.conn is not None:
            r = self.conn.close()
            self.conn = None




def accept(address):
    """
    Accepts a connection and returns a connection object.
    """
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        for i in xrange(5):
            try:
    #            log_debug("trying to bind the socket...")
                s.bind(address)
                s.listen(5)
    #            log_debug("the socket is now listening")
                break
            except:
                if i<4:
                    t = .25*2**i
                    log_debug("unable to bind the socket, trying again in %.2f seconds..." % t)
                    time.sleep(t)
                else:
                    log_debug("unable to bind the socket")
                    raise Exception("unable to bind the socket")
        try:
            conn, addr = s.accept()
        except:
            raise Exception("unable to accept incoming connections")
        conn = Connection(conn)
        
        auth = conn.recv()
        if auth == hashlib.md5(USERPREF['authkey']).hexdigest():
            conn.send('right authkey')
            break
        else:
            log_warn("Wrong authkey, listening to new connection")
            conn.send('wrong authkey')
            continue
        s.close()
        time.sleep(.1)
        
    # The client can send its id at each connection, otherwise it can be retrieved
    # from the socket.accept() function.
#    clientid = conn.recv()
#    if clientid is None:
#        clientid = addr[0]
#    log_debug("client address: %s" % clientid)
    
    return conn, addr[0]

def connect(address):
    """
    Connects to a server and returns a Connection object.
    """
    def _create_connection():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for i in xrange(5):
            try:
                s.connect(address)
                break
            except:
                if i<4:
                    t = .1*2**i
                    log_debug("client: unable to connect, trying again in %.2f seconds..." % t)
                    time.sleep(t)
                else:
                    msg = "unable to connect to '%s' on port %d" % address
                    log_warn(msg)
                    raise Exception(msg)
                    return None
        conn = Connection(s)
        return conn

    hash = hashlib.md5(USERPREF['authkey']).hexdigest()
    
    for i in xrange(4):
        try:
            conn = _create_connection()
            conn.send(hash)
            resp = conn.recv()
            break
        except Exception as e:
            log_warn("Connection error: %s, trying again... (%d/4)" % (str(e), i+1))
            time.sleep(.1*2**i)
    
    if resp == 'wrong authkey':
        raise Exception("Wrong authentication key")
    
    # sends the client id to the server
#    conn.send(clientid)
    time.sleep(.05) # waits a bit for the server to be ready to receive
    return conn